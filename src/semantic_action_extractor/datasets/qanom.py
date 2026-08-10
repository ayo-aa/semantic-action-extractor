"""Strict, dependency-free adapter for the released QANom CSV files.

QANom stores one row per question-answer annotation and repeats predicate- and
sentence-level values across those rows.  This adapter reconstructs one
``AnnotationRecord`` per sentence without collapsing candidate nouns, source
workers, question rows, or alternative answers.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any

from ..annotation_schema import (
    AnnotationProvenance,
    AnnotationRecord,
    AnswerAlternative,
    EventivityJudgment,
    PredicateCandidate,
    QASRLQuestion,
    QASRLQuestionSlots,
    QuestionJudgment,
)
from .common import (
    DatasetFormatError,
    canonicalize_pretokenized_text,
    infer_qasrl_document_id,
    normalize_slot,
    parse_bool,
    parse_int,
    require_string,
    split_sources,
    token_aligned_span,
)


QANOM_RELEASE = "2bce70e8a39b40157ba97f38e1a8ae7619b30162"
QANOM_ADAPTER_VERSION = "0.2.0"
QANOM_SPLITS = frozenset({"train", "dev", "test"})
QANOM_SEPARATOR = "~!~"

# The verified release has split-specific headers.  Train contains these 22
# semantic fields.  Development additionally contains ``source_assign_id``.
# The test CSV contains an accidental saved-index column named ``Unnamed: 0``.
QANOM_SHARED_FIELDS = frozenset(
    {
        "qasrl_id",
        "sentence",
        "target_idx",
        "key",
        "noun",
        "worker_id",
        "is_verbal",
        "verb_form",
        "question",
        "answer_range",
        "answer",
        "wh",
        "subj",
        "obj",
        "obj2",
        "aux",
        "prep",
        "verb_prefix",
        "is_passive",
        "is_negated",
        "source_worker_ids",
        "verb_slot_inflection",
    }
)
QANOM_OPTIONAL_FIELDS = frozenset({"source_assign_id"})
QANOM_RELEASE_ARTIFACT_FIELDS = frozenset({"Unnamed: 0"})

_CANDIDATE_CONSTANT_FIELDS = (
    "qasrl_id",
    "sentence",
    "target_idx",
    "key",
    "noun",
    "worker_id",
    "is_verbal",
    "verb_form",
)
_QUESTION_TEXT_FIELDS = (
    "answer_range",
    "answer",
    "wh",
    "subj",
    "obj",
    "obj2",
    "aux",
    "prep",
    "verb_prefix",
    "verb_slot_inflection",
)
_KNOWN_WH_VALUES = frozenset(
    {"who", "what", "when", "where", "why", "how", "how much", "how long"}
)


@dataclass(frozen=True, slots=True)
class _QANomRow:
    values: Mapping[str, str]
    row_number: int
    source_line: int | None
    target_index: int
    is_eventive_nominal: bool
    is_passive: bool
    is_negated: bool
    release_artifacts: Mapping[str, str]


def iter_qanom_csv(
    path: str | PathLike[str],
    *,
    split: str,
    release: str = QANOM_RELEASE,
) -> Iterator[AnnotationRecord]:
    """Stream sentence records from one official QANom CSV split.

    The released files are grouped by sentence, so only one sentence is held in
    memory at a time.  A repeated sentence after its group has closed is treated
    as a format error rather than silently producing two records.
    """

    source_path = Path(path)
    _validate_split(split)
    _validate_release(release)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        artifact_fields = _validate_header(
            reader.fieldnames,
            label=str(source_path),
        )

        def numbered_rows() -> Iterator[tuple[int, int, Mapping[str, Any]]]:
            for row_number, row in enumerate(reader, start=1):
                yield row_number, reader.line_num, row

        yield from _iter_numbered_rows(
            numbered_rows(),
            split=split,
            release=release,
            source_name=source_path.name,
            expected_fields=frozenset(reader.fieldnames or ()),
            artifact_fields=artifact_fields,
        )


def load_qanom_csv(
    path: str | PathLike[str],
    *,
    split: str,
    release: str = QANOM_RELEASE,
) -> tuple[AnnotationRecord, ...]:
    """Load all sentence records from one QANom CSV split."""

    return tuple(iter_qanom_csv(path, split=split, release=release))


def iter_qanom_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    release: str = QANOM_RELEASE,
    source_name: str = "<rows>",
) -> Iterator[AnnotationRecord]:
    """Adapt a header-consistent, sentence-grouped iterable of CSV-like rows.

    This API is useful for tests and for callers that already read an archive.
    It applies the same header and value validation as :func:`iter_qanom_csv`.
    """

    _validate_split(split)
    _validate_release(release)
    iterator = iter(rows)
    try:
        first = next(iterator)
    except StopIteration:
        return
    if not isinstance(first, Mapping):
        raise DatasetFormatError(f"{source_name} row 1 must be a mapping")
    expected_fields = frozenset(first.keys())
    if any(not isinstance(field, str) for field in expected_fields):
        raise DatasetFormatError(f"{source_name} header names must be strings")
    artifact_fields = _validate_header(tuple(first.keys()), label=source_name)
    numbered = (
        (row_number, None, row)
        for row_number, row in enumerate(chain((first,), iterator), start=1)
    )
    yield from _iter_numbered_rows(
        numbered,
        split=split,
        release=release,
        source_name=source_name,
        expected_fields=expected_fields,
        artifact_fields=artifact_fields,
    )


def _iter_numbered_rows(
    rows: Iterable[tuple[int, int | None, Mapping[str, Any]]],
    *,
    split: str,
    release: str,
    source_name: str,
    expected_fields: frozenset[str],
    artifact_fields: frozenset[str],
) -> Iterator[AnnotationRecord]:
    current_sentence_id: str | None = None
    current_rows: list[_QANomRow] = []
    completed_sentence_ids: set[str] = set()

    for row_number, source_line, raw_row in rows:
        row = _normalize_row(
            raw_row,
            row_number=row_number,
            source_line=source_line,
            source_name=source_name,
            expected_fields=expected_fields,
            artifact_fields=artifact_fields,
        )
        sentence_id = row.values["qasrl_id"]
        if current_sentence_id is None:
            current_sentence_id = sentence_id
        elif sentence_id != current_sentence_id:
            yield _build_record(
                current_rows,
                split=split,
                release=release,
                source_name=source_name,
                artifact_fields=artifact_fields,
            )
            completed_sentence_ids.add(current_sentence_id)
            if sentence_id in completed_sentence_ids:
                raise DatasetFormatError(
                    f"{source_name} row {row_number}: sentence {sentence_id!r} "
                    "reappears after its group was closed"
                )
            current_sentence_id = sentence_id
            current_rows = []
        current_rows.append(row)

    if current_rows:
        yield _build_record(
            current_rows,
            split=split,
            release=release,
            source_name=source_name,
            artifact_fields=artifact_fields,
        )


def _normalize_row(
    raw_row: Mapping[str, Any],
    *,
    row_number: int,
    source_line: int | None,
    source_name: str,
    expected_fields: frozenset[str],
    artifact_fields: frozenset[str],
) -> _QANomRow:
    label = _row_label(source_name, row_number, source_line)
    if not isinstance(raw_row, Mapping):
        raise DatasetFormatError(f"{label} must be a mapping")
    actual_fields = frozenset(raw_row.keys())
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields, key=str)
        extra = sorted(actual_fields - expected_fields, key=str)
        raise DatasetFormatError(
            f"{label} fields differ from the header; missing={missing}, extra={extra}"
        )

    values: dict[str, str] = {}
    for field in QANOM_SHARED_FIELDS | QANOM_OPTIONAL_FIELDS:
        raw_value = raw_row.get(field, "")
        values[field] = require_string(
            raw_value,
            label=f"{label} {field}",
            allow_empty=field
            not in {
                "qasrl_id",
                "sentence",
                "key",
                "noun",
                "worker_id",
                "verb_form",
            },
        )

    target_index = parse_int(values["target_idx"], label=f"{label} target_idx")
    is_eventive_nominal = parse_bool(
        values["is_verbal"],
        label=f"{label} is_verbal",
    )
    is_passive = parse_bool(
        values["is_passive"],
        label=f"{label} is_passive",
    )
    is_negated = parse_bool(
        values["is_negated"],
        label=f"{label} is_negated",
    )
    release_artifacts = {
        field: require_string(
            raw_row[field],
            label=f"{label} {field}",
            allow_empty=True,
        )
        for field in artifact_fields
    }
    return _QANomRow(
        values=values,
        row_number=row_number,
        source_line=source_line,
        target_index=target_index,
        is_eventive_nominal=is_eventive_nominal,
        is_passive=is_passive,
        is_negated=is_negated,
        release_artifacts=release_artifacts,
    )


def _build_record(
    rows: Sequence[_QANomRow],
    *,
    split: str,
    release: str,
    source_name: str,
    artifact_fields: frozenset[str],
) -> AnnotationRecord:
    first = rows[0]
    sentence_id = first.values["qasrl_id"]
    sentence = first.values["sentence"]
    text, tokens = canonicalize_pretokenized_text(
        sentence,
        label=f"{source_name} sentence {sentence_id}",
    )
    for row in rows[1:]:
        if row.values["sentence"] != sentence:
            raise DatasetFormatError(
                f"{_row_context(row, source_name)}: sentence text changed within "
                f"qasrl_id {sentence_id!r}"
            )

    candidate_rows: dict[str, list[_QANomRow]] = {}
    for row in rows:
        candidate_rows.setdefault(row.values["key"], []).append(row)

    candidates = tuple(
        _build_candidate(
            grouped_rows,
            text=text,
            tokens=tokens,
            source_name=source_name,
        )
        for grouped_rows in candidate_rows.values()
    )
    duplicate_count = sum(
        int(candidate.metadata["duplicate_qa_row_count"])
        for candidate in candidates
    )
    artifact_metadata = _artifact_metadata(rows, artifact_fields)
    provenance_metadata: dict[str, Any] = {
        "adapter": "semantic_action_extractor.datasets.qanom",
        "adapter_version": QANOM_ADAPTER_VERSION,
        "official_split": split,
        "source_file": source_name,
    }
    provenance_metadata.update(artifact_metadata)
    record_metadata: dict[str, Any] = {
        "upstream_row_count": len(rows),
        "duplicate_qa_row_count": duplicate_count,
    }
    record_metadata.update(artifact_metadata)

    try:
        return AnnotationRecord(
            text=text,
            tokens=tokens,
            provenance=AnnotationProvenance(
                dataset="qanom",
                release=release,
                split=split,
                source_id=sentence_id,
                record_id=f"{split}:{sentence_id}",
                document_id=infer_qasrl_document_id(sentence_id),
                metadata=provenance_metadata,
            ),
            candidates=candidates,
            metadata=record_metadata,
        )
    except ValueError as error:
        raise DatasetFormatError(
            f"{source_name} sentence {sentence_id}: {error}"
        ) from error


def _build_candidate(
    rows: Sequence[_QANomRow],
    *,
    text: str,
    tokens: Sequence[Any],
    source_name: str,
) -> PredicateCandidate:
    first = rows[0]
    values = first.values
    key = values["key"]
    for row in rows[1:]:
        for field in _CANDIDATE_CONSTANT_FIELDS:
            if row.values[field] != values[field]:
                raise DatasetFormatError(
                    f"{_row_context(row, source_name)}: {field} changed within "
                    f"candidate {key!r}"
                )

    expected_key = f'{values["qasrl_id"]}_{first.target_index}'
    if key != expected_key:
        raise DatasetFormatError(
            f"{_row_context(first, source_name)}: key must be {expected_key!r}"
        )
    if first.target_index < 0 or first.target_index >= len(tokens):
        raise DatasetFormatError(
            f"{_row_context(first, source_name)}: target_idx "
            f"{first.target_index} falls outside the sentence"
        )
    target_span = token_aligned_span(
        text,
        tokens,
        first.target_index,
        first.target_index + 1,
        label=f"{_row_context(first, source_name)} target",
    )
    # The release lowercases ``noun`` for sentence-initial and proper-name
    # candidates.  Validate identity without pretending that the raw field is
    # a case-preserving surface form.
    if target_span.span.text.casefold() != values["noun"].casefold():
        raise DatasetFormatError(
            f"{_row_context(first, source_name)}: noun {values['noun']!r} does "
            f"not match target token {target_span.span.text!r}"
        )

    questions: list[QASRLQuestion] = []
    seen_qa_rows: set[tuple[str, ...]] = set()
    duplicate_count = 0
    no_question_row_count = 0
    upstream_source_worker_ids: list[str] = []
    source_assignment_ids: list[str] = []
    for row in rows:
        source_workers = _parse_sources_strict(
            row.values["source_worker_ids"],
            label=f"{_row_context(row, source_name)} source_worker_ids",
        )
        for source_worker in source_workers:
            if source_worker not in upstream_source_worker_ids:
                upstream_source_worker_ids.append(source_worker)
        assignments = _parse_sources_strict(
            row.values["source_assign_id"],
            label=f"{_row_context(row, source_name)} source_assign_id",
        )
        for assignment in assignments:
            if assignment not in source_assignment_ids:
                source_assignment_ids.append(assignment)
        if not row.values["question"]:
            _validate_no_question_row(row, source_name=source_name)
            no_question_row_count += 1
            continue
        fingerprint = tuple(
            row.values[field]
            for field in sorted(QANOM_SHARED_FIELDS | QANOM_OPTIONAL_FIELDS)
        )
        if fingerprint in seen_qa_rows:
            duplicate_count += 1
            continue
        seen_qa_rows.add(fingerprint)
        question_id = (
            f"{key}:question:{_stable_digest(chr(31).join(fingerprint))}"
        )
        questions.append(
            _build_question(
                row,
                question_id=question_id,
                text=text,
                tokens=tokens,
                source_name=source_name,
            )
        )

    source_assignments = tuple(source_assignment_ids)
    eventivity_question_conflict = bool(
        questions and not first.is_eventive_nominal
    )
    candidate_metadata = {
        "lemma_source": "raw_upstream_noun",
        "lemma_source_field": "noun",
        "target_surface_form": target_span.span.text,
        "noun_matches_target_case_sensitive": (
            values["noun"] == target_span.span.text
        ),
        "eventivity_source_field": "is_verbal",
        "final_worker_id": values["worker_id"],
        "source_worker_ids": tuple(upstream_source_worker_ids),
        "source_assignment_ids": source_assignments,
        "upstream_row_count": len(rows),
        "no_question_row_count": no_question_row_count,
        "duplicate_qa_row_count": duplicate_count,
        "upstream_eventivity_question_conflict": eventivity_question_conflict,
    }
    try:
        return PredicateCandidate(
            candidate_id=key,
            span=target_span,
            # QANom's ``noun`` field is a surface token, not a normalized lemma.
            # We retain it exactly and state that limitation in metadata.
            lemma=values["noun"],
            predicate_type="nominal",
            related_verbal_form=values["verb_form"],
            eventivity_judgments=(
                EventivityJudgment(
                    judgment_id=f"{key}:eventivity:{values['worker_id']}",
                    annotator_id=values["worker_id"],
                    is_eventive=first.is_eventive_nominal,
                    metadata={
                        "upstream_field": "is_verbal",
                        "source_worker_ids": tuple(upstream_source_worker_ids),
                        "source_assignment_ids": source_assignments,
                    },
                ),
            ),
            questions=tuple(questions),
            metadata=candidate_metadata,
        )
    except ValueError as error:
        raise DatasetFormatError(
            f"{source_name} candidate {key}: {error}"
        ) from error


def _build_question(
    row: _QANomRow,
    *,
    question_id: str,
    text: str,
    tokens: Sequence[Any],
    source_name: str,
) -> QASRLQuestion:
    values = row.values
    context = _row_context(row, source_name)
    question = require_string(values["question"], label=f"{context} question")
    if question != question.strip():
        raise DatasetFormatError(
            f"{context}: question cannot have leading or trailing whitespace"
        )
    wh = require_string(values["wh"], label=f"{context} wh")
    if wh not in _KNOWN_WH_VALUES:
        raise DatasetFormatError(f"{context}: unsupported wh value {wh!r}")
    for field in ("subj", "obj", "obj2", "aux", "prep", "verb_prefix"):
        require_string(values[field], label=f"{context} {field}", allow_empty=True)

    answers = _parse_answers(
        values["answer_range"],
        values["answer"],
        question_id=question_id,
        text=text,
        tokens=tokens,
        context=context,
    )
    source_workers = _parse_sources_strict(
        values["source_worker_ids"],
        label=f"{context} source_worker_ids",
    )
    question_sources = source_workers or (values["worker_id"],)
    source_assignments = _parse_sources_strict(
        values["source_assign_id"],
        label=f"{context} source_assign_id",
    )
    verb_prefix = values["verb_prefix"]
    verb_slot_inflection = values["verb_slot_inflection"]
    canonical_verb = _canonical_verb_slot(
        verb_prefix,
        verb_slot_inflection,
        context=context,
    )
    try:
        return QASRLQuestion(
            question_id=question_id,
            slots=QASRLQuestionSlots(
                wh=normalize_slot(values["wh"], label=f"{context} wh"),
                aux=normalize_slot(values["aux"], label=f"{context} aux"),
                subj=normalize_slot(values["subj"], label=f"{context} subj"),
                verb=canonical_verb,
                obj=normalize_slot(values["obj"], label=f"{context} obj"),
                prep=normalize_slot(values["prep"], label=f"{context} prep"),
                obj2=normalize_slot(values["obj2"], label=f"{context} obj2"),
                verb_prefix=verb_prefix,
                verb_slot_inflection=verb_slot_inflection,
            ),
            surface_form=question,
            question_sources=question_sources,
            judgments=(
                QuestionJudgment(
                    source_id=values["worker_id"],
                    is_valid=True,
                    answers=answers,
                    metadata={
                        "final_worker_id": values["worker_id"],
                        "source_worker_ids": source_workers,
                        "source_assignment_ids": source_assignments,
                    },
                ),
            ),
            is_negated=row.is_negated,
            is_passive=row.is_passive,
            metadata={
                "upstream_row_number": row.row_number,
                "source_line": row.source_line,
                "final_worker_id": values["worker_id"],
                "source_worker_ids": source_workers,
                "source_assignment_ids": source_assignments,
            },
        )
    except ValueError as error:
        raise DatasetFormatError(f"{context}: {error}") from error


def _parse_answers(
    raw_ranges: str,
    raw_answers: str,
    *,
    question_id: str,
    text: str,
    tokens: Sequence[Any],
    context: str,
) -> tuple[AnswerAlternative, ...]:
    if not raw_ranges:
        raise DatasetFormatError(
            f"{context}: a question must have an answer_range value"
        )
    range_values = _split_nonempty(
        raw_ranges,
        label=f"{context} answer_range",
    )
    answer_values = tuple(raw_answers.split(QANOM_SEPARATOR))
    if len(range_values) != len(answer_values):
        raise DatasetFormatError(
            f"{context}: answer_range and answer must have the same number of "
            "~!~-separated values"
        )

    alternatives: list[AnswerAlternative] = []
    for answer_index, (raw_range, surface_form) in enumerate(
        zip(range_values, answer_values)
    ):
        pieces = raw_range.split(":")
        if len(pieces) != 2 or not all(
            piece and piece.isdigit() for piece in pieces
        ):
            raise DatasetFormatError(
                f"{context}: answer range {raw_range!r} must be START:END"
            )
        token_start = parse_int(
            pieces[0],
            label=f"{context} answer range start",
        )
        token_end = parse_int(
            pieces[1],
            label=f"{context} answer range end",
        )
        span = token_aligned_span(
            text,
            tokens,
            token_start,
            token_end,
            label=f"{context} answer {answer_index}",
        )
        answer_text_missing = not surface_form
        if answer_text_missing:
            surface_form = span.span.text
        elif span.span.text != surface_form:
            raise DatasetFormatError(
                f"{context}: answer text {surface_form!r} does not match "
                f"range {raw_range!r} ({span.span.text!r})"
            )
        alternatives.append(
            AnswerAlternative(
                alternative_id=f"{question_id}:answer:{answer_index}",
                spans=(span,),
                surface_form=surface_form,
                metadata={
                    "upstream_answer_index": answer_index,
                    "upstream_answer_text_missing": answer_text_missing,
                },
            )
        )
    return tuple(alternatives)


def _validate_no_question_row(row: _QANomRow, *, source_name: str) -> None:
    context = _row_context(row, source_name)
    nonempty = [field for field in _QUESTION_TEXT_FIELDS if row.values[field]]
    if nonempty:
        raise DatasetFormatError(
            f"{context}: an empty question must also have empty question fields; "
            f"nonempty={nonempty}"
        )
    if row.is_passive or row.is_negated:
        raise DatasetFormatError(
            f"{context}: an empty question must set is_passive and is_negated "
            "to False"
        )


def _canonical_verb_slot(
    raw_prefix: str,
    raw_inflection: str,
    *,
    context: str,
) -> str:
    if not raw_inflection:
        return "_"
    allowed = {
        "Stem",
        "PresentSingular3rd",
        "PresentParticiple",
        "Past",
        "PastParticiple",
    }
    if raw_inflection not in allowed:
        raise DatasetFormatError(
            f"{context}: unsupported verb_slot_inflection {raw_inflection!r}"
        )
    prefix_parts = _parse_sources_strict(
        raw_prefix,
        label=f"{context} verb_prefix",
    )
    canonical_inflection = raw_inflection[0].lower() + raw_inflection[1:]
    return " ".join((*prefix_parts, canonical_inflection))


def _parse_sources_strict(value: str, *, label: str) -> tuple[str, ...]:
    parts = split_sources(value, label=label)
    if value and QANOM_SEPARATOR.join(parts) != value:
        raise DatasetFormatError(f"{label} contains an empty source ID")
    return parts


def _split_nonempty(value: str, *, label: str) -> tuple[str, ...]:
    parts = tuple(value.split(QANOM_SEPARATOR))
    if not parts or any(not part for part in parts):
        raise DatasetFormatError(f"{label} contains an empty separated value")
    return parts


def _artifact_metadata(
    rows: Sequence[_QANomRow],
    artifact_fields: frozenset[str],
) -> dict[str, Any]:
    if not artifact_fields:
        return {}
    return {
        "discarded_release_artifact_fields": tuple(sorted(artifact_fields)),
        "nonempty_release_artifact_value_counts": {
            field: sum(bool(row.release_artifacts[field]) for row in rows)
            for field in sorted(artifact_fields)
        },
    }


def _validate_header(
    fieldnames: Sequence[str] | None,
    *,
    label: str,
) -> frozenset[str]:
    if not fieldnames:
        raise DatasetFormatError(f"{label} must have a CSV header")
    if any(not isinstance(field, str) or not field for field in fieldnames):
        raise DatasetFormatError(f"{label} header names must be non-empty strings")
    if len(fieldnames) != len(set(fieldnames)):
        raise DatasetFormatError(f"{label} header names must be unique")
    fields = frozenset(fieldnames)
    missing = QANOM_SHARED_FIELDS - fields
    accepted = (
        QANOM_SHARED_FIELDS | QANOM_OPTIONAL_FIELDS | QANOM_RELEASE_ARTIFACT_FIELDS
    )
    unknown = fields - accepted
    if missing or unknown:
        raise DatasetFormatError(
            f"{label} has an unsupported QANom header; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return fields & QANOM_RELEASE_ARTIFACT_FIELDS


def _validate_split(split: str) -> None:
    if split not in QANOM_SPLITS:
        choices = ", ".join(sorted(QANOM_SPLITS))
        raise DatasetFormatError(f"QANom split must be one of: {choices}")


def _validate_release(release: str) -> None:
    require_string(release, label="QANom release")


def _row_context(row: _QANomRow, source_name: str) -> str:
    return _row_label(source_name, row.row_number, row.source_line)


def _row_label(
    source_name: str,
    row_number: int,
    source_line: int | None,
) -> str:
    if source_line is None:
        return f"{source_name} row {row_number}"
    return f"{source_name} row {row_number} (CSV line {source_line})"


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
