"""Strict, streaming adapters for QA-SRL Bank 2.1 and Gold Standard.

The upstream release stores one sentence per JSON line.  A sentence contains
verbal predicate entries, structured questions, and the individual validation
judgments collected for each question.  This module converts that evidence to
the evidence-preserving research schema without consolidating votes or treating separate
answer alternatives as a discontinuous span.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, TextIO

from ..annotation_schema import (
    AnnotationProvenance,
    AnnotationRecord,
    AnswerAlternative,
    PredicateCandidate,
    QASRLQuestion,
    QASRLQuestionSlots,
    QuestionJudgment,
    VerbInflectionParadigm,
)
from .common import (
    DatasetFormatError,
    canonicalize_tokens,
    infer_qasrl_document_id,
    normalize_slot,
    parse_int,
    require_string,
    token_aligned_span,
)


QASRL_BANK_RELEASE = "2.1"
QASRL_GOLD_RELEASE = "f7c64ae9b6fe48ff3910c3e59850a12ec278bf83"
QASRL_ADAPTER_VERSION = "0.1.0"


_RECORD_KEYS = frozenset({"sentenceId", "sentenceTokens", "verbEntries"})
_RECORD_OPTIONAL_KEYS = frozenset({"nonPredicates"})
_VERB_KEYS = frozenset(
    {"verbIndex", "verbInflectedForms", "questionLabels"}
)
_INFLECTION_KEYS = frozenset(
    {
        "stem",
        "presentSingular3rd",
        "presentParticiple",
        "past",
        "pastParticiple",
    }
)
_QUESTION_KEYS = frozenset(
    {
        "questionString",
        "questionSources",
        "answerJudgments",
        "questionSlots",
        "tense",
        "isPerfect",
        "isProgressive",
        "isNegated",
        "isPassive",
    }
)
_SLOT_KEYS = frozenset({"wh", "aux", "subj", "verb", "obj", "prep", "obj2"})
_JUDGMENT_REQUIRED_KEYS = frozenset({"sourceId", "isValid"})
_JUDGMENT_OPTIONAL_KEYS = frozenset({"spans"})
_INDEX_KEYS = frozenset({"documents", "denseIds"})
_INDEX_DOCUMENT_KEYS = frozenset({"part", "idString", "domain", "id", "title"})
_SUPPORTED_DOMAINS = frozenset({"tqa", "wikipedia", "wikinews"})
_SUPPORTED_SPLITS = frozenset({"train", "dev", "test"})
_SUPPORTED_LAYERS = frozenset({"orig", "expanded", "dense"})


@dataclass(frozen=True, slots=True)
class QASRLDocument:
    """One document entry from QA-SRL Bank's ``index.json.gz``."""

    document_id: str
    split: str
    domain: str
    source_id: str
    title: str


@dataclass(frozen=True, slots=True)
class QASRLIndex:
    """Validated document metadata and dense-sentence membership."""

    documents: Mapping[str, QASRLDocument]
    dense_sentence_ids: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "documents", MappingProxyType(dict(self.documents)))
        object.__setattr__(
            self,
            "dense_sentence_ids",
            frozenset(self.dense_sentence_ids),
        )

    def document_for_sentence(
        self,
        sentence_id: str,
        *,
        split: str | None = None,
    ) -> QASRLDocument:
        """Resolve and optionally split-check a sentence's source document."""

        document_id = infer_qasrl_document_id(sentence_id)
        try:
            document = self.documents[document_id]
        except KeyError as error:
            raise DatasetFormatError(
                f"sentence {sentence_id!r} refers to document {document_id!r}, "
                "which is absent from the QA-SRL index"
            ) from error
        if split is not None and document.split != split:
            raise DatasetFormatError(
                f"sentence {sentence_id!r} is declared as split {split!r}, but its "
                f"indexed document belongs to {document.split!r}"
            )
        return document


def load_qasrl_index(path: str | Path) -> QASRLIndex:
    """Load and strictly validate QA-SRL Bank's JSON document index."""

    source = Path(path)
    with _open_json_text(source, expected_suffixes=(".json", ".json.gz")) as handle:
        try:
            payload = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise DatasetFormatError(
                f"invalid QA-SRL index JSON in {source}: {error}"
            ) from error

    root = _require_mapping(payload, label="QA-SRL index")
    _require_exact_keys(root, _INDEX_KEYS, label="QA-SRL index")
    documents_by_split = _require_mapping(
        root["documents"],
        label="QA-SRL index documents",
    )

    documents: dict[str, QASRLDocument] = {}
    for split, raw_documents in documents_by_split.items():
        split_name = require_string(split, label="QA-SRL index split")
        if split_name not in _SUPPORTED_SPLITS:
            raise DatasetFormatError(
                f"QA-SRL index split {split_name!r} is unsupported"
            )
        if not isinstance(raw_documents, list):
            raise DatasetFormatError(
                f"QA-SRL index documents[{split_name!r}] must be a list"
            )
        for position, raw_document in enumerate(raw_documents):
            label = f"QA-SRL index documents[{split_name!r}][{position}]"
            document_payload = _require_mapping(raw_document, label=label)
            _require_exact_keys(document_payload, _INDEX_DOCUMENT_KEYS, label=label)

            part = require_string(document_payload["part"], label=f"{label}.part")
            if part != split_name:
                raise DatasetFormatError(
                    f"{label}.part is {part!r}, expected containing split "
                    f"{split_name!r}"
                )
            document_id = require_string(
                document_payload["idString"],
                label=f"{label}.idString",
            )
            domain = require_string(
                document_payload["domain"],
                label=f"{label}.domain",
            )
            if domain not in _SUPPORTED_DOMAINS:
                raise DatasetFormatError(f"{label}.domain {domain!r} is unsupported")
            _validate_document_identity(document_id, domain=domain, label=label)

            document = QASRLDocument(
                document_id=document_id,
                split=part,
                domain=domain,
                source_id=require_string(
                    document_payload["id"],
                    label=f"{label}.id",
                ),
                title=require_string(
                    document_payload["title"],
                    label=f"{label}.title",
                    allow_empty=True,
                ),
            )
            if document_id in documents:
                raise DatasetFormatError(
                    f"QA-SRL index repeats document id {document_id!r}"
                )
            documents[document_id] = document

    raw_dense_ids = root["denseIds"]
    if not isinstance(raw_dense_ids, list):
        raise DatasetFormatError("QA-SRL index denseIds must be a list")
    dense_ids: list[str] = []
    for position, raw_sentence_id in enumerate(raw_dense_ids):
        sentence_id = require_string(
            raw_sentence_id,
            label=f"QA-SRL index denseIds[{position}]",
        )
        document_id = infer_qasrl_document_id(sentence_id)
        if document_id not in documents:
            raise DatasetFormatError(
                f"dense sentence {sentence_id!r} has no indexed document"
            )
        dense_ids.append(sentence_id)
    if len(dense_ids) != len(set(dense_ids)):
        raise DatasetFormatError("QA-SRL index denseIds contains duplicates")

    return QASRLIndex(documents=documents, dense_sentence_ids=frozenset(dense_ids))


def adapt_qasrl_record(
    raw_record: Mapping[str, Any],
    *,
    release: str,
    split: str,
    layer: str | None = None,
    document_index: QASRLIndex | None = None,
) -> AnnotationRecord:
    """Convert one upstream QA-SRL sentence to the research annotation schema.

    The function is deliberately strict: unknown or missing fields, mismatched
    dictionary keys, invalid spans, and contradictory judgments are rejected
    instead of being silently repaired.
    """

    release_name = require_string(release, label="QA-SRL release")
    split_name = require_string(split, label="QA-SRL split")
    if split_name not in _SUPPORTED_SPLITS:
        raise DatasetFormatError(f"QA-SRL split {split_name!r} is unsupported")
    layer_name = None if layer is None else require_string(layer, label="QA-SRL layer")
    if layer_name is not None and layer_name not in _SUPPORTED_LAYERS:
        raise DatasetFormatError(f"QA-SRL layer {layer_name!r} is unsupported")

    record = _require_mapping(raw_record, label="QA-SRL record")
    _require_keys(
        record,
        required=_RECORD_KEYS,
        optional=_RECORD_OPTIONAL_KEYS,
        label="QA-SRL record",
    )

    sentence_id = require_string(record["sentenceId"], label="sentenceId")
    document_id = infer_qasrl_document_id(sentence_id)
    text, tokens = canonicalize_tokens(
        record["sentenceTokens"],
        label="sentenceTokens",
    )
    domain = _infer_domain(sentence_id)
    document_title: str | None = None
    document_source_id: str | None = None
    if document_index is not None:
        document = document_index.document_for_sentence(sentence_id, split=split_name)
        if document.domain != domain:
            raise DatasetFormatError(
                f"sentence {sentence_id!r} encodes domain {domain!r}, but its index "
                f"entry declares {document.domain!r}"
            )
        if (
            layer_name == "dense"
            and sentence_id not in document_index.dense_sentence_ids
        ):
            raise DatasetFormatError(
                f"dense record {sentence_id!r} is absent from index denseIds"
            )
        document_title = document.title
        document_source_id = document.source_id

    raw_verbs = _require_mapping(record["verbEntries"], label="verbEntries")
    candidates: list[PredicateCandidate] = []
    empty_valid_span_judgment_count = 0
    parsed_verb_keys: dict[int, str] = {}
    for raw_verb_key in raw_verbs:
        verb_index = parse_int(raw_verb_key, label="verbEntries key")
        if raw_verb_key != str(verb_index):
            raise DatasetFormatError(
                f"verbEntries key {raw_verb_key!r} is not a canonical token index"
            )
        if verb_index in parsed_verb_keys:
            raise DatasetFormatError(
                f"verbEntries keys {parsed_verb_keys[verb_index]!r} and "
                f"{raw_verb_key!r} encode the same token index"
            )
        parsed_verb_keys[verb_index] = raw_verb_key

    for verb_index in sorted(parsed_verb_keys):
        raw_verb_key = parsed_verb_keys[verb_index]
        verb_label = f"verbEntries[{raw_verb_key!r}]"
        verb = _require_mapping(raw_verbs[raw_verb_key], label=verb_label)
        _require_exact_keys(verb, _VERB_KEYS, label=verb_label)
        declared_index = _require_json_integer(
            verb["verbIndex"],
            label=f"{verb_label}.verbIndex",
        )
        if declared_index != verb_index:
            raise DatasetFormatError(
                f"{verb_label}.verbIndex is {declared_index}, expected {verb_index}"
            )
        if not 0 <= verb_index < len(tokens):
            raise DatasetFormatError(
                f"{verb_label}.verbIndex {verb_index} falls outside sentenceTokens"
            )

        inflections = _adapt_inflections(
            verb["verbInflectedForms"],
            label=f"{verb_label}.verbInflectedForms",
        )
        candidate_id = f"qa-srl:{sentence_id}:verb:{verb_index}"
        questions, candidate_empty_valid_span_count = _adapt_questions(
            verb["questionLabels"],
            text=text,
            tokens=tokens,
            candidate_id=candidate_id,
            label=f"{verb_label}.questionLabels",
        )
        empty_valid_span_judgment_count += candidate_empty_valid_span_count
        candidates.append(
            PredicateCandidate(
                candidate_id=candidate_id,
                span=token_aligned_span(
                    text,
                    tokens,
                    verb_index,
                    verb_index + 1,
                    label=f"{verb_label} predicate",
                ),
                lemma=inflections.stem,
                predicate_type="verbal",
                verb_inflected_forms=inflections,
                questions=questions,
                metadata={"upstream_verb_key": raw_verb_key},
            )
        )

    provenance_metadata: dict[str, Any] = {
        "domain": domain,
        "layer": layer_name,
    }
    if document_title is not None:
        provenance_metadata["document_title"] = document_title
        provenance_metadata["document_source_id"] = document_source_id

    record_metadata: dict[str, Any] = {
        "adapter": "qa-srl-jsonl-v1",
        "upstream_anomaly_counts": {
            "empty_valid_spans": empty_valid_span_judgment_count,
        },
    }
    if "nonPredicates" in record:
        non_predicates = _require_mapping(
            record["nonPredicates"],
            label="nonPredicates",
        )
        record_metadata["non_predicates"] = dict(non_predicates)

    return AnnotationRecord(
        text=text,
        tokens=tokens,
        provenance=AnnotationProvenance(
            dataset="qa-srl",
            release=release_name,
            split=split_name,
            source_id=sentence_id,
            record_id=(
                f"qa-srl:{release_name}:{layer_name or 'gold'}:"
                f"{split_name}:{sentence_id}"
            ),
            document_id=document_id,
            metadata=provenance_metadata,
        ),
        candidates=tuple(candidates),
        metadata=record_metadata,
    )


def iter_qasrl_records(
    path: str | Path,
    *,
    release: str,
    split: str,
    layer: str | None = None,
    document_index: QASRLIndex | None = None,
) -> Iterator[AnnotationRecord]:
    """Stream adapted records from an uncompressed or gzipped JSONL file."""

    source = Path(path)
    with _open_json_text(source, expected_suffixes=(".jsonl", ".jsonl.gz")) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DatasetFormatError(
                    f"blank JSONL record at {source}:{line_number}"
                )
            try:
                raw_record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise DatasetFormatError(
                    f"invalid QA-SRL JSON at {source}:{line_number}: {error}"
                ) from error
            try:
                yield adapt_qasrl_record(
                    raw_record,
                    release=release,
                    split=split,
                    layer=layer,
                    document_index=document_index,
                )
            except DatasetFormatError as error:
                raise DatasetFormatError(
                    f"invalid QA-SRL record at {source}:{line_number}: {error}"
                ) from error


def _adapt_inflections(raw_value: Any, *, label: str) -> VerbInflectionParadigm:
    payload = _require_mapping(raw_value, label=label)
    _require_exact_keys(payload, _INFLECTION_KEYS, label=label)
    return VerbInflectionParadigm(
        stem=require_string(payload["stem"], label=f"{label}.stem"),
        present_singular_3rd=require_string(
            payload["presentSingular3rd"],
            label=f"{label}.presentSingular3rd",
        ),
        present_participle=require_string(
            payload["presentParticiple"],
            label=f"{label}.presentParticiple",
        ),
        past=require_string(payload["past"], label=f"{label}.past"),
        past_participle=require_string(
            payload["pastParticiple"],
            label=f"{label}.pastParticiple",
        ),
    )


def _adapt_questions(
    raw_value: Any,
    *,
    text: str,
    tokens: tuple[Any, ...],
    candidate_id: str,
    label: str,
) -> tuple[tuple[QASRLQuestion, ...], int]:
    raw_questions = _require_mapping(raw_value, label=label)
    questions: list[QASRLQuestion] = []
    empty_valid_span_judgment_count = 0
    for raw_question_key in sorted(raw_questions):
        question_key = require_string(raw_question_key, label=f"{label} key")
        question_label = f"{label}[{question_key!r}]"
        raw_question = _require_mapping(
            raw_questions[raw_question_key],
            label=question_label,
        )
        _require_exact_keys(raw_question, _QUESTION_KEYS, label=question_label)

        surface_form = require_string(
            raw_question["questionString"],
            label=f"{question_label}.questionString",
        )
        if surface_form != question_key:
            raise DatasetFormatError(
                f"{question_label}.questionString does not match its dictionary key"
            )
        question_id = f"{candidate_id}:question:{_stable_digest(question_key)}"

        raw_sources = raw_question["questionSources"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise DatasetFormatError(
                f"{question_label}.questionSources must be a non-empty list"
            )
        question_sources = tuple(
            require_string(source, label=f"{question_label}.questionSources[{index}]")
            for index, source in enumerate(raw_sources)
        )
        if len(question_sources) != len(set(question_sources)):
            raise DatasetFormatError(
                f"{question_label}.questionSources contains duplicates"
            )

        slots = _adapt_slots(
            raw_question["questionSlots"],
            label=f"{question_label}.questionSlots",
        )
        judgments, question_empty_valid_span_count = _adapt_judgments(
            raw_question["answerJudgments"],
            text=text,
            tokens=tokens,
            question_id=question_id,
            label=f"{question_label}.answerJudgments",
        )
        empty_valid_span_judgment_count += question_empty_valid_span_count
        questions.append(
            QASRLQuestion(
                question_id=question_id,
                slots=slots,
                surface_form=surface_form,
                question_sources=question_sources,
                judgments=judgments,
                tense=require_string(
                    raw_question["tense"],
                    label=f"{question_label}.tense",
                ),
                is_perfect=_require_json_boolean(
                    raw_question["isPerfect"],
                    label=f"{question_label}.isPerfect",
                ),
                is_progressive=_require_json_boolean(
                    raw_question["isProgressive"],
                    label=f"{question_label}.isProgressive",
                ),
                is_negated=_require_json_boolean(
                    raw_question["isNegated"],
                    label=f"{question_label}.isNegated",
                ),
                is_passive=_require_json_boolean(
                    raw_question["isPassive"],
                    label=f"{question_label}.isPassive",
                ),
                metadata={"upstream_question_key": question_key},
            )
        )
    return tuple(questions), empty_valid_span_judgment_count


def _adapt_slots(raw_value: Any, *, label: str) -> QASRLQuestionSlots:
    payload = _require_mapping(raw_value, label=label)
    _require_exact_keys(payload, _SLOT_KEYS, label=label)
    try:
        return QASRLQuestionSlots(
            wh=normalize_slot(payload["wh"], label=f"{label}.wh"),
            aux=normalize_slot(payload["aux"], label=f"{label}.aux"),
            subj=normalize_slot(payload["subj"], label=f"{label}.subj"),
            verb=normalize_slot(payload["verb"], label=f"{label}.verb"),
            obj=normalize_slot(payload["obj"], label=f"{label}.obj"),
            prep=normalize_slot(payload["prep"], label=f"{label}.prep"),
            obj2=normalize_slot(payload["obj2"], label=f"{label}.obj2"),
        )
    except ValueError as error:
        raise DatasetFormatError(f"{label} is invalid: {error}") from error


def _adapt_judgments(
    raw_value: Any,
    *,
    text: str,
    tokens: tuple[Any, ...],
    question_id: str,
    label: str,
) -> tuple[tuple[QuestionJudgment, ...], int]:
    if not isinstance(raw_value, list) or not raw_value:
        raise DatasetFormatError(f"{label} must be a non-empty list")
    judgments: list[QuestionJudgment] = []
    empty_valid_span_judgment_count = 0
    source_counts: dict[str, int] = {}
    for judgment_index, raw_judgment in enumerate(raw_value):
        judgment_label = f"{label}[{judgment_index}]"
        payload = _require_mapping(raw_judgment, label=judgment_label)
        _require_keys(
            payload,
            required=_JUDGMENT_REQUIRED_KEYS,
            optional=_JUDGMENT_OPTIONAL_KEYS,
            label=judgment_label,
        )
        source_id = require_string(
            payload["sourceId"],
            label=f"{judgment_label}.sourceId",
        )
        is_valid = _require_json_boolean(
            payload["isValid"],
            label=f"{judgment_label}.isValid",
        )
        raw_spans = payload.get("spans", [])
        if not isinstance(raw_spans, list):
            raise DatasetFormatError(f"{judgment_label}.spans must be a list")
        if is_valid and "spans" not in payload:
            raise DatasetFormatError(
                f"{judgment_label} is valid but omits its spans field"
            )
        if not is_valid and raw_spans:
            raise DatasetFormatError(
                f"{judgment_label} is invalid but contains answer spans"
            )

        source_ordinal = source_counts.get(source_id, 0)
        source_counts[source_id] = source_ordinal + 1
        has_empty_valid_spans = is_valid and not raw_spans
        empty_valid_span_judgment_count += int(has_empty_valid_spans)
        answers: list[AnswerAlternative] = []
        seen_spans: set[tuple[int, int]] = set()
        for span_index, raw_span in enumerate(raw_spans):
            span_label = f"{judgment_label}.spans[{span_index}]"
            if not isinstance(raw_span, list) or len(raw_span) != 2:
                raise DatasetFormatError(
                    f"{span_label} must be a two-item token range"
                )
            token_start = _require_json_integer(
                raw_span[0],
                label=f"{span_label}[0]",
            )
            token_end = _require_json_integer(
                raw_span[1],
                label=f"{span_label}[1]",
            )
            span_key = (token_start, token_end)
            if span_key in seen_spans:
                raise DatasetFormatError(
                    f"{judgment_label}.spans repeats range {span_key}"
                )
            seen_spans.add(span_key)
            aligned = token_aligned_span(
                text,
                tokens,
                token_start,
                token_end,
                label=span_label,
            )
            answer_id = (
                f"{question_id}:judgment:{_stable_digest(source_id)}:"
                f"{source_ordinal}:answer:{token_start}-{token_end}"
            )
            answers.append(
                AnswerAlternative(
                    alternative_id=answer_id,
                    spans=(aligned,),
                    surface_form=aligned.span.text,
                    metadata={"upstream_span_index": span_index},
                )
            )

        judgments.append(
            QuestionJudgment(
                source_id=source_id,
                is_valid=is_valid,
                answers=tuple(answers),
                metadata={
                    "upstream_judgment_index": judgment_index,
                    "source_ordinal": source_ordinal,
                    "upstream_empty_valid_spans": has_empty_valid_spans,
                },
            )
        )
    return tuple(judgments), empty_valid_span_judgment_count


def _infer_domain(sentence_id: str) -> str:
    if sentence_id.startswith("TQA:"):
        return "tqa"
    if sentence_id.startswith("Wiki1k:"):
        parts = sentence_id.split(":")
        if len(parts) >= 2 and parts[1] in {"wikipedia", "wikinews"}:
            return parts[1]
    raise DatasetFormatError(
        f"cannot infer a supported QA-SRL domain from sentence id {sentence_id!r}"
    )


def _validate_document_identity(document_id: str, *, domain: str, label: str) -> None:
    if domain == "tqa":
        if not document_id.startswith("TQA:T_") or ":" in document_id[4:]:
            raise DatasetFormatError(
                f"{label}.idString is inconsistent with tqa domain"
            )
        return
    prefix = f"Wiki1k:{domain}:"
    if not document_id.startswith(prefix) or not document_id[len(prefix) :]:
        raise DatasetFormatError(
            f"{label}.idString is inconsistent with {domain} domain"
        )


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _require_json_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetFormatError(f"{label} must be a JSON integer")
    return value


def _require_json_boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetFormatError(f"{label} must be a JSON boolean")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetFormatError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise DatasetFormatError(f"{label} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    _require_keys(value, required=expected, optional=frozenset(), label=label)


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    label: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise DatasetFormatError(f"{label} has invalid fields: {', '.join(details)}")


def _open_json_text(
    path: Path,
    *,
    expected_suffixes: tuple[str, ...],
) -> TextIO:
    path_string = str(path)
    if not any(path_string.endswith(suffix) for suffix in expected_suffixes):
        choices = ", ".join(expected_suffixes)
        raise DatasetFormatError(f"{path} must end with one of: {choices}")
    try:
        if path_string.endswith(".gz"):
            return gzip.open(path, mode="rt", encoding="utf-8", newline="")
        return path.open(mode="r", encoding="utf-8", newline="")
    except OSError as error:
        raise DatasetFormatError(f"cannot open {path}: {error}") from error
