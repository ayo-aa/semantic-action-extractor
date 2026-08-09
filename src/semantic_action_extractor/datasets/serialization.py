"""Strict readers for canonical annotation JSONL and preparation manifests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from ..annotation_schema import (
    AnnotationProvenance,
    AnnotationRecord,
    AnnotationToken,
    AnswerAlternative,
    EventivityJudgment,
    PredicateCandidate,
    QASRLQuestion,
    QASRLQuestionSlots,
    QuestionJudgment,
    TokenAlignedSpan,
    VerbInflectionParadigm,
)
from ..schema import TextSpan
from .common import DatasetFormatError
from .io import sha256_file
from .manifest import PreparationManifest, SourceArtifactIdentity


def annotation_record_from_dict(raw: object) -> AnnotationRecord:
    """Reconstruct and validate one canonical annotation record."""

    try:
        payload = _object(raw, label="annotation record")
        _keys(
            payload,
            required={
                "schema_version",
                "text",
                "tokens",
                "provenance",
                "candidates",
                "metadata",
            },
            label="annotation record",
        )
        return AnnotationRecord(
            schema_version=_string(
                payload["schema_version"], label="annotation schema_version"
            ),
            text=_string(payload["text"], label="annotation text"),
            tokens=tuple(
                _token(item, label=f"tokens[{index}]")
                for index, item in enumerate(_list(payload["tokens"], label="tokens"))
            ),
            provenance=_provenance(payload["provenance"]),
            candidates=tuple(
                _candidate(item, label=f"candidates[{index}]")
                for index, item in enumerate(
                    _list(payload["candidates"], label="candidates")
                )
            ),
            metadata=_metadata(payload["metadata"], label="record metadata"),
        )
    except DatasetFormatError:
        raise
    except (TypeError, ValueError) as error:
        raise DatasetFormatError(f"invalid annotation record: {error}") from error


def iter_adapted_jsonl(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> Iterator[AnnotationRecord]:
    """Stream canonical records and verify an optional preparation manifest."""

    source = Path(path)
    manifest = (
        load_preparation_manifest(manifest_path)
        if manifest_path is not None
        else None
    )
    if manifest is not None:
        actual = sha256_file(source)
        if actual != manifest.record_fingerprint:
            raise DatasetFormatError(
                f"adapted JSONL checksum mismatch: expected "
                f"{manifest.record_fingerprint}, found {actual}"
            )

    record_count = 0
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DatasetFormatError(
                        f"{source} line {line_number}: blank lines are not allowed"
                    )
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetFormatError(
                        f"{source} line {line_number}: invalid JSON: {error.msg}"
                    ) from error
                try:
                    record = annotation_record_from_dict(raw)
                except DatasetFormatError as error:
                    raise DatasetFormatError(
                        f"{source} line {line_number}: {error}"
                    ) from error
                if manifest is not None:
                    _validate_record_manifest_identity(record, manifest, line_number)
                record_count += 1
                yield record
    except UnicodeDecodeError as error:
        raise DatasetFormatError(f"{source} is not valid UTF-8: {error}") from error

    if not record_count:
        raise DatasetFormatError(f"adapted JSONL contains no records: {source}")
    if manifest is not None and record_count != manifest.counts.get("records"):
        raise DatasetFormatError(
            f"adapted JSONL record count is {record_count}, expected "
            f"{manifest.counts.get('records')} from its manifest"
        )
    if manifest is not None:
        final_digest = sha256_file(source)
        if final_digest != manifest.record_fingerprint:
            raise DatasetFormatError(
                f"adapted JSONL changed while it was being read: {source}"
            )


def load_adapted_jsonl(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[AnnotationRecord, ...]:
    """Load and fully validate a canonical annotation JSONL file."""

    return tuple(iter_adapted_jsonl(path, manifest_path=manifest_path))


def load_preparation_manifest(path: str | Path) -> PreparationManifest:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DatasetFormatError(f"invalid preparation manifest {source}: {error}") from error
    try:
        payload = _object(raw, label="preparation manifest")
        _keys(
            payload,
            required={
                "manifest_version",
                "schema_version",
                "dataset",
                "release",
                "split",
                "adapter",
                "adapter_version",
                "source_artifacts",
                "counts",
                "anomalies",
                "exclusions",
                "record_fingerprint",
            },
            label="preparation manifest",
        )
        artifacts = []
        for index, item in enumerate(
            _list(payload["source_artifacts"], label="source_artifacts")
        ):
            artifact = _object(item, label=f"source_artifacts[{index}]")
            _keys(
                artifact,
                required={"name", "sha256"},
                label=f"source_artifacts[{index}]",
            )
            artifacts.append(
                SourceArtifactIdentity(
                    name=_string(
                        artifact["name"], label=f"source_artifacts[{index}].name"
                    ),
                    sha256=_string(
                        artifact["sha256"],
                        label=f"source_artifacts[{index}].sha256",
                    ),
                )
            )
        return PreparationManifest(
            manifest_version=_string(
                payload["manifest_version"], label="manifest_version"
            ),
            schema_version=_string(payload["schema_version"], label="schema_version"),
            dataset=_string(payload["dataset"], label="dataset"),
            release=_string(payload["release"], label="release"),
            split=_string(payload["split"], label="split"),
            adapter=_string(payload["adapter"], label="adapter"),
            adapter_version=_string(
                payload["adapter_version"], label="adapter_version"
            ),
            source_artifacts=tuple(artifacts),
            counts=_integer_mapping(payload["counts"], label="counts"),
            anomalies=_integer_mapping(payload["anomalies"], label="anomalies"),
            exclusions=_integer_mapping(payload["exclusions"], label="exclusions"),
            record_fingerprint=_string(
                payload["record_fingerprint"], label="record_fingerprint"
            ),
        )
    except DatasetFormatError:
        raise
    except (TypeError, ValueError) as error:
        raise DatasetFormatError(f"invalid preparation manifest {source}: {error}") from error


def _provenance(raw: object) -> AnnotationProvenance:
    payload = _object(raw, label="provenance")
    _keys(
        payload,
        required={
            "dataset",
            "release",
            "split",
            "source_id",
            "record_id",
            "document_id",
            "metadata",
        },
        label="provenance",
    )
    return AnnotationProvenance(
        dataset=_string(payload["dataset"], label="provenance.dataset"),
        release=_string(payload["release"], label="provenance.release"),
        split=_string(payload["split"], label="provenance.split"),
        source_id=_string(payload["source_id"], label="provenance.source_id"),
        record_id=_string(payload["record_id"], label="provenance.record_id"),
        document_id=_optional_string(
            payload["document_id"], label="provenance.document_id"
        ),
        metadata=_metadata(payload["metadata"], label="provenance.metadata"),
    )


def _token(raw: object, *, label: str) -> AnnotationToken:
    payload = _object(raw, label=label)
    _keys(payload, required={"index", "text", "start", "end"}, label=label)
    return AnnotationToken(
        index=_integer(payload["index"], label=f"{label}.index"),
        span=_text_span(payload, label=label),
    )


def _aligned_span(raw: object, *, label: str) -> TokenAlignedSpan:
    payload = _object(raw, label=label)
    _keys(
        payload,
        required={"text", "start", "end", "token_start", "token_end"},
        label=label,
    )
    return TokenAlignedSpan(
        span=_text_span(payload, label=label),
        token_start=_integer(payload["token_start"], label=f"{label}.token_start"),
        token_end=_integer(payload["token_end"], label=f"{label}.token_end"),
    )


def _text_span(payload: Mapping[str, Any], *, label: str) -> TextSpan:
    return TextSpan(
        text=_string(payload["text"], label=f"{label}.text"),
        start=_integer(payload["start"], label=f"{label}.start"),
        end=_integer(payload["end"], label=f"{label}.end"),
    )


def _inflections(raw: object, *, label: str) -> VerbInflectionParadigm:
    payload = _object(raw, label=label)
    _keys(
        payload,
        required={
            "stem",
            "presentSingular3rd",
            "presentParticiple",
            "past",
            "pastParticiple",
        },
        label=label,
    )
    return VerbInflectionParadigm(
        stem=_string(payload["stem"], label=f"{label}.stem"),
        present_singular_3rd=_string(
            payload["presentSingular3rd"], label=f"{label}.presentSingular3rd"
        ),
        present_participle=_string(
            payload["presentParticiple"], label=f"{label}.presentParticiple"
        ),
        past=_string(payload["past"], label=f"{label}.past"),
        past_participle=_string(
            payload["pastParticiple"], label=f"{label}.pastParticiple"
        ),
    )


def _answer(raw: object, *, label: str) -> AnswerAlternative:
    payload = _object(raw, label=label)
    _keys(
        payload,
        required={"alternative_id", "spans", "surface_form", "metadata"},
        label=label,
    )
    return AnswerAlternative(
        alternative_id=_string(
            payload["alternative_id"], label=f"{label}.alternative_id"
        ),
        spans=tuple(
            _aligned_span(item, label=f"{label}.spans[{index}]")
            for index, item in enumerate(_list(payload["spans"], label=f"{label}.spans"))
        ),
        surface_form=_optional_string(
            payload["surface_form"], label=f"{label}.surface_form"
        ),
        metadata=_metadata(payload["metadata"], label=f"{label}.metadata"),
    )


def _question_judgment(raw: object, *, label: str) -> QuestionJudgment:
    payload = _object(raw, label=label)
    _keys(
        payload,
        required={"source_id", "is_valid", "answers", "metadata"},
        optional={"confidence", "confidence_type"},
        label=label,
    )
    return QuestionJudgment(
        source_id=_string(payload["source_id"], label=f"{label}.source_id"),
        is_valid=_boolean(payload["is_valid"], label=f"{label}.is_valid"),
        answers=tuple(
            _answer(item, label=f"{label}.answers[{index}]")
            for index, item in enumerate(
                _list(payload["answers"], label=f"{label}.answers")
            )
        ),
        confidence=_optional_number(payload.get("confidence"), label=f"{label}.confidence"),
        confidence_type=_optional_string(
            payload.get("confidence_type"), label=f"{label}.confidence_type"
        ),
        metadata=_metadata(payload["metadata"], label=f"{label}.metadata"),
    )


def _slots(raw: object, *, label: str) -> QASRLQuestionSlots:
    payload = _object(raw, label=label)
    fields = {
        "wh",
        "aux",
        "subj",
        "verb",
        "obj",
        "prep",
        "obj2",
        "verb_prefix",
        "verb_slot_inflection",
    }
    _keys(payload, required=fields, label=label)
    return QASRLQuestionSlots(
        wh=_string(payload["wh"], label=f"{label}.wh"),
        aux=_string(payload["aux"], label=f"{label}.aux"),
        subj=_string(payload["subj"], label=f"{label}.subj"),
        verb=_string(payload["verb"], label=f"{label}.verb"),
        obj=_string(payload["obj"], label=f"{label}.obj"),
        prep=_string(payload["prep"], label=f"{label}.prep"),
        obj2=_string(payload["obj2"], label=f"{label}.obj2"),
        verb_prefix=_optional_string(
            payload["verb_prefix"], label=f"{label}.verb_prefix", allow_empty=True
        ),
        verb_slot_inflection=_optional_string(
            payload["verb_slot_inflection"],
            label=f"{label}.verb_slot_inflection",
            allow_empty=True,
        ),
    )


def _question(raw: object, *, label: str) -> QASRLQuestion:
    payload = _object(raw, label=label)
    _keys(
        payload,
        required={
            "question_id",
            "slots",
            "surface_form",
            "question_sources",
            "judgments",
            "tense",
            "is_perfect",
            "is_progressive",
            "is_negated",
            "is_passive",
            "metadata",
        },
        label=label,
    )
    return QASRLQuestion(
        question_id=_string(payload["question_id"], label=f"{label}.question_id"),
        slots=_slots(payload["slots"], label=f"{label}.slots"),
        surface_form=_string(payload["surface_form"], label=f"{label}.surface_form"),
        question_sources=tuple(
            _string(item, label=f"{label}.question_sources[{index}]")
            for index, item in enumerate(
                _list(payload["question_sources"], label=f"{label}.question_sources")
            )
        ),
        judgments=tuple(
            _question_judgment(item, label=f"{label}.judgments[{index}]")
            for index, item in enumerate(
                _list(payload["judgments"], label=f"{label}.judgments")
            )
        ),
        tense=_optional_string(payload["tense"], label=f"{label}.tense"),
        is_perfect=_optional_boolean(
            payload["is_perfect"], label=f"{label}.is_perfect"
        ),
        is_progressive=_optional_boolean(
            payload["is_progressive"], label=f"{label}.is_progressive"
        ),
        is_negated=_boolean(payload["is_negated"], label=f"{label}.is_negated"),
        is_passive=_boolean(payload["is_passive"], label=f"{label}.is_passive"),
        metadata=_metadata(payload["metadata"], label=f"{label}.metadata"),
    )


def _eventivity(raw: object, *, label: str) -> EventivityJudgment:
    payload = _object(raw, label=label)
    _keys(
        payload,
        required={"judgment_id", "annotator_id", "is_eventive", "metadata"},
        optional={"confidence", "confidence_type"},
        label=label,
    )
    return EventivityJudgment(
        judgment_id=_string(payload["judgment_id"], label=f"{label}.judgment_id"),
        annotator_id=_optional_string(
            payload["annotator_id"], label=f"{label}.annotator_id"
        ),
        is_eventive=_boolean(payload["is_eventive"], label=f"{label}.is_eventive"),
        confidence=_optional_number(payload.get("confidence"), label=f"{label}.confidence"),
        confidence_type=_optional_string(
            payload.get("confidence_type"), label=f"{label}.confidence_type"
        ),
        metadata=_metadata(payload["metadata"], label=f"{label}.metadata"),
    )


def _candidate(raw: object, *, label: str) -> PredicateCandidate:
    payload = _object(raw, label=label)
    _keys(
        payload,
        required={
            "candidate_id",
            "span",
            "lemma",
            "predicate_type",
            "related_verbal_form",
            "verb_inflected_forms",
            "eventivity_judgments",
            "questions",
            "metadata",
        },
        label=label,
    )
    raw_inflections = payload["verb_inflected_forms"]
    return PredicateCandidate(
        candidate_id=_string(payload["candidate_id"], label=f"{label}.candidate_id"),
        span=_aligned_span(payload["span"], label=f"{label}.span"),
        lemma=_string(payload["lemma"], label=f"{label}.lemma"),
        predicate_type=_string(
            payload["predicate_type"], label=f"{label}.predicate_type"
        ),
        related_verbal_form=_optional_string(
            payload["related_verbal_form"], label=f"{label}.related_verbal_form"
        ),
        verb_inflected_forms=(
            None
            if raw_inflections is None
            else _inflections(raw_inflections, label=f"{label}.verb_inflected_forms")
        ),
        eventivity_judgments=tuple(
            _eventivity(item, label=f"{label}.eventivity_judgments[{index}]")
            for index, item in enumerate(
                _list(
                    payload["eventivity_judgments"],
                    label=f"{label}.eventivity_judgments",
                )
            )
        ),
        questions=tuple(
            _question(item, label=f"{label}.questions[{index}]")
            for index, item in enumerate(
                _list(payload["questions"], label=f"{label}.questions")
            )
        ),
        metadata=_metadata(payload["metadata"], label=f"{label}.metadata"),
    )


def _validate_record_manifest_identity(
    record: AnnotationRecord,
    manifest: PreparationManifest,
    line_number: int,
) -> None:
    expected = (manifest.dataset, manifest.release, manifest.split)
    actual = (
        record.provenance.dataset,
        record.provenance.release,
        record.provenance.split,
    )
    if actual != expected:
        raise DatasetFormatError(
            f"record on line {line_number} has dataset/release/split {actual}, "
            f"expected {expected} from its manifest"
        )


def _object(raw: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise DatasetFormatError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in raw):
        raise DatasetFormatError(f"{label} keys must be strings")
    return raw


def _list(raw: object, *, label: str) -> Sequence[object]:
    if not isinstance(raw, list):
        raise DatasetFormatError(f"{label} must be a list")
    return raw


def _keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    label: str,
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    actual = set(payload)
    missing = sorted(required - actual)
    extra = sorted(actual - allowed)
    if missing or extra:
        raise DatasetFormatError(
            f"{label} fields differ from the schema; missing={missing}, extra={extra}"
        )


def _string(raw: object, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(raw, str) or (not allow_empty and not raw.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise DatasetFormatError(f"{label} must be {qualifier}")
    return raw


def _optional_string(
    raw: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> str | None:
    if raw is None:
        return None
    return _string(raw, label=label, allow_empty=allow_empty)


def _integer(raw: object, *, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise DatasetFormatError(f"{label} must be an integer")
    return raw


def _boolean(raw: object, *, label: str) -> bool:
    if not isinstance(raw, bool):
        raise DatasetFormatError(f"{label} must be a boolean")
    return raw


def _optional_boolean(raw: object, *, label: str) -> bool | None:
    if raw is None:
        return None
    return _boolean(raw, label=label)


def _optional_number(raw: object, *, label: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DatasetFormatError(f"{label} must be numeric")
    return float(raw)


def _metadata(raw: object, *, label: str) -> dict[str, Any]:
    return dict(_object(raw, label=label))


def _integer_mapping(raw: object, *, label: str) -> dict[str, int]:
    payload = _object(raw, label=label)
    return {
        key: _integer(value, label=f"{label}.{key}")
        for key, value in payload.items()
    }
