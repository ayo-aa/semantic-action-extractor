"""Versioned, reusable evaluation-corpus bundles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import uuid

from ..datasets.common import DatasetFormatError
from .types import (
    EvaluationArgument,
    EvaluationCorpus,
    EvaluationPredicate,
    EvaluationQAPair,
    EvaluationQuestion,
    PredicateKey,
)


EVALUATION_BUNDLE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    corpus: EvaluationCorpus
    predicate_source: str
    consolidation_rule: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    bundle_version: str = EVALUATION_BUNDLE_VERSION

    def __post_init__(self) -> None:
        for value, label in (
            (self.predicate_source, "predicate_source"),
            (self.consolidation_rule, "consolidation_rule"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evaluation {label} cannot be empty")
        if self.bundle_version != EVALUATION_BUNDLE_VERSION:
            raise ValueError(
                f"bundle_version must be {EVALUATION_BUNDLE_VERSION}"
            )
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("evaluation metadata must be JSON-serializable") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_version": self.bundle_version,
            "predicate_source": self.predicate_source,
            "consolidation_rule": self.consolidation_rule,
            "metadata": dict(self.metadata),
            "corpus": self.corpus.to_dict(),
        }

    def write(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to replace {destination}")
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(f"evaluation output is a directory: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.partial"
        )
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        try:
            partial.write_text(f"{payload}\n", encoding="utf-8")
            partial.replace(destination)
        except Exception:
            if partial.exists() and partial.is_file():
                partial.unlink()
            raise
        return destination


def load_evaluation_bundle(path: str | Path) -> EvaluationBundle:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        payload = _object(raw, label="evaluation bundle")
        _keys(
            payload,
            {
                "bundle_version",
                "predicate_source",
                "consolidation_rule",
                "metadata",
                "corpus",
            },
            label="evaluation bundle",
        )
        corpus_payload = _object(payload["corpus"], label="evaluation corpus")
        _keys(corpus_payload, {"predicates"}, label="evaluation corpus")
        predicates = tuple(
            _predicate(item, label=f"predicates[{index}]")
            for index, item in enumerate(
                _list(corpus_payload["predicates"], label="predicates")
            )
        )
        return EvaluationBundle(
            bundle_version=_string(payload["bundle_version"], label="bundle_version"),
            predicate_source=_string(
                payload["predicate_source"], label="predicate_source"
            ),
            consolidation_rule=_string(
                payload["consolidation_rule"], label="consolidation_rule"
            ),
            metadata=dict(_object(payload["metadata"], label="metadata")),
            corpus=EvaluationCorpus(predicates=predicates),
        )
    except DatasetFormatError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as error:
        raise DatasetFormatError(f"invalid evaluation bundle {source}: {error}") from error


def _predicate(raw: object, *, label: str) -> EvaluationPredicate:
    payload = _object(raw, label=label)
    _keys(payload, {"key", "is_eventive", "lemma", "pairs"}, label=label)
    key_payload = _object(payload["key"], label=f"{label}.key")
    _keys(
        key_payload,
        {"source_id", "token_start", "token_end", "predicate_type"},
        label=f"{label}.key",
    )
    return EvaluationPredicate(
        key=PredicateKey(
            source_id=_string(
                key_payload["source_id"], label=f"{label}.key.source_id"
            ),
            token_start=_integer(
                key_payload["token_start"], label=f"{label}.key.token_start"
            ),
            token_end=_integer(
                key_payload["token_end"], label=f"{label}.key.token_end"
            ),
            predicate_type=_string(
                key_payload["predicate_type"],
                label=f"{label}.key.predicate_type",
            ),
        ),
        is_eventive=_boolean(payload["is_eventive"], label=f"{label}.is_eventive"),
        lemma=_optional_string(payload["lemma"], label=f"{label}.lemma"),
        pairs=tuple(
            _pair(item, label=f"{label}.pairs[{index}]")
            for index, item in enumerate(_list(payload["pairs"], label=f"{label}.pairs"))
        ),
    )


def _pair(raw: object, *, label: str) -> EvaluationQAPair:
    payload = _object(raw, label=label)
    _keys(
        payload,
        {"pair_id", "role_id", "question", "argument", "metadata"},
        label=label,
    )
    question = _object(payload["question"], label=f"{label}.question")
    question_fields = {
        "surface_form",
        "wh",
        "aux",
        "subj",
        "verb",
        "obj",
        "prep",
        "obj2",
        "is_passive",
        "is_negated",
    }
    _keys(question, question_fields, label=f"{label}.question")
    argument = _object(payload["argument"], label=f"{label}.argument")
    _keys(
        argument,
        {"token_spans", "character_spans"},
        label=f"{label}.argument",
    )
    raw_character_spans = argument["character_spans"]
    return EvaluationQAPair(
        pair_id=_string(payload["pair_id"], label=f"{label}.pair_id"),
        role_id=_optional_string(payload["role_id"], label=f"{label}.role_id"),
        question=EvaluationQuestion(
            surface_form=_string(
                question["surface_form"], label=f"{label}.question.surface_form"
            ),
            wh=_string(question["wh"], label=f"{label}.question.wh"),
            aux=_string(question["aux"], label=f"{label}.question.aux"),
            subj=_string(question["subj"], label=f"{label}.question.subj"),
            verb=_string(question["verb"], label=f"{label}.question.verb"),
            obj=_string(question["obj"], label=f"{label}.question.obj"),
            prep=_string(question["prep"], label=f"{label}.question.prep"),
            obj2=_string(question["obj2"], label=f"{label}.question.obj2"),
            is_passive=_boolean(
                question["is_passive"], label=f"{label}.question.is_passive"
            ),
            is_negated=_boolean(
                question["is_negated"], label=f"{label}.question.is_negated"
            ),
        ),
        argument=EvaluationArgument(
            token_spans=_ranges(
                argument["token_spans"], label=f"{label}.argument.token_spans"
            ),
            character_spans=(
                None
                if raw_character_spans is None
                else _ranges(
                    raw_character_spans,
                    label=f"{label}.argument.character_spans",
                )
            ),
        ),
        metadata=dict(_object(payload["metadata"], label=f"{label}.metadata")),
    )


def _ranges(raw: object, *, label: str) -> tuple[tuple[int, int], ...]:
    values = _list(raw, label=label)
    result = []
    for index, item in enumerate(values):
        pair = _list(item, label=f"{label}[{index}]")
        if len(pair) != 2:
            raise DatasetFormatError(f"{label}[{index}] must contain two integers")
        result.append(
            (
                _integer(pair[0], label=f"{label}[{index}][0]"),
                _integer(pair[1], label=f"{label}[{index}][1]"),
            )
        )
    return tuple(result)


def _object(raw: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise DatasetFormatError(f"{label} must be an object with string keys")
    return raw


def _list(raw: object, *, label: str) -> list[object]:
    if not isinstance(raw, list):
        raise DatasetFormatError(f"{label} must be a list")
    return raw


def _keys(payload: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise DatasetFormatError(
            f"{label} fields differ from the schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise DatasetFormatError(f"{label} must be a non-empty string")
    return raw


def _optional_string(raw: object, *, label: str) -> str | None:
    return None if raw is None else _string(raw, label=label)


def _integer(raw: object, *, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise DatasetFormatError(f"{label} must be an integer")
    return raw


def _boolean(raw: object, *, label: str) -> bool:
    if not isinstance(raw, bool):
        raise DatasetFormatError(f"{label} must be a boolean")
    return raw
