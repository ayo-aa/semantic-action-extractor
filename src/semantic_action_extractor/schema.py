"""Typed, JSON-serializable output schema.

All character offsets are zero-based, end-exclusive Unicode code-point indices,
equivalent to indexes into a Python string. They are not UTF-8 byte offsets or
UTF-16 code-unit offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable


SCHEMA_VERSION = "0.3.0"
PREDICATE_TYPES = frozenset({"verbal", "nominal"})
ROLE_SCHEMES = frozenset({"surface", "qa_srl", "coarse"})
MENTION_QUALIFIER_KINDS = frozenset(
    {
        "conditional",
        "future",
        "hypothetical",
        "necessary",
        "negated",
        "planned",
        "possible",
        "questioned",
        "reported",
    }
)


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A source span addressed by zero-based, end-exclusive code-point offsets."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("span text length must match its start and end offsets")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class MentionQualifier:
    """Source wording that qualifies an event mention without asserting its status.

    Qualifiers describe how the source presents a mention.  They do not establish
    whether an event occurred, was completed, was assigned, or should be executed.
    """

    kind: str
    evidence: tuple[TextSpan, ...]

    def __post_init__(self) -> None:
        if self.kind not in MENTION_QUALIFIER_KINDS:
            choices = ", ".join(sorted(MENTION_QUALIFIER_KINDS))
            raise ValueError(f"mention qualifier kind must be one of: {choices}")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError(
                "mention qualifier evidence must be a non-empty tuple of spans"
            )
        previous_end = -1
        for span in self.evidence:
            if not isinstance(span, TextSpan):
                raise TypeError(
                    "mention qualifier evidence must contain only TextSpan values"
                )
            if span.start < previous_end:
                raise ValueError(
                    "mention qualifier evidence must be ordered and non-overlapping"
                )
            previous_end = span.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "evidence": [span.to_dict() for span in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class ActionArgument:
    """One source-grounded participant or circumstance of a predicate."""

    role: str
    span: TextSpan
    role_scheme: str = "surface"
    cue: TextSpan | None = None
    group_id: str | None = None
    confidence: float | None = None
    confidence_type: str | None = None

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("argument role cannot be empty")
        if self.role_scheme not in ROLE_SCHEMES:
            choices = ", ".join(sorted(ROLE_SCHEMES))
            raise ValueError(f"argument role_scheme must be one of: {choices}")
        if self.group_id is not None and (
            not isinstance(self.group_id, str) or not self.group_id.strip()
        ):
            raise ValueError("argument group_id cannot be empty when provided")
        _validate_optional_confidence(
            self.confidence,
            self.confidence_type,
            label="argument confidence",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "role": self.role,
            "role_scheme": self.role_scheme,
            "span": self.span.to_dict(),
            "cue": None if self.cue is None else self.cue.to_dict(),
        }
        if self.group_id is not None:
            payload["group_id"] = self.group_id
        if self.confidence is not None:
            payload["confidence"] = self.confidence
            payload["confidence_type"] = self.confidence_type
        return payload


@dataclass(frozen=True, slots=True)
class ActionFrame:
    """One source-grounded verbal or nominal action mention."""

    predicate: TextSpan
    predicate_lemma: str
    predicate_type: str
    sentence_index: int
    score: float
    score_type: str
    extractor: str
    arguments: tuple[ActionArgument, ...] = field(default_factory=tuple)
    related_verbal_form: str | None = None
    predicate_confidence: float | None = None
    predicate_confidence_type: str | None = None
    mention_qualifiers: tuple[MentionQualifier, ...] | None = None

    def __post_init__(self) -> None:
        if not self.predicate_lemma:
            raise ValueError("predicate lemma cannot be empty")
        if (
            self.related_verbal_form is not None
            and not self.related_verbal_form.strip()
        ):
            raise ValueError("related_verbal_form cannot be empty when provided")
        if self.predicate_type not in PREDICATE_TYPES:
            choices = ", ".join(sorted(PREDICATE_TYPES))
            raise ValueError(f"predicate_type must be one of: {choices}")
        if self.sentence_index < 0:
            raise ValueError("sentence index must be non-negative")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("score must be between 0 and 1")
        if not self.score_type.strip():
            raise ValueError("score_type cannot be empty")
        if not self.extractor.strip():
            raise ValueError("extractor cannot be empty")
        _validate_optional_confidence(
            self.predicate_confidence,
            self.predicate_confidence_type,
            label="predicate confidence",
        )
        if self.mention_qualifiers is not None:
            if not isinstance(self.mention_qualifiers, tuple):
                raise TypeError(
                    "mention_qualifiers must be a tuple or None when not assessed"
                )
            if any(
                not isinstance(qualifier, MentionQualifier)
                for qualifier in self.mention_qualifiers
            ):
                raise TypeError(
                    "mention_qualifiers must contain only MentionQualifier values"
                )
            kinds = [qualifier.kind for qualifier in self.mention_qualifiers]
            if len(kinds) != len(set(kinds)):
                raise ValueError(
                    "mention qualifier kinds must be unique within an action frame"
                )
        grouped_roles: dict[str, tuple[str, str]] = {}
        for argument in self.arguments:
            if argument.group_id is None:
                continue
            signature = (argument.role, argument.role_scheme)
            previous = grouped_roles.setdefault(argument.group_id, signature)
            if previous != signature:
                raise ValueError(
                    "arguments with one group_id must share a role and role_scheme"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "predicate": self.predicate.to_dict(),
            "predicate_lemma": self.predicate_lemma,
            "predicate_type": self.predicate_type,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "mention_qualifiers": (
                None
                if self.mention_qualifiers is None
                else [qualifier.to_dict() for qualifier in self.mention_qualifiers]
            ),
            "sentence_index": self.sentence_index,
            "score": self.score,
            "score_type": self.score_type,
            "extractor": self.extractor,
        }
        if self.related_verbal_form is not None:
            payload["related_verbal_form"] = self.related_verbal_form
        if self.predicate_confidence is not None:
            payload["predicate_confidence"] = self.predicate_confidence
            payload["predicate_confidence_type"] = self.predicate_confidence_type
        return payload


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Complete extractor response for one input string."""

    text: str
    actions: tuple[ActionFrame, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        for action in self.actions:
            for span in _action_spans(action):
                if (
                    span.end > len(self.text)
                    or self.text[span.start : span.end] != span.text
                ):
                    raise ValueError(
                        "every predicate, argument, and cue span must map exactly "
                        "to ExtractionResult.text"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "text": self.text,
            "actions": [action.to_dict() for action in self.actions],
            "warnings": list(self.warnings),
        }


def _action_spans(action: ActionFrame) -> Iterable[TextSpan]:
    yield action.predicate
    for argument in action.arguments:
        yield argument.span
        if argument.cue is not None:
            yield argument.cue
    if action.mention_qualifiers is not None:
        for qualifier in action.mention_qualifiers:
            yield from qualifier.evidence


def _validate_optional_confidence(
    value: float | None,
    value_type: str | None,
    *,
    label: str,
) -> None:
    """Validate a confidence value whose interpretation must travel with it."""

    if (value is None) != (value_type is None):
        raise ValueError(f"{label} and its type must be provided together")
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{label} must be between 0 and 1")
    if not isinstance(value_type, str) or not value_type.strip():
        raise ValueError(f"{label} type cannot be empty")
