"""Small, model-neutral types consumed by every scorer mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..schema import MENTION_QUALIFIER_KINDS, PREDICATE_TYPES


def _nonempty(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")


@dataclass(frozen=True, order=True, slots=True)
class PredicateKey:
    """Exact predicate identity used by the corrected end-to-end scorer."""

    source_id: str
    token_start: int
    token_end: int
    predicate_type: str

    def __post_init__(self) -> None:
        _nonempty(self.source_id, label="predicate source_id")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("predicate token range must be non-empty")
        if self.predicate_type not in PREDICATE_TYPES:
            raise ValueError("unsupported predicate_type")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "source_id": self.source_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "predicate_type": self.predicate_type,
        }


@dataclass(frozen=True, slots=True)
class EvaluationArgument:
    """One answer, including grouped or discontinuous source spans."""

    token_spans: tuple[tuple[int, int], ...]
    character_spans: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        _validate_ranges(self.token_spans, label="argument token spans")
        if self.character_spans is not None:
            _validate_ranges(self.character_spans, label="argument character spans")
            if len(self.character_spans) != len(self.token_spans):
                raise ValueError(
                    "argument character spans must parallel the token spans"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_spans": [list(span) for span in self.token_spans],
            "character_spans": (
                None
                if self.character_spans is None
                else [list(span) for span in self.character_spans]
            ),
        }


@dataclass(frozen=True, slots=True)
class EvaluationMentionQualifier:
    """One qualifier kind and its grouped token/character evidence spans."""

    kind: str
    evidence: EvaluationArgument

    def __post_init__(self) -> None:
        if self.kind not in MENTION_QUALIFIER_KINDS:
            choices = ", ".join(sorted(MENTION_QUALIFIER_KINDS))
            raise ValueError(f"mention qualifier kind must be one of: {choices}")
        if not isinstance(self.evidence, EvaluationArgument):
            raise TypeError(
                "mention qualifier evidence must be an EvaluationArgument"
            )
        if self.evidence.character_spans is None:
            raise ValueError(
                "mention qualifier evidence requires character spans"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "evidence": self.evidence.to_dict()}


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """The seven QA-SRL slots plus the two reference voice features."""

    surface_form: str
    wh: str
    aux: str
    subj: str
    verb: str
    obj: str
    prep: str
    obj2: str
    is_passive: bool
    is_negated: bool

    def __post_init__(self) -> None:
        _nonempty(self.surface_form, label="question surface_form")
        for field_name in ("wh", "aux", "subj", "verb", "obj", "prep", "obj2"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"question {field_name} must use '_' rather than an empty value"
                )
        if not isinstance(self.is_passive, bool) or not isinstance(
            self.is_negated, bool
        ):
            raise ValueError("question voice flags must be booleans")

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "surface_form": self.surface_form,
            "wh": self.wh,
            "aux": self.aux,
            "subj": self.subj,
            "verb": self.verb,
            "obj": self.obj,
            "prep": self.prep,
            "obj2": self.obj2,
            "is_passive": self.is_passive,
            "is_negated": self.is_negated,
        }


@dataclass(frozen=True, slots=True)
class EvaluationQAPair:
    pair_id: str
    question: EvaluationQuestion
    argument: EvaluationArgument
    role_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.pair_id, label="pair_id")
        if self.role_id is not None:
            _nonempty(self.role_id, label="role_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "role_id": self.role_id,
            "question": self.question.to_dict(),
            "argument": self.argument.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvaluationPredicate:
    key: PredicateKey
    is_eventive: bool
    lemma: str | None = None
    pairs: tuple[EvaluationQAPair, ...] = field(default_factory=tuple)
    mention_qualifiers: tuple[EvaluationMentionQualifier, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.is_eventive, bool):
            raise ValueError("predicate is_eventive must be a boolean")
        if self.lemma is not None:
            _nonempty(self.lemma, label="predicate lemma")
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("pair IDs must be unique within a predicate")
        if not self.is_eventive and self.pairs:
            raise ValueError("a non-eventive predicate cannot contain QA pairs")
        if not self.is_eventive and self.mention_qualifiers is not None:
            raise ValueError(
                "a non-eventive predicate cannot contain mention qualifiers"
            )
        if self.mention_qualifiers is not None:
            if not isinstance(self.mention_qualifiers, tuple):
                raise TypeError(
                    "mention_qualifiers must be a tuple or None when not assessed"
                )
            if any(
                not isinstance(qualifier, EvaluationMentionQualifier)
                for qualifier in self.mention_qualifiers
            ):
                raise TypeError(
                    "mention_qualifiers must contain only "
                    "EvaluationMentionQualifier values"
                )
            kinds = [qualifier.kind for qualifier in self.mention_qualifiers]
            if len(kinds) != len(set(kinds)):
                raise ValueError(
                    "mention qualifier kinds must be unique within a predicate"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "is_eventive": self.is_eventive,
            "lemma": self.lemma,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "mention_qualifiers": (
                None
                if self.mention_qualifiers is None
                else [item.to_dict() for item in self.mention_qualifiers]
            ),
        }


@dataclass(frozen=True, slots=True)
class EvaluationCorpus:
    predicates: tuple[EvaluationPredicate, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        keys = [predicate.key for predicate in self.predicates]
        if len(keys) != len(set(keys)):
            raise ValueError("predicate keys must be unique within an evaluation corpus")

    def by_key(self) -> dict[PredicateKey, EvaluationPredicate]:
        return {predicate.key: predicate for predicate in self.predicates}

    def to_dict(self) -> dict[str, Any]:
        return {"predicates": [predicate.to_dict() for predicate in self.predicates]}


def _validate_ranges(ranges: tuple[tuple[int, int], ...], *, label: str) -> None:
    if not ranges:
        raise ValueError(f"{label} cannot be empty")
    previous_end = -1
    for item in ranges:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or isinstance(item[0], bool)
            or isinstance(item[1], bool)
            or not isinstance(item[0], int)
            or not isinstance(item[1], int)
        ):
            raise ValueError(f"{label} must contain integer (start, end) tuples")
        start, end = item
        if start < 0 or end <= start:
            raise ValueError(f"{label} must contain non-empty ranges")
        if start < previous_end:
            raise ValueError(f"{label} must be ordered and non-overlapping")
        previous_end = end
