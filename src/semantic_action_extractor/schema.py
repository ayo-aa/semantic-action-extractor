"""Typed, JSON-serializable output schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A zero-based, end-exclusive span copied from the input text."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end < self.start:
            raise ValueError("span end must be greater than or equal to start")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class Qualifier:
    """A prepositional relation preserved without guessing a deeper role."""

    relation: str
    value: TextSpan

    def __post_init__(self) -> None:
        if not self.relation:
            raise ValueError("qualifier relation cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"relation": self.relation, "value": self.value.to_dict()}


@dataclass(frozen=True, slots=True)
class ActionFrame:
    """One source-grounded action identified in a sentence."""

    predicate: TextSpan
    predicate_lemma: str
    sentence_index: int
    actor: TextSpan | None = None
    patient: TextSpan | None = None
    qualifiers: tuple[Qualifier, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    extractor: str = "rule-based-v0"

    def __post_init__(self) -> None:
        if not self.predicate_lemma:
            raise ValueError("predicate lemma cannot be empty")
        if self.sentence_index < 0:
            raise ValueError("sentence index must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": None if self.actor is None else self.actor.to_dict(),
            "predicate": self.predicate.to_dict(),
            "predicate_lemma": self.predicate_lemma,
            "patient": None if self.patient is None else self.patient.to_dict(),
            "qualifiers": [qualifier.to_dict() for qualifier in self.qualifiers],
            "sentence_index": self.sentence_index,
            "confidence": round(self.confidence, 3),
            "extractor": self.extractor,
        }


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Complete extractor response for one input string."""

    text: str
    actions: tuple[ActionFrame, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "actions": [action.to_dict() for action in self.actions],
            "warnings": list(self.warnings),
        }
