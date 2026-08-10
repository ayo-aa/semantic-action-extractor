"""Typed, JSON-serializable output schema."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A zero-based, end-exclusive span copied from the input text."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("span text must be a string")
        if not self.text:
            raise ValueError("span text cannot be empty")
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("span offsets must be integers")
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("span text length must match its character offsets")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class Qualifier:
    """A prepositional relation preserved without guessing a deeper role."""

    relation: str
    value: TextSpan

    def __post_init__(self) -> None:
        if not isinstance(self.relation, str):
            raise TypeError("qualifier relation must be a string")
        if not self.relation.strip():
            raise ValueError("qualifier relation cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"relation": self.relation, "value": self.value.to_dict()}


@dataclass(frozen=True, slots=True)
class PredicateCandidate:
    """One rule-proposed predicate with sentence-local model input."""

    words: tuple[TextSpan, ...]
    predicate_index: int
    predicate_lemma: str
    sentence_index: int

    def __post_init__(self) -> None:
        if not self.words:
            raise ValueError("candidate words cannot be empty")
        if any(
            earlier.end > later.start
            for earlier, later in zip(self.words, self.words[1:], strict=False)
        ):
            raise ValueError("candidate words must be ordered and non-overlapping")
        if type(self.predicate_index) is not int:
            raise TypeError("predicate index must be an integer")
        if not 0 <= self.predicate_index < len(self.words):
            raise ValueError("predicate index must refer to a candidate word")
        if not isinstance(self.predicate_lemma, str):
            raise TypeError("predicate lemma must be a string")
        if not self.predicate_lemma.strip():
            raise ValueError("predicate lemma cannot be empty")
        if type(self.sentence_index) is not int:
            raise TypeError("sentence index must be an integer")
        if self.sentence_index < 0:
            raise ValueError("sentence index must be non-negative")

    @property
    def predicate(self) -> TextSpan:
        return self.words[self.predicate_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_index": self.sentence_index,
            "predicate_index": self.predicate_index,
            "predicate": self.predicate.to_dict(),
            "predicate_lemma": self.predicate_lemma,
            "words": [word.to_dict() for word in self.words],
        }


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
        if not isinstance(self.predicate_lemma, str):
            raise TypeError("predicate lemma must be a string")
        if not self.predicate_lemma.strip():
            raise ValueError("predicate lemma cannot be empty")
        if type(self.sentence_index) is not int:
            raise TypeError("sentence index must be an integer")
        if self.sentence_index < 0:
            raise ValueError("sentence index must be non-negative")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise TypeError("confidence must be a real number")
        if not math.isfinite(float(self.confidence)):
            raise ValueError("confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.extractor, str):
            raise TypeError("extractor must be a string")
        if not self.extractor.strip():
            raise ValueError("extractor cannot be empty")

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

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("result text must be a string")
        for action in self.actions:
            spans = [action.predicate]
            if action.actor is not None:
                spans.append(action.actor)
            if action.patient is not None:
                spans.append(action.patient)
            spans.extend(qualifier.value for qualifier in action.qualifiers)
            for span in spans:
                if span.end > len(self.text) or self.text[span.start : span.end] != span.text:
                    raise ValueError("every action span must match the source text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "actions": [action.to_dict() for action in self.actions],
            "warnings": list(self.warnings),
        }
