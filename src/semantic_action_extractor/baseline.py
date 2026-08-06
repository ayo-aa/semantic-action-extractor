"""Transparent rule baseline for source-grounded action extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib
from typing import Any, Iterable

from .schema import ActionFrame, ExtractionResult, Qualifier, TextSpan


_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.MULTILINE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)

_BASE_VERBS = frozenset(
    {
        "analyze",
        "approve",
        "archive",
        "ask",
        "assign",
        "book",
        "build",
        "call",
        "cancel",
        "close",
        "complete",
        "contact",
        "create",
        "delete",
        "deliver",
        "deploy",
        "download",
        "email",
        "escalate",
        "find",
        "hire",
        "launch",
        "make",
        "meet",
        "move",
        "notify",
        "open",
        "order",
        "pay",
        "plan",
        "process",
        "provide",
        "purchase",
        "read",
        "receive",
        "refund",
        "reject",
        "remind",
        "request",
        "resolve",
        "review",
        "run",
        "schedule",
        "send",
        "ship",
        "sign",
        "submit",
        "tell",
        "test",
        "train",
        "transfer",
        "update",
        "upload",
        "use",
        "want",
        "work",
        "write",
    }
)

_IRREGULAR_LEMMAS = {
    "asked": "ask",
    "bought": "purchase",
    "built": "build",
    "called": "call",
    "cancelled": "cancel",
    "canceled": "cancel",
    "did": "do",
    "done": "do",
    "found": "find",
    "made": "make",
    "met": "meet",
    "paid": "pay",
    "ran": "run",
    "read": "read",
    "sent": "send",
    "told": "tell",
    "wrote": "write",
    "written": "write",
}

_AUXILIARIES = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
    }
)
_DETERMINERS = frozenset({"a", "an", "the", "this", "that", "these", "those"})
_DISCOURSE_PREFIXES = frozenset({"also", "next", "now", "please", "then"})
_COORDINATORS = frozenset({"and", "but", "or", "then"})
_PREPOSITIONS = frozenset(
    {
        "after",
        "at",
        "before",
        "by",
        "during",
        "for",
        "from",
        "in",
        "into",
        "on",
        "through",
        "to",
        "with",
        "without",
    }
)
_CLAUSE_PUNCTUATION = frozenset({",", ";", ":", ".", "!", "?"})
_TRAILING_PUNCTUATION = frozenset({",", ";", ":", ".", "!", "?", ")", "]", "}"})
_LEADING_PUNCTUATION = frozenset({",", ";", ":", "(", "[", "{"})


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Configuration for the rule baseline."""

    additional_verbs: tuple[str, ...] = ()
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        invalid = [verb for verb in self.additional_verbs if not _normalise_verb(verb)]
        if invalid:
            raise ValueError(f"additional verbs cannot be empty: {invalid!r}")

    @classmethod
    def from_toml(cls, path: str | Path) -> "BaselineConfig":
        """Load `[baseline]` values from a TOML file."""

        config_path = Path(path)
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
        section = document.get("baseline", {})
        if not isinstance(section, dict):
            raise ValueError("[baseline] must be a TOML table")

        allowed = {"additional_verbs", "min_confidence"}
        unknown = sorted(set(section) - allowed)
        if unknown:
            raise ValueError(f"unknown baseline configuration keys: {', '.join(unknown)}")

        verbs = section.get("additional_verbs", [])
        if not isinstance(verbs, list) or not all(isinstance(item, str) for item in verbs):
            raise ValueError("baseline.additional_verbs must be an array of strings")

        threshold = section.get("min_confidence", 0.0)
        if not isinstance(threshold, (int, float)):
            raise ValueError("baseline.min_confidence must be a number")

        return cls(
            additional_verbs=tuple(_normalise_verb(verb) for verb in verbs),
            min_confidence=float(threshold),
        )


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    start: int
    end: int

    @property
    def lower(self) -> str:
        return self.text.casefold()


class RuleBasedExtractor:
    """Extract simple action frames without a model or external dependency."""

    extractor_id = "rule-based-v0"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self._verbs = _BASE_VERBS | frozenset(self.config.additional_verbs)

    def extract(self, text: str) -> ExtractionResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        actions: list[ActionFrame] = []
        warnings: list[str] = []

        for sentence_index, sentence_match in enumerate(_SENTENCE_RE.finditer(text)):
            tokens = [
                _Token(
                    text=token_match.group(0),
                    start=token_match.start(),
                    end=token_match.end(),
                )
                for token_match in _TOKEN_RE.finditer(
                    text, sentence_match.start(), sentence_match.end()
                )
            ]
            sentence_actions = self._extract_sentence(text, tokens, sentence_index)
            actions.extend(sentence_actions)

            lowered = {token.lower for token in tokens}
            if sentence_actions and lowered & {"never", "no", "not", "without"}:
                warnings.append(
                    f"Sentence {sentence_index} contains possible negation; "
                    "rule-based-v0 does not encode action polarity."
                )

        if text.strip() and not actions:
            warnings.append("No action predicates matched the rule-based vocabulary.")

        return ExtractionResult(text=text, actions=tuple(actions), warnings=tuple(warnings))

    def _extract_sentence(
        self, text: str, tokens: list[_Token], sentence_index: int
    ) -> list[ActionFrame]:
        candidates = self._verb_candidates(tokens)
        candidate_positions = {position for position, _ in candidates}
        actions: list[ActionFrame] = []
        inherited_actor: TextSpan | None = None

        for candidate_number, (verb_index, lemma) in enumerate(candidates):
            actor_tokens = self._actor_tokens(tokens, verb_index)
            actor = _span_from_tokens(text, actor_tokens)

            if actor is None and candidate_number > 0:
                actor = inherited_actor
            if actor is not None:
                inherited_actor = actor

            tail_end = self._tail_end(tokens, verb_index, candidate_positions)
            patient, qualifiers = self._parse_tail(text, tokens[verb_index + 1 : tail_end])

            confidence = 0.4
            confidence += 0.15 if actor is not None else 0.0
            confidence += 0.15 if patient is not None else 0.0
            confidence += 0.05 if qualifiers else 0.0
            confidence += 0.1 if lemma in self._verbs else 0.0
            confidence = min(confidence, 0.95)

            if confidence < self.config.min_confidence:
                continue

            predicate_token = tokens[verb_index]
            actions.append(
                ActionFrame(
                    actor=actor,
                    predicate=TextSpan(
                        text=text[predicate_token.start : predicate_token.end],
                        start=predicate_token.start,
                        end=predicate_token.end,
                    ),
                    predicate_lemma=lemma,
                    patient=patient,
                    qualifiers=tuple(qualifiers),
                    sentence_index=sentence_index,
                    confidence=confidence,
                    extractor=self.extractor_id,
                )
            )

        return actions

    def _verb_candidates(self, tokens: list[_Token]) -> list[tuple[int, str]]:
        candidates: list[tuple[int, str]] = []

        for index, token in enumerate(tokens):
            lemma = _verb_lemma(token.lower, self._verbs)
            if lemma is None or token.lower in _AUXILIARIES:
                continue

            # Avoid common nominal/adjectival false positives after a
            # determiner, as in "the request" or "the signed contract".
            if index > 0 and tokens[index - 1].lower in _DETERMINERS:
                continue

            candidates.append((index, lemma))

        return candidates

    def _actor_tokens(self, tokens: list[_Token], verb_index: int) -> list[_Token]:
        start = 0
        for index in range(verb_index):
            token = tokens[index]
            if token.text in _CLAUSE_PUNCTUATION or token.lower in _COORDINATORS:
                start = index + 1

        actor_tokens = list(tokens[start:verb_index])
        actor_tokens = _trim_tokens(actor_tokens)

        while actor_tokens and actor_tokens[-1].lower in _AUXILIARIES:
            actor_tokens.pop()
        while actor_tokens and actor_tokens[0].lower in _DISCOURSE_PREFIXES:
            actor_tokens.pop(0)

        # A predicate at the start of a clause is treated as an imperative with
        # an implicit actor rather than inventing one.
        return _trim_tokens(actor_tokens)

    def _tail_end(
        self, tokens: list[_Token], verb_index: int, candidate_positions: set[int]
    ) -> int:
        end = len(tokens)
        for index in range(verb_index + 1, len(tokens)):
            token = tokens[index]
            if token.text in {";", ".", "!", "?"}:
                return index
            if token.lower in _COORDINATORS:
                if any(position > index for position in candidate_positions):
                    return index
        return end

    def _parse_tail(
        self, text: str, tokens: list[_Token]
    ) -> tuple[TextSpan | None, list[Qualifier]]:
        tokens = _trim_tokens(tokens)
        if not tokens:
            return None, []

        first_preposition = next(
            (index for index, token in enumerate(tokens) if token.lower in _PREPOSITIONS),
            len(tokens),
        )
        patient = _span_from_tokens(text, _trim_tokens(tokens[:first_preposition]))
        qualifiers: list[Qualifier] = []

        index = first_preposition
        while index < len(tokens):
            relation_token = tokens[index]
            if relation_token.lower not in _PREPOSITIONS:
                index += 1
                continue

            value_start = index + 1
            value_end = value_start
            while value_end < len(tokens):
                value_token = tokens[value_end]
                if (
                    value_token.lower in _PREPOSITIONS
                    or value_token.lower in _COORDINATORS
                    or value_token.text in _CLAUSE_PUNCTUATION
                ):
                    break
                value_end += 1

            value = _span_from_tokens(text, _trim_tokens(tokens[value_start:value_end]))
            if value is not None:
                qualifiers.append(Qualifier(relation=relation_token.lower, value=value))

            index = max(value_end, index + 1)

        return patient, qualifiers


def _normalise_verb(value: str) -> str:
    return value.strip().casefold()


def _verb_lemma(word: str, verbs: Iterable[str]) -> str | None:
    vocabulary = verbs if isinstance(verbs, (set, frozenset)) else set(verbs)
    if word in _IRREGULAR_LEMMAS:
        lemma = _IRREGULAR_LEMMAS[word]
        return lemma if lemma in vocabulary else None
    if word in vocabulary:
        return word

    candidates: list[str] = []
    if word.endswith("ied") and len(word) > 3:
        candidates.append(word[:-3] + "y")
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        candidates.extend((stem, stem + "e"))
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
    if word.endswith("ed") and len(word) > 3:
        stem = word[:-2]
        candidates.extend((word[:-1], stem))
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
    if word.endswith("es") and len(word) > 3:
        candidates.extend((word[:-2], word[:-1]))
    if word.endswith("s") and len(word) > 2:
        candidates.append(word[:-1])

    return next((candidate for candidate in candidates if candidate in vocabulary), None)


def _trim_tokens(tokens: list[_Token]) -> list[_Token]:
    trimmed = list(tokens)
    while trimmed and (
        trimmed[0].text in _LEADING_PUNCTUATION
        or trimmed[0].lower in _COORDINATORS
    ):
        trimmed.pop(0)
    while trimmed and (
        trimmed[-1].text in _TRAILING_PUNCTUATION
        or trimmed[-1].lower in _COORDINATORS
    ):
        trimmed.pop()
    return trimmed


def _span_from_tokens(text: str, tokens: list[_Token]) -> TextSpan | None:
    if not tokens:
        return None
    start = tokens[0].start
    end = tokens[-1].end
    return TextSpan(text=text[start:end], start=start, end=end)
