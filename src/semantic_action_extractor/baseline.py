"""Transparent rule baseline for source-grounded action extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import tomllib
from typing import Any, Iterable

from .schema import ActionArgument, ActionFrame, ExtractionResult, TextSpan


_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.MULTILINE)
_TOKEN_RE = re.compile(
    r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\d+(?:[.,]\d+)*|[^\w\s]",
    re.UNICODE,
)

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
_HARD_CLAUSE_BOUNDARIES = frozenset({";", ":", ".", "!", "?"})
_TRAILING_PUNCTUATION = frozenset({",", ";", ":", ".", "!", "?", ")", "]", "}"})
_LEADING_PUNCTUATION = frozenset({",", ";", ":", "(", "[", "{"})


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Configuration for the rule baseline."""

    additional_verbs: tuple[str, ...] = ()
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_score, bool)
            or not isinstance(self.min_score, (int, float))
            or not math.isfinite(self.min_score)
            or not 0.0 <= self.min_score <= 1.0
        ):
            raise ValueError("min_score must be between 0 and 1")

        if isinstance(self.additional_verbs, str) or not all(
            isinstance(verb, str) for verb in self.additional_verbs
        ):
            raise TypeError("additional_verbs must contain only strings")

        normalised_verbs = tuple(
            _normalise_verb(verb) for verb in self.additional_verbs
        )
        invalid = [
            original
            for original, normalised in zip(
                self.additional_verbs, normalised_verbs
            )
            if not normalised
        ]
        if invalid:
            raise ValueError(f"additional verbs cannot be empty: {invalid!r}")
        object.__setattr__(self, "additional_verbs", normalised_verbs)

    @classmethod
    def from_toml(cls, path: str | Path) -> "BaselineConfig":
        """Load `[baseline]` values from a TOML file."""

        config_path = Path(path)
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
        section = document.get("baseline", {})
        if not isinstance(section, dict):
            raise ValueError("[baseline] must be a TOML table")

        allowed = {"additional_verbs", "min_score"}
        unknown = sorted(set(section) - allowed)
        if unknown:
            raise ValueError(
                f"unknown baseline configuration keys: {', '.join(unknown)}"
            )

        verbs = section.get("additional_verbs", [])
        if not isinstance(verbs, list) or not all(
            isinstance(item, str) for item in verbs
        ):
            raise ValueError("baseline.additional_verbs must be an array of strings")

        threshold = section.get("min_score", 0.0)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("baseline.min_score must be a number")

        return cls(
            additional_verbs=tuple(verbs),
            min_score=float(threshold),
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

    extractor_id = "rule-based-v1"

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
                    "rule-based-v1 does not encode action polarity."
                )

        if text.strip() and not actions:
            warnings.append("No action predicates matched the rule-based vocabulary.")

        return ExtractionResult(
            text=text,
            actions=tuple(actions),
            warnings=tuple(warnings),
        )

    def _extract_sentence(
        self, text: str, tokens: list[_Token], sentence_index: int
    ) -> list[ActionFrame]:
        candidates = self._verb_candidates(tokens)
        candidate_positions = {position for position, _ in candidates}
        actions: list[ActionFrame] = []
        previous_verb_index: int | None = None
        previous_left_context: TextSpan | None = None

        for verb_index, lemma in candidates:
            left_context_tokens = self._left_context_tokens(tokens, verb_index)
            left_context = _span_from_tokens(text, left_context_tokens)

            if (
                left_context is None
                and previous_verb_index is not None
                and previous_left_context is not None
                and self._shares_left_context(
                    tokens, previous_verb_index, verb_index
                )
            ):
                left_context = previous_left_context

            previous_verb_index = verb_index
            previous_left_context = left_context

            tail_end = self._tail_end(tokens, verb_index, candidate_positions)
            right_arguments = self._parse_tail(
                text, tokens[verb_index + 1 : tail_end]
            )

            arguments: list[ActionArgument] = []
            if left_context is not None:
                arguments.append(
                    ActionArgument(
                        role="before_predicate",
                        role_scheme="surface",
                        span=left_context,
                    )
                )
            arguments.extend(right_arguments)

            has_direct_right_context = any(
                argument.role == "after_predicate" for argument in right_arguments
            )
            has_prepositional_context = any(
                argument.cue is not None for argument in right_arguments
            )
            score_points = 35
            score_points += 20 if left_context is not None else 0
            score_points += 20 if has_direct_right_context else 0
            score_points += 10 if has_prepositional_context else 0
            score = score_points / 100

            if score < self.config.min_score:
                continue

            predicate_token = tokens[verb_index]
            actions.append(
                ActionFrame(
                    predicate=TextSpan(
                        text=text[predicate_token.start : predicate_token.end],
                        start=predicate_token.start,
                        end=predicate_token.end,
                    ),
                    predicate_lemma=lemma,
                    predicate_type="verbal",
                    arguments=tuple(arguments),
                    sentence_index=sentence_index,
                    score=score,
                    score_type="heuristic_completeness",
                    extractor=self.extractor_id,
                )
            )

        return actions

    def _shares_left_context(
        self, tokens: list[_Token], previous_verb_index: int, verb_index: int
    ) -> bool:
        bridge = tokens[previous_verb_index + 1 : verb_index]
        if any(token.text in _HARD_CLAUSE_BOUNDARIES for token in bridge):
            return False
        return any(token.lower in _COORDINATORS for token in bridge)

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

    def _left_context_tokens(
        self, tokens: list[_Token], verb_index: int
    ) -> list[_Token]:
        start = 0
        for index in range(verb_index):
            token = tokens[index]
            if token.text in _CLAUSE_PUNCTUATION or token.lower in _COORDINATORS:
                start = index + 1

        context_tokens = list(tokens[start:verb_index])
        context_tokens = _trim_tokens(context_tokens)

        while context_tokens and context_tokens[-1].lower in _AUXILIARIES:
            context_tokens.pop()
        while context_tokens and context_tokens[0].lower in _DISCOURSE_PREFIXES:
            context_tokens.pop(0)

        # Do not invent a left-context argument when the predicate begins a clause.
        return _trim_tokens(context_tokens)

    def _tail_end(
        self, tokens: list[_Token], verb_index: int, candidate_positions: set[int]
    ) -> int:
        end = len(tokens)
        for index in range(verb_index + 1, len(tokens)):
            token = tokens[index]
            if token.text in _HARD_CLAUSE_BOUNDARIES:
                return index
            if token.lower in _COORDINATORS:
                if any(position > index for position in candidate_positions):
                    return index
        return end

    def _parse_tail(self, text: str, tokens: list[_Token]) -> list[ActionArgument]:
        tokens = _trim_tokens(tokens)
        if not tokens:
            return []

        first_preposition = next(
            (
                index
                for index, token in enumerate(tokens)
                if token.lower in _PREPOSITIONS
            ),
            len(tokens),
        )
        arguments: list[ActionArgument] = []
        direct_span = _span_from_tokens(text, _trim_tokens(tokens[:first_preposition]))
        if direct_span is not None:
            arguments.append(
                ActionArgument(
                    role="after_predicate",
                    role_scheme="surface",
                    span=direct_span,
                )
            )

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
                arguments.append(
                    ActionArgument(
                        role=relation_token.lower,
                        role_scheme="surface",
                        span=value,
                        cue=TextSpan(
                            text=text[relation_token.start : relation_token.end],
                            start=relation_token.start,
                            end=relation_token.end,
                        ),
                    )
                )

            index = max(value_end, index + 1)

        return arguments


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

    return next(
        (candidate for candidate in candidates if candidate in vocabulary),
        None,
    )


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
