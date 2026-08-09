"""Shared, dependency-free validation helpers for corpus adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..annotation_schema import AnnotationToken, TokenAlignedSpan
from ..schema import TextSpan


class DatasetFormatError(ValueError):
    """Raised when an upstream record violates its declared release format."""


def canonicalize_tokens(
    raw_tokens: Sequence[Any],
    *,
    label: str,
) -> tuple[str, tuple[AnnotationToken, ...]]:
    """Join pretokenized input with one space and derive exact character spans."""

    if isinstance(raw_tokens, (str, bytes)) or not raw_tokens:
        raise DatasetFormatError(f"{label} must be a non-empty token sequence")

    values: list[str] = []
    for index, token in enumerate(raw_tokens):
        if not isinstance(token, str) or not token:
            raise DatasetFormatError(f"{label}[{index}] must be a non-empty string")
        # QA-SRL contains a small number of tokens with an internal non-breaking
        # space. Preserve those code points exactly; only reject characters that
        # would conflict with the release's ASCII-space token separator.
        if any(character in {" ", "\t", "\r", "\n"} for character in token):
            raise DatasetFormatError(
                f"{label}[{index}] contains whitespace and is not pretokenized"
            )
        values.append(token)

    text = " ".join(values)
    tokens: list[AnnotationToken] = []
    cursor = 0
    for index, token in enumerate(values):
        start = cursor
        end = start + len(token)
        tokens.append(
            AnnotationToken(
                index=index,
                span=TextSpan(text=token, start=start, end=end),
            )
        )
        cursor = end + 1
    return text, tuple(tokens)


def canonicalize_pretokenized_text(
    sentence: Any,
    *,
    label: str,
) -> tuple[str, tuple[AnnotationToken, ...]]:
    """Validate text that is already represented as space-separated tokens."""

    if not isinstance(sentence, str) or not sentence:
        raise DatasetFormatError(f"{label} must be a non-empty string")
    raw_tokens = sentence.split(" ")
    if any(not token for token in raw_tokens):
        raise DatasetFormatError(f"{label} must use exactly one space between tokens")
    text, tokens = canonicalize_tokens(raw_tokens, label=f"{label} tokens")
    if text != sentence:
        raise DatasetFormatError(f"{label} is not in canonical pretokenized form")
    return text, tokens


def token_aligned_span(
    text: str,
    tokens: Sequence[AnnotationToken],
    token_start: Any,
    token_end: Any,
    *,
    label: str,
) -> TokenAlignedSpan:
    """Convert an upstream half-open token range to a grounded text span."""

    if (
        isinstance(token_start, bool)
        or isinstance(token_end, bool)
        or not isinstance(token_start, int)
        or not isinstance(token_end, int)
    ):
        raise DatasetFormatError(f"{label} token boundaries must be integers")
    if token_start < 0 or token_end <= token_start or token_end > len(tokens):
        raise DatasetFormatError(
            f"{label} token range [{token_start}, {token_end}) is invalid"
        )

    char_start = tokens[token_start].span.start
    char_end = tokens[token_end - 1].span.end
    return TokenAlignedSpan(
        span=TextSpan(
            text=text[char_start:char_end],
            start=char_start,
            end=char_end,
        ),
        token_start=token_start,
        token_end=token_end,
    )


def parse_int(value: Any, *, label: str) -> int:
    """Parse an integer without accepting booleans or lossy numeric strings."""

    if isinstance(value, bool):
        raise DatasetFormatError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    raise DatasetFormatError(f"{label} must be an integer")


def parse_bool(value: Any, *, label: str) -> bool:
    """Parse only explicit upstream boolean encodings."""

    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise DatasetFormatError(f"{label} must be True or False")


def require_string(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise DatasetFormatError(f"{label} must be a string")
    if not allow_empty and not value.strip():
        raise DatasetFormatError(f"{label} cannot be empty")
    return value


def normalize_slot(value: Any, *, label: str) -> str:
    """Represent an empty upstream question slot with the QA-SRL `_` marker."""

    slot = require_string(value, label=label, allow_empty=True)
    return slot if slot else "_"


def infer_qasrl_document_id(sentence_id: str) -> str:
    """Derive the document key encoded by the two QA-SRL identifier schemes."""

    require_string(sentence_id, label="sentence_id")
    if sentence_id.startswith("Wiki1k:"):
        parts = sentence_id.split(":")
        if len(parts) < 5 or not all(parts[:3]):
            raise DatasetFormatError(
                f"unsupported Wiki1k sentence identifier: {sentence_id}"
            )
        return ":".join(parts[:3])
    if sentence_id.startswith("TQA:"):
        prefix, separator, sentence_number = sentence_id.rpartition("_")
        if not separator or not prefix or not sentence_number.isdigit():
            raise DatasetFormatError(
                f"unsupported TQA sentence identifier: {sentence_id}"
            )
        return prefix
    raise DatasetFormatError(f"unsupported QA-SRL sentence identifier: {sentence_id}")


def split_sources(value: Any, *, label: str) -> tuple[str, ...]:
    """Parse QANom's `~!~`-separated source fields without empty IDs."""

    source = require_string(value, label=label, allow_empty=True)
    if not source:
        return ()
    parts = tuple(part for part in source.split("~!~") if part)
    if not parts or len(parts) != len(set(parts)):
        raise DatasetFormatError(f"{label} contains empty or duplicate source IDs")
    return parts
