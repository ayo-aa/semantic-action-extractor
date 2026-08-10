"""BIO repair and exact-span decoding for PropBank role labels.

The decoder uses one explicit repair rule: an ``I-X`` tag that does not follow
``B-X`` or ``I-X`` is treated as ``B-X``.  This is the conventional lenient
BIO interpretation and makes evaluation deterministic even when a model emits
an invalid transition.  Other malformed tags are rejected rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True, slots=True)
class LabeledSpan:
    """A word-indexed, end-exclusive span with its PropBank role label."""

    label: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("span label must be a string")
        if not self.label:
            raise ValueError("span label cannot be empty")
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("span indexes must be integers")
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")


def split_bio_tag(tag: str) -> tuple[str, str | None]:
    """Split one BIO tag into its prefix and role.

    ``O`` has no role.  Roles may themselves contain hyphens, as in
    ``B-ARGM-TMP``.
    """

    if not isinstance(tag, str):
        raise TypeError("BIO tag must be a string")
    if tag == "O":
        return "O", None
    prefix, separator, label = tag.partition("-")
    if not separator or prefix not in {"B", "I"} or not label:
        raise ValueError(f"invalid BIO tag: {tag!r}")
    return prefix, label


def continuation_tag(tag: str) -> str:
    """Return the tag for a continuation WordPiece of the same word."""

    prefix, label = split_bio_tag(tag)
    if prefix == "O":
        return "O"
    return f"I-{label}"


def repair_bio(tags: Iterable[str]) -> tuple[str, ...]:
    """Repair orphaned or role-mismatched ``I`` tags into ``B`` tags."""

    repaired: list[str] = []
    previous_prefix = "O"
    previous_label: str | None = None

    for tag in tags:
        prefix, label = split_bio_tag(tag)
        if prefix == "I" and not (
            previous_prefix in {"B", "I"} and previous_label == label
        ):
            prefix = "B"
            tag = f"B-{label}"
        repaired.append(tag)
        previous_prefix = prefix
        previous_label = label

    return tuple(repaired)


def decode_bio(tags: Sequence[str], *, repair: bool = True) -> tuple[LabeledSpan, ...]:
    """Decode word-level BIO tags into exact labeled spans.

    Args:
        tags: One BIO tag per sentence word.
        repair: Apply :func:`repair_bio` before decoding.  When false, an
            invalid ``I`` transition raises ``ValueError``.
    """

    normalized = repair_bio(tags) if repair else tuple(tags)
    spans: list[LabeledSpan] = []
    open_label: str | None = None
    open_start = 0

    def close_span(end: int) -> None:
        nonlocal open_label
        if open_label is not None:
            spans.append(LabeledSpan(open_label, open_start, end))
            open_label = None

    previous_prefix = "O"
    previous_label: str | None = None
    for index, tag in enumerate(normalized):
        prefix, label = split_bio_tag(tag)
        if not repair and prefix == "I" and not (
            previous_prefix in {"B", "I"} and previous_label == label
        ):
            raise ValueError(f"invalid I-tag transition at word {index}: {tag!r}")

        if prefix == "O":
            close_span(index)
        elif prefix == "B":
            close_span(index)
            open_label = label
            open_start = index
        elif open_label != label:
            # This branch is reachable only if callers bypass repair with an
            # invalid sequence; the explicit check above supplies the error.
            raise ValueError(f"invalid I-tag transition at word {index}: {tag!r}")

        previous_prefix = prefix
        previous_label = label

    close_span(len(normalized))
    return tuple(spans)
