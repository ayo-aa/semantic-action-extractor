"""Source-neutral word-level records for supplied-predicate SRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .bio import decode_bio, split_bio_tag


DatasetSplit = Literal["train", "development", "test"]
_VALID_SPLITS = frozenset({"train", "development", "test"})


def _validate_common_fields(
    example: PreparedWordLevelSRLExample | WordLevelSRLExample,
) -> None:
    for name in ("example_id", "document_id", "sentence_id"):
        value = getattr(example, name)
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value.strip():
            raise ValueError(f"{name} cannot be empty")

    if not isinstance(example.words, tuple):
        raise TypeError("words must be a tuple")
    if not example.words:
        raise ValueError("words cannot be empty")
    for word in example.words:
        if not isinstance(word, str):
            raise TypeError("every word must be a string")
        if not word:
            raise ValueError("words cannot contain empty strings")

    if type(example.predicate_index) is not int:
        raise TypeError("predicate_index must be an integer")
    if example.predicate_index < 0 or example.predicate_index >= len(example.words):
        raise IndexError("predicate_index is outside the sentence")

    if not isinstance(example.tags, tuple):
        raise TypeError("tags must be a tuple")
    if len(example.tags) != len(example.words):
        raise ValueError("words and tags must have the same length")
    for tag in example.tags:
        split_bio_tag(tag)
    decode_bio(example.tags, repair=False)

    if example.tags[example.predicate_index] != "B-V":
        raise ValueError("the supplied predicate must have tag 'B-V'")
    if example.tags.count("B-V") != 1:
        raise ValueError("tags must contain exactly one 'B-V' anchor")

    if not isinstance(example.predicate_roleset, str):
        raise TypeError("predicate_roleset must be a string")
    if not example.predicate_roleset.strip():
        raise ValueError("predicate_roleset cannot be empty")


@dataclass(frozen=True, slots=True)
class PreparedWordLevelSRLExample:
    """One validated example before the leakage-safe split is frozen."""

    example_id: str
    document_id: str
    sentence_id: str
    words: tuple[str, ...]
    predicate_index: int
    tags: tuple[str, ...]
    predicate_roleset: str

    def __post_init__(self) -> None:
        _validate_common_fields(self)

    def assign_split(self, split: DatasetSplit) -> WordLevelSRLExample:
        """Create the final model record after document-level split assignment."""

        return WordLevelSRLExample(
            example_id=self.example_id,
            document_id=self.document_id,
            sentence_id=self.sentence_id,
            split=split,
            words=self.words,
            predicate_index=self.predicate_index,
            tags=self.tags,
            predicate_roleset=self.predicate_roleset,
        )


@dataclass(frozen=True, slots=True)
class WordLevelSRLExample:
    """One corpus-independent sentence/predicate example.

    Corpus-specific pointers, parse terminals, link metadata, and rejection
    details belong in an adapter provenance record keyed by ``example_id``.
    Keeping that information out of this class prevents data-source quirks
    from leaking into the neural model boundary.
    """

    example_id: str
    document_id: str
    sentence_id: str
    split: DatasetSplit
    words: tuple[str, ...]
    predicate_index: int
    tags: tuple[str, ...]
    predicate_roleset: str

    def __post_init__(self) -> None:
        if self.split not in _VALID_SPLITS:
            raise ValueError(
                "split must be 'train', 'development', or 'test'"
            )
        _validate_common_fields(self)
