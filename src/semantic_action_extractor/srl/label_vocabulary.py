"""Leakage-safe BIO label vocabulary for supplied-predicate SRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

from .bio import continuation_tag, split_bio_tag
from .example import WordLevelSRLExample


@dataclass(frozen=True, slots=True)
class SRLLabelVocabulary:
    """An immutable, contiguous mapping between BIO labels and integer IDs.

    ``O`` is always ID zero. Remaining labels are sorted lexicographically,
    and every non-``O`` label has its ``I-*`` continuation in the vocabulary
    so :func:`alignment.align_word_labels` cannot create an unknown label.
    """

    labels: tuple[str, ...]
    label_to_id: Mapping[str, int] = field(
        init=False, repr=False, compare=False, hash=False
    )
    id_to_label: Mapping[int, str] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.labels, tuple):
            raise TypeError("labels must be a tuple")
        if not self.labels:
            raise ValueError("labels cannot be empty")

        for label in self.labels:
            split_bio_tag(label)
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels cannot contain duplicates")
        if self.labels[0] != "O":
            raise ValueError("O must be the first label")
        if self.labels != ("O", *sorted(self.labels[1:])):
            raise ValueError("labels after O must be in deterministic sorted order")

        missing_continuations = sorted(
            {
                continuation_tag(label)
                for label in self.labels
                if label != "O"
            }.difference(self.labels)
        )
        if missing_continuations:
            missing = ", ".join(missing_continuations)
            raise ValueError(f"vocabulary is missing continuation labels: {missing}")

        label_to_id = {label: index for index, label in enumerate(self.labels)}
        id_to_label = {index: label for index, label in enumerate(self.labels)}
        object.__setattr__(self, "label_to_id", MappingProxyType(label_to_id))
        object.__setattr__(self, "id_to_label", MappingProxyType(id_to_label))

    def __len__(self) -> int:
        return len(self.labels)

    def encode(self, label: str) -> int:
        """Return the integer ID for one known, well-formed BIO label."""

        split_bio_tag(label)
        try:
            return self.label_to_id[label]
        except KeyError as error:
            raise KeyError(f"unknown BIO label: {label!r}") from error

    def decode(self, label_id: int) -> str:
        """Return the BIO label for one in-range integer ID."""

        if type(label_id) is not int:
            raise TypeError("label ID must be an integer")
        if label_id < 0 or label_id >= len(self.labels):
            raise IndexError(f"label ID is outside the vocabulary: {label_id}")
        return self.id_to_label[label_id]


def build_training_label_vocabulary(
    examples: Iterable[WordLevelSRLExample],
) -> SRLLabelVocabulary:
    """Build a BIO vocabulary from nonempty training records only.

    Development and test records are rejected instead of ignored so their
    label inventory cannot leak into model configuration. Duplicate example
    IDs are also rejected because they make the training inventory ambiguous.
    """

    if isinstance(examples, (str, bytes)):
        raise TypeError("examples must be an iterable of WordLevelSRLExample records")
    try:
        records = tuple(examples)
    except TypeError as error:
        raise TypeError(
            "examples must be an iterable of WordLevelSRLExample records"
        ) from error
    if not records:
        raise ValueError("at least one training example is required")

    observed_labels = {"O"}
    seen_example_ids: set[str] = set()
    for index, example in enumerate(records):
        if not isinstance(example, WordLevelSRLExample):
            raise TypeError(
                f"example {index} must be a WordLevelSRLExample training record"
            )
        if example.split != "train":
            raise ValueError(
                f"example {index} has split {example.split!r}; "
                "label vocabulary must be built from training records only"
            )
        if example.example_id in seen_example_ids:
            raise ValueError(f"duplicate training example ID: {example.example_id!r}")
        seen_example_ids.add(example.example_id)

        for tag in example.tags:
            split_bio_tag(tag)
            observed_labels.add(tag)
            observed_labels.add(continuation_tag(tag))

    labels = ("O", *sorted(observed_labels.difference({"O"})))
    return SRLLabelVocabulary(labels=labels)
