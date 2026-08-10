"""Dependency-free preparation and collation for supplied-predicate SRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .alignment import AlignedSRLExample, WordTokenizer, align_word_labels
from .bio import continuation_tag
from .example import DatasetSplit, WordLevelSRLExample
from .label_vocabulary import SRLLabelVocabulary


DropReason = Literal["tokenized_length_exceeds_max_length"]
_VALID_SPLITS = frozenset({"train", "development", "test"})


@dataclass(frozen=True, slots=True)
class OverlengthDrop:
    """One explicitly reported example excluded before model collation."""

    example_id: str
    tokenized_length: int
    max_length: int
    reason: DropReason = "tokenized_length_exceeds_max_length"

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str):
            raise TypeError("drop example ID must be a string")
        if not self.example_id.strip():
            raise ValueError("drop example ID cannot be empty")
        for name in ("tokenized_length", "max_length"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tokenized_length <= self.max_length:
            raise ValueError("an overlength drop must exceed max_length")
        if self.reason != "tokenized_length_exceeds_max_length":
            raise ValueError("unsupported overlength drop reason")


@dataclass(frozen=True, slots=True)
class AlignedModelExample:
    """Aligned tensors plus the source-neutral metadata needed for decoding."""

    example_id: str
    word_count: int
    alignment: AlignedSRLExample

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str):
            raise TypeError("example ID must be a string")
        if not self.example_id.strip():
            raise ValueError("example ID cannot be empty")
        if type(self.word_count) is not int:
            raise TypeError("word count must be an integer")
        if self.word_count <= 0:
            raise ValueError("word count must be positive")
        if not isinstance(self.alignment, AlignedSRLExample):
            raise TypeError("alignment must be an AlignedSRLExample")

        fields = (
            self.alignment.input_ids,
            self.alignment.attention_mask,
            self.alignment.token_type_ids,
            self.alignment.labels,
            self.alignment.word_ids,
        )
        if any(not isinstance(values, tuple) for values in fields):
            raise TypeError("aligned token fields must be immutable tuples")
        lengths = {len(values) for values in fields}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("aligned token fields must have one nonempty length")

        visible_word_order: list[int] = []
        previous_word_id: int | None = None
        for token_index, (input_id, mask, token_type_id, label, word_id) in enumerate(
            zip(*fields, strict=True)
        ):
            if type(input_id) is not int or input_id < 0:
                raise ValueError(f"invalid input ID at token {token_index}")
            if type(mask) is not int or mask not in {0, 1}:
                raise ValueError(f"invalid attention mask at token {token_index}")
            if type(token_type_id) is not int or token_type_id not in {0, 1}:
                raise ValueError(f"invalid token type ID at token {token_index}")
            if type(label) is not int:
                raise TypeError(f"label at token {token_index} must be an integer")
            if word_id is not None and type(word_id) is not int:
                raise TypeError("word IDs must be integers or None")

            if word_id is None:
                if label != -100:
                    raise ValueError("special and padding labels must be -100")
                if token_type_id != 0:
                    raise ValueError(
                        "special and padding predicate indicators must be 0"
                    )
                previous_word_id = None
                continue
            if mask != 1:
                raise ValueError("a word piece cannot occupy a padding position")
            if word_id < 0 or word_id >= self.word_count:
                raise ValueError(f"invalid word ID at token {token_index}: {word_id}")
            if label < 0:
                raise ValueError("word-piece labels must be non-negative")
            if word_id != previous_word_id:
                visible_word_order.append(word_id)
            previous_word_id = word_id

        if visible_word_order != list(range(self.word_count)):
            raise ValueError("word IDs must cover each input word once in order")
        if sum(self.alignment.token_type_ids) != 1:
            raise ValueError("aligned examples must have one predicate indicator")


@dataclass(frozen=True, slots=True)
class PreparedSRLSplit:
    """A selected split with retained examples and explicit drop accounting."""

    split: DatasetSplit
    total_records: int
    selected_records: int
    skipped_other_splits: int
    examples: tuple[AlignedModelExample, ...]
    overlength_drops: tuple[OverlengthDrop, ...]

    def __post_init__(self) -> None:
        if self.split not in _VALID_SPLITS:
            raise ValueError("invalid prepared split")
        for name in ("total_records", "selected_records", "skipped_other_splits"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total_records != self.selected_records + self.skipped_other_splits:
            raise ValueError("prepared split record counts do not reconcile")
        if self.selected_records != len(self.examples) + len(self.overlength_drops):
            raise ValueError("retained and dropped example counts do not reconcile")
        if not isinstance(self.examples, tuple):
            raise TypeError("prepared examples must be a tuple")
        if not isinstance(self.overlength_drops, tuple):
            raise TypeError("overlength drops must be a tuple")
        if any(
            not isinstance(example, AlignedModelExample)
            for example in self.examples
        ):
            raise TypeError("every prepared example must be an AlignedModelExample")
        if any(
            not isinstance(drop, OverlengthDrop) for drop in self.overlength_drops
        ):
            raise TypeError("every reported drop must be an OverlengthDrop")

        reported_ids = [example.example_id for example in self.examples]
        reported_ids.extend(drop.example_id for drop in self.overlength_drops)
        if len(reported_ids) != len(set(reported_ids)):
            raise ValueError("prepared split contains duplicate example IDs")

    @property
    def retained_count(self) -> int:
        return len(self.examples)

    @property
    def overlength_drop_count(self) -> int:
        return len(self.overlength_drops)


@dataclass(frozen=True, slots=True)
class PaddedSRLBatch:
    """Immutable padded values ready for a framework tensorization layer."""

    example_ids: tuple[str, ...]
    word_counts: tuple[int, ...]
    word_ids: tuple[tuple[int | None, ...], ...]
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    token_type_ids: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.example_ids, tuple):
            raise TypeError("batch example IDs must be an immutable tuple")
        if any(
            not isinstance(example_id, str) or not example_id.strip()
            for example_id in self.example_ids
        ):
            raise ValueError("batch example IDs must be nonempty strings")
        batch_size = len(self.example_ids)
        if batch_size == 0:
            raise ValueError("a padded batch cannot be empty")
        if len(set(self.example_ids)) != batch_size:
            raise ValueError("batch example IDs must be unique")
        fields: tuple[tuple[Any, ...], ...] = (
            self.word_counts,
            self.word_ids,
            self.input_ids,
            self.attention_mask,
            self.token_type_ids,
            self.labels,
        )
        if any(not isinstance(field, tuple) for field in fields):
            raise TypeError("batch fields must be immutable tuples")
        if any(len(field) != batch_size for field in fields):
            raise ValueError("all batch fields must have the same batch size")
        if any(
            not isinstance(row, tuple) for field in fields[1:] for row in field
        ):
            raise TypeError("padded token rows must be immutable tuples")
        if any(
            type(word_count) is not int or word_count <= 0
            for word_count in self.word_counts
        ):
            raise ValueError("batch word counts must be positive integers")

        row_widths = {len(row) for field in fields[1:] for row in field}
        if len(row_widths) != 1 or not row_widths or next(iter(row_widths)) == 0:
            raise ValueError("all padded token rows must have one nonzero width")

    def to_model_inputs(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        """Return only numerical model fields, preserving immutable rows."""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "token_type_ids": self.token_type_ids,
            "labels": self.labels,
        }


def _validate_selected_labels(
    examples: tuple[WordLevelSRLExample, ...],
    vocabulary: SRLLabelVocabulary,
    split: DatasetSplit,
) -> None:
    for example in examples:
        required_labels = set(example.tags)
        required_labels.update(continuation_tag(tag) for tag in example.tags)
        missing = sorted(required_labels.difference(vocabulary.label_to_id))
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(
                f"{split} example {example.example_id!r} contains labels absent "
                f"from the training vocabulary: {missing_text}"
            )


def prepare_srl_split(
    tokenizer: WordTokenizer,
    records: Iterable[WordLevelSRLExample],
    vocabulary: SRLLabelVocabulary,
    split: DatasetSplit,
    *,
    max_length: int,
) -> PreparedSRLSplit:
    """Align one selected split without tokenizer-side truncation.

    Input order is preserved. Examples whose complete tokenization exceeds
    ``max_length`` are excluded only after their true length is measured and
    are returned in ``overlength_drops``.
    """

    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    if not isinstance(vocabulary, SRLLabelVocabulary):
        raise TypeError("vocabulary must be an SRLLabelVocabulary")
    if split not in _VALID_SPLITS:
        raise ValueError("split must be 'train', 'development', or 'test'")
    if type(max_length) is not int:
        raise TypeError("max_length must be an integer")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if isinstance(records, (str, bytes)):
        raise TypeError("records must be an iterable of WordLevelSRLExample values")
    try:
        all_records = tuple(records)
    except TypeError as error:
        raise TypeError(
            "records must be an iterable of WordLevelSRLExample values"
        ) from error

    seen_ids: set[str] = set()
    for index, record in enumerate(all_records):
        if not isinstance(record, WordLevelSRLExample):
            raise TypeError(f"record {index} must be a WordLevelSRLExample")
        if record.example_id in seen_ids:
            raise ValueError(f"duplicate example ID: {record.example_id!r}")
        seen_ids.add(record.example_id)

    selected = tuple(record for record in all_records if record.split == split)
    _validate_selected_labels(selected, vocabulary, split)

    retained: list[AlignedModelExample] = []
    drops: list[OverlengthDrop] = []
    for example in selected:
        aligned = align_word_labels(
            tokenizer,
            example.words,
            example.tags,
            example.predicate_index,
            vocabulary.label_to_id,
            padding=False,
            truncation=False,
        )
        model_example = AlignedModelExample(
            example_id=example.example_id,
            word_count=len(example.words),
            alignment=aligned,
        )
        tokenized_length = len(aligned.input_ids)
        if tokenized_length > max_length:
            drops.append(
                OverlengthDrop(
                    example_id=example.example_id,
                    tokenized_length=tokenized_length,
                    max_length=max_length,
                )
            )
            continue
        retained.append(model_example)

    return PreparedSRLSplit(
        split=split,
        total_records=len(all_records),
        selected_records=len(selected),
        skipped_other_splits=len(all_records) - len(selected),
        examples=tuple(retained),
        overlength_drops=tuple(drops),
    )


def collate_srl_batch(
    examples: Iterable[AlignedModelExample],
    *,
    pad_token_id: int,
    predicate_signal: bool,
) -> PaddedSRLBatch:
    """Right-pad examples and optionally zero the predicate indicator."""

    if type(pad_token_id) is not int:
        raise TypeError("pad token ID must be an integer")
    if pad_token_id < 0:
        raise ValueError("pad token ID must be non-negative")
    if type(predicate_signal) is not bool:
        raise TypeError("predicate signal must be a boolean")
    if isinstance(examples, (str, bytes)):
        raise TypeError("examples must be an iterable of AlignedModelExample values")
    try:
        rows = tuple(examples)
    except TypeError as error:
        raise TypeError(
            "examples must be an iterable of AlignedModelExample values"
        ) from error
    if not rows:
        raise ValueError("cannot collate an empty batch")
    for index, example in enumerate(rows):
        if not isinstance(example, AlignedModelExample):
            raise TypeError(f"example {index} must be an AlignedModelExample")
    example_ids = tuple(example.example_id for example in rows)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("cannot collate duplicate example IDs")

    width = max(len(example.alignment.input_ids) for example in rows)

    def pad_ints(values: tuple[int, ...], value: int) -> tuple[int, ...]:
        return values + (value,) * (width - len(values))

    def pad_word_ids(
        values: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        return values + (None,) * (width - len(values))

    input_ids = tuple(
        pad_ints(example.alignment.input_ids, pad_token_id) for example in rows
    )
    attention_mask = tuple(
        pad_ints(example.alignment.attention_mask, 0) for example in rows
    )
    labels = tuple(pad_ints(example.alignment.labels, -100) for example in rows)
    word_ids = tuple(pad_word_ids(example.alignment.word_ids) for example in rows)
    if predicate_signal:
        token_type_ids = tuple(
            pad_ints(example.alignment.token_type_ids, 0) for example in rows
        )
    else:
        token_type_ids = tuple((0,) * width for _ in rows)

    return PaddedSRLBatch(
        example_ids=example_ids,
        word_counts=tuple(example.word_count for example in rows),
        word_ids=word_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels,
    )
