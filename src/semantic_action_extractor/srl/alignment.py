"""Align word-level PropBank BIO labels to tokenizer subwords."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .bio import continuation_tag, split_bio_tag


class WordTokenizer(Protocol):
    """The small fast-tokenizer interface needed by this module."""

    def __call__(self, words: list[str], **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AlignedSRLExample:
    """One tokenized predicate-conditioned SRL example.

    ``labels`` uses ``-100`` for special and padding tokens so PyTorch loss can
    ignore them.  ``token_type_ids`` is the predicate indicator used by the
    original model design: exactly the first WordPiece of the supplied
    predicate receives 1; every other position receives 0.
    """

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...]
    labels: tuple[int, ...]
    word_ids: tuple[int | None, ...]

    def to_model_inputs(self) -> dict[str, list[int]]:
        """Return list-valued inputs suitable for a tensorizing data collator."""

        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "token_type_ids": list(self.token_type_ids),
            "labels": list(self.labels),
        }


def _word_ids(encoded: Mapping[str, Any]) -> list[int | None]:
    accessor = getattr(encoded, "word_ids", None)
    if accessor is None:
        raise TypeError("tokenizer output must provide word_ids()")
    try:
        values = accessor()
    except TypeError:
        values = accessor(batch_index=0)
    if values is None:
        raise TypeError("tokenizer output returned no word IDs")
    return list(values)


def _integer_list(encoded: Mapping[str, Any], key: str) -> list[int]:
    if key not in encoded:
        raise KeyError(f"tokenizer output is missing {key!r}")
    values = encoded[key]
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("alignment accepts one sentence at a time")
        values = values[0]
    return [int(value) for value in values]


def align_word_labels(
    tokenizer: WordTokenizer,
    words: Sequence[str],
    tags: Sequence[str],
    predicate_index: int,
    label_to_id: Mapping[str, int],
    *,
    max_length: int | None = None,
    padding: bool | str = False,
    truncation: bool = True,
) -> AlignedSRLExample:
    """Tokenize and align one supplied-predicate SRL example.

    The first subword retains its word's BIO tag.  Each later subword receives
    the corresponding ``I`` tag (``B-ARG1`` becomes ``I-ARG1``); ``O`` remains
    ``O``.  Special and padding positions receive label ``-100``.
    """

    if not words:
        raise ValueError("words cannot be empty")
    if len(words) != len(tags):
        raise ValueError("words and tags must have the same length")
    if type(predicate_index) is not int:
        raise TypeError("predicate index must be an integer")
    if predicate_index < 0 or predicate_index >= len(words):
        raise IndexError("predicate index is outside the sentence")
    for tag in tags:
        split_bio_tag(tag)

    tokenizer_kwargs: dict[str, Any] = {
        "is_split_into_words": True,
        "padding": padding,
        "truncation": truncation,
        "return_attention_mask": True,
    }
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length

    encoded = tokenizer(list(words), **tokenizer_kwargs)
    input_ids = _integer_list(encoded, "input_ids")
    word_ids = _word_ids(encoded)
    if len(word_ids) != len(input_ids):
        raise ValueError("word IDs and input IDs must have the same length")

    visible_word_ids = [word_id for word_id in word_ids if word_id is not None]
    for word_id in visible_word_ids:
        if type(word_id) is not int:
            raise TypeError("tokenizer word IDs must be integers or None")
        if word_id < 0 or word_id >= len(words):
            raise ValueError(f"tokenizer returned invalid word ID: {word_id}")
    visible_word_order = [
        word_id
        for position, word_id in enumerate(visible_word_ids)
        if position == 0 or word_id != visible_word_ids[position - 1]
    ]
    if visible_word_order != list(range(len(words))):
        raise ValueError(
            "tokenization truncated, omitted, or reordered one or more input words"
        )

    if "attention_mask" in encoded:
        attention_mask = _integer_list(encoded, "attention_mask")
    else:
        attention_mask = [1] * len(input_ids)
    if len(attention_mask) != len(input_ids):
        raise ValueError("attention mask and input IDs must have the same length")

    aligned_labels: list[int] = []
    predicate_indicator = [0] * len(input_ids)
    previous_word_id: int | None = None
    predicate_visible = False

    for token_index, word_id in enumerate(word_ids):
        if word_id is None:
            aligned_labels.append(-100)
            previous_word_id = None
            continue
        first_piece = word_id != previous_word_id
        tag = tags[word_id] if first_piece else continuation_tag(tags[word_id])
        try:
            aligned_labels.append(label_to_id[tag])
        except KeyError as error:
            raise KeyError(f"label mapping is missing aligned tag {tag!r}") from error

        if word_id == predicate_index and first_piece:
            predicate_indicator[token_index] = 1
            predicate_visible = True
        previous_word_id = word_id

    if not predicate_visible:
        raise ValueError("tokenization truncated the supplied predicate")

    return AlignedSRLExample(
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        token_type_ids=tuple(predicate_indicator),
        labels=tuple(aligned_labels),
        word_ids=tuple(word_ids),
    )
