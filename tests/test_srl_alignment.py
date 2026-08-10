import unittest

from semantic_action_extractor.srl.alignment import (
    align_word_labels,
    collapse_subword_predictions,
)


LABELS = {
    "O": 0,
    "B-ARG0": 1,
    "I-ARG0": 2,
    "B-V": 3,
    "I-V": 4,
    "B-ARG1": 5,
    "I-ARG1": 6,
}


class FakeEncoding(dict):
    def __init__(self, word_ids, **values):
        super().__init__(values)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    def __init__(self, encoding):
        self.encoding = encoding
        self.words = None
        self.kwargs = None

    def __call__(self, words, **kwargs):
        self.words = words
        self.kwargs = kwargs
        return self.encoding


class AlignmentTests(unittest.TestCase):
    def test_aligns_bio_and_marks_only_first_predicate_piece(self) -> None:
        tokenizer = FakeTokenizer(
            FakeEncoding(
                [None, 0, 1, 1, 2, None, None],
                input_ids=[101, 10, 20, 21, 30, 102, 0],
                attention_mask=[1, 1, 1, 1, 1, 1, 0],
            )
        )

        aligned = align_word_labels(
            tokenizer,
            ["They", "retrained", "staff"],
            ["B-ARG0", "B-V", "B-ARG1"],
            predicate_index=1,
            label_to_id=LABELS,
            max_length=7,
            padding="max_length",
        )

        self.assertEqual(aligned.labels, (-100, 1, 3, 4, 5, -100, -100))
        self.assertEqual(aligned.token_type_ids, (0, 0, 1, 0, 0, 0, 0))
        self.assertEqual(sum(aligned.token_type_ids), 1)
        self.assertEqual(tokenizer.words, ["They", "retrained", "staff"])
        self.assertTrue(tokenizer.kwargs["is_split_into_words"])
        self.assertEqual(aligned.to_model_inputs()["labels"], list(aligned.labels))

    def test_turns_argument_b_into_i_on_continuation_pieces(self) -> None:
        tokenizer = FakeTokenizer(
            FakeEncoding(
                [None, 0, 0, 1, None],
                input_ids=[101, 10, 11, 20, 102],
                attention_mask=[1, 1, 1, 1, 1],
            )
        )

        aligned = align_word_labels(
            tokenizer,
            ["Warehouse", "closed"],
            ["B-ARG0", "B-V"],
            predicate_index=1,
            label_to_id=LABELS,
        )

        self.assertEqual(aligned.labels, (-100, 1, 2, 3, -100))

    def test_rejects_tokenization_that_truncates_predicate(self) -> None:
        tokenizer = FakeTokenizer(
            FakeEncoding(
                [None, 0, None],
                input_ids=[101, 10, 102],
                attention_mask=[1, 1, 1],
            )
        )

        with self.assertRaisesRegex(ValueError, "truncated"):
            align_word_labels(
                tokenizer,
                ["They", "closed"],
                ["B-ARG0", "B-V"],
                predicate_index=1,
                label_to_id=LABELS,
            )

    def test_rejects_truncated_tail_when_predicate_is_still_visible(self) -> None:
        tokenizer = FakeTokenizer(
            FakeEncoding(
                [None, 0, 1, None],
                input_ids=[101, 10, 20, 102],
                attention_mask=[1, 1, 1, 1],
            )
        )

        with self.assertRaisesRegex(ValueError, "truncated"):
            align_word_labels(
                tokenizer,
                ["They", "closed", "stores"],
                ["B-ARG0", "B-V", "B-ARG1"],
                predicate_index=1,
                label_to_id=LABELS,
            )

    def test_validates_word_and_tag_lengths(self) -> None:
        tokenizer = FakeTokenizer({})
        with self.assertRaisesRegex(ValueError, "same length"):
            align_word_labels(tokenizer, ["closed"], [], 0, LABELS)
        with self.assertRaisesRegex(TypeError, "integer"):
            align_word_labels(tokenizer, ["closed"], ["B-V"], False, LABELS)


class PredictionCollapseTests(unittest.TestCase):
    def test_keeps_first_piece_and_ignores_special_and_padding_tokens(self) -> None:
        collapsed = collapse_subword_predictions(
            [99, 4, 5, 3, 99, 99],
            [None, 0, 0, 1, None, None],
            2,
            attention_mask=[1, 1, 1, 1, 1, 0],
        )

        self.assertEqual(collapsed, (4, 3))

    def test_policy_is_first_piece_even_when_continuation_disagrees(self) -> None:
        collapsed = collapse_subword_predictions(
            ["special", "B-ARG0", "O", "B-V", "special"],
            [None, 0, 0, 1, None],
            2,
        )

        self.assertEqual(collapsed, ("B-ARG0", "B-V"))

    def test_rejects_token_and_mask_shape_mismatches(self) -> None:
        with self.assertRaisesRegex(ValueError, "predictions and word IDs"):
            collapse_subword_predictions([1], [None, 0], 1)
        with self.assertRaisesRegex(ValueError, "attention mask and word IDs"):
            collapse_subword_predictions(
                [1, 2], [None, 0], 1, attention_mask=[1]
            )
        with self.assertRaisesRegex(TypeError, "scalar value"):
            collapse_subword_predictions([[1, 2]], [0], 1)

    def test_rejects_invalid_attention_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer 0 or 1"):
            collapse_subword_predictions(
                [1, 2], [None, 0], 1, attention_mask=[1, 2]
            )
        with self.assertRaisesRegex(ValueError, "integer 0 or 1"):
            collapse_subword_predictions(
                [1, 2], [None, 0], 1, attention_mask=[1, True]
            )

    def test_rejects_padding_that_references_a_word(self) -> None:
        with self.assertRaisesRegex(ValueError, "padding token"):
            collapse_subword_predictions(
                [1, 2], [None, 0], 1, attention_mask=[1, 0]
            )

    def test_rejects_invalid_word_ids(self) -> None:
        with self.assertRaisesRegex(TypeError, "integers or None"):
            collapse_subword_predictions([1], [False], 1)
        with self.assertRaisesRegex(ValueError, "invalid word ID"):
            collapse_subword_predictions([1], [-1], 1)
        with self.assertRaisesRegex(ValueError, "invalid word ID"):
            collapse_subword_predictions([1], [1], 1)

    def test_requires_each_word_once_in_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "omitted, reordered, or split"):
            collapse_subword_predictions([1, 2], [0, 2], 3)
        with self.assertRaisesRegex(ValueError, "omitted, reordered, or split"):
            collapse_subword_predictions([1, 2], [1, 0], 2)
        with self.assertRaisesRegex(ValueError, "omitted, reordered, or split"):
            collapse_subword_predictions([1, 2, 3], [0, 1, 0], 2)
        with self.assertRaisesRegex(ValueError, "omitted, reordered, or split"):
            collapse_subword_predictions([1, 2, 3], [0, None, 0], 1)

    def test_validates_number_of_words_and_flat_sequences(self) -> None:
        with self.assertRaisesRegex(TypeError, "number of words"):
            collapse_subword_predictions([1], [0], True)
        with self.assertRaisesRegex(ValueError, "positive"):
            collapse_subword_predictions([], [], 0)
        with self.assertRaisesRegex(TypeError, "flat sequence"):
            collapse_subword_predictions("B-V", [0, 0, 0], 1)


if __name__ == "__main__":
    unittest.main()
