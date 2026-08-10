import json
import unittest

from semantic_action_extractor.srl.batching import (
    AlignedModelExample,
    collate_srl_batch,
    prepare_srl_split,
)
from semantic_action_extractor.srl.example import WordLevelSRLExample
from semantic_action_extractor.srl.label_vocabulary import (
    build_training_label_vocabulary,
)


class FakeEncoding(dict):
    def __init__(self, word_ids, **values):
        super().__init__(values)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class InventedPieceTokenizer:
    def __init__(self, piece_counts=None):
        self.piece_counts = piece_counts or {}
        self.calls = []

    def __call__(self, words, **kwargs):
        self.calls.append((tuple(words), dict(kwargs)))
        input_ids = [101]
        word_ids = [None]
        next_id = 10
        for word_index, word in enumerate(words):
            for _ in range(self.piece_counts.get(word, 1)):
                input_ids.append(next_id)
                word_ids.append(word_index)
                next_id += 1
        input_ids.append(102)
        word_ids.append(None)
        return FakeEncoding(
            word_ids,
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
        )


class SRLBatchingTests(unittest.TestCase):
    def test_prepares_only_selected_split_in_supplied_order(self) -> None:
        train_a = self._example(
            "invented-train-a:0:1", "train", ("Mina", "filed"), ("B-ARG0", "B-V")
        )
        train_b = self._example(
            "invented-train-b:0:1", "train", ("Oren", "filed"), ("B-ARG0", "B-V")
        )
        development_a = self._example(
            "invented-development-a:0:1",
            "development",
            ("Pia", "filed"),
            ("B-ARG0", "B-V"),
        )
        development_b = self._example(
            "invented-development-b:0:1",
            "development",
            ("Quin", "refiled"),
            ("B-ARG0", "B-V"),
        )
        vocabulary = build_training_label_vocabulary([train_a, train_b])
        tokenizer = InventedPieceTokenizer({"refiled": 2})

        prepared = prepare_srl_split(
            tokenizer,
            [development_b, train_a, development_a, train_b],
            vocabulary,
            "development",
            max_length=8,
        )

        self.assertEqual(prepared.total_records, 4)
        self.assertEqual(prepared.selected_records, 2)
        self.assertEqual(prepared.skipped_other_splits, 2)
        self.assertEqual(prepared.retained_count, 2)
        self.assertEqual(prepared.overlength_drop_count, 0)
        self.assertEqual(
            tuple(example.example_id for example in prepared.examples),
            (development_b.example_id, development_a.example_id),
        )
        self.assertEqual(
            tuple(example.word_count for example in prepared.examples), (2, 2)
        )
        self.assertEqual(prepared.examples[0].alignment.word_ids, (None, 0, 1, 1, None))
        self.assertEqual(
            [call[0] for call in tokenizer.calls],
            [development_b.words, development_a.words],
        )
        for _, kwargs in tokenizer.calls:
            self.assertIs(kwargs["padding"], False)
            self.assertIs(kwargs["truncation"], False)
            self.assertNotIn("max_length", kwargs)

    def test_fails_before_tokenization_on_unseen_evaluation_label(self) -> None:
        training = self._example(
            "invented-training:0:1", "train", ("Rae", "logged"), ("B-ARG0", "B-V")
        )
        evaluation = self._example(
            "invented-evaluation:0:1",
            "test",
            ("Rae", "logged", "results"),
            ("B-ARG0", "B-V", "B-ARG1"),
        )
        vocabulary = build_training_label_vocabulary([training])
        tokenizer = InventedPieceTokenizer()

        with self.assertRaisesRegex(KeyError, "absent from the training vocabulary"):
            prepare_srl_split(
                tokenizer,
                [evaluation],
                vocabulary,
                "test",
                max_length=8,
            )

        self.assertEqual(tokenizer.calls, [])

    def test_reports_measured_overlength_drops_without_truncating(self) -> None:
        short = self._example(
            "invented-short:0:1", "train", ("Sol", "coded"), ("B-ARG0", "B-V")
        )
        long = self._example(
            "invented-long:0:1",
            "train",
            ("Tao", "multisegmented"),
            ("B-ARG0", "B-V"),
        )
        vocabulary = build_training_label_vocabulary([short, long])
        tokenizer = InventedPieceTokenizer({"multisegmented": 5})

        prepared = prepare_srl_split(
            tokenizer,
            [short, long],
            vocabulary,
            "train",
            max_length=5,
        )

        self.assertEqual(prepared.selected_records, 2)
        self.assertEqual(prepared.retained_count, 1)
        self.assertEqual(prepared.overlength_drop_count, 1)
        drop = prepared.overlength_drops[0]
        self.assertEqual(drop.example_id, long.example_id)
        self.assertEqual(drop.tokenized_length, 8)
        self.assertEqual(drop.max_length, 5)
        self.assertEqual(drop.reason, "tokenized_length_exceeds_max_length")
        self.assertTrue(all(call[1]["truncation"] is False for call in tokenizer.calls))

    def test_collates_padding_and_predicate_ablation_as_a_paired_input(self) -> None:
        first = self._example(
            "invented-first:0:1", "train", ("Uma", "retagged"), ("B-ARG0", "B-V")
        )
        second = self._example(
            "invented-second:0:1",
            "train",
            ("Vic", "tagged", "items"),
            ("B-ARG0", "B-V", "B-ARG1"),
        )
        vocabulary = build_training_label_vocabulary([first, second])
        prepared = prepare_srl_split(
            InventedPieceTokenizer({"retagged": 3}),
            [first, second],
            vocabulary,
            "train",
            max_length=8,
        )

        conditioned = collate_srl_batch(
            prepared.examples, pad_token_id=0, predicate_signal=True
        )
        unconditioned = collate_srl_batch(
            prepared.examples, pad_token_id=0, predicate_signal=False
        )

        self.assertEqual(conditioned.example_ids, (first.example_id, second.example_id))
        self.assertEqual(conditioned.word_counts, (2, 3))
        self.assertEqual(conditioned.example_ids, unconditioned.example_ids)
        self.assertEqual(conditioned.word_counts, unconditioned.word_counts)
        self.assertEqual(conditioned.word_ids, unconditioned.word_ids)
        self.assertEqual(conditioned.input_ids[1][-1], 0)
        self.assertEqual(conditioned.attention_mask[1][-1], 0)
        self.assertEqual(conditioned.labels[1][-1], -100)
        self.assertIsNone(conditioned.word_ids[1][-1])
        self.assertEqual(sum(conditioned.token_type_ids[0]), 1)
        self.assertEqual(sum(conditioned.token_type_ids[1]), 1)
        self.assertTrue(
            all(
                value == 0
                for row in unconditioned.token_type_ids
                for value in row
            )
        )

        conditioned_inputs = conditioned.to_model_inputs()
        unconditioned_inputs = unconditioned.to_model_inputs()
        conditioned_without_signal = {
            key: value
            for key, value in conditioned_inputs.items()
            if key != "token_type_ids"
        }
        unconditioned_without_signal = {
            key: value
            for key, value in unconditioned_inputs.items()
            if key != "token_type_ids"
        }
        self.assertEqual(
            json.dumps(conditioned_without_signal, sort_keys=True).encode(),
            json.dumps(unconditioned_without_signal, sort_keys=True).encode(),
        )

    def test_validates_batch_and_preparation_configuration(self) -> None:
        record = self._example(
            "invented-validate:0:1", "train", ("Wes", "indexed"), ("B-ARG0", "B-V")
        )
        vocabulary = build_training_label_vocabulary([record])
        tokenizer = InventedPieceTokenizer()

        with self.assertRaisesRegex(ValueError, "split must be"):
            prepare_srl_split(
                tokenizer, [record], vocabulary, "validation", max_length=8
            )
        with self.assertRaisesRegex(TypeError, "max_length"):
            prepare_srl_split(tokenizer, [record], vocabulary, "train", max_length=True)
        with self.assertRaisesRegex(ValueError, "empty batch"):
            collate_srl_batch([], pad_token_id=0, predicate_signal=True)
        with self.assertRaisesRegex(TypeError, "predicate signal"):
            collate_srl_batch([], pad_token_id=0, predicate_signal=1)

    def test_rejects_duplicate_batch_examples(self) -> None:
        record = self._example(
            "invented-duplicate:0:1", "train", ("Xia", "sorted"), ("B-ARG0", "B-V")
        )
        vocabulary = build_training_label_vocabulary([record])
        prepared = prepare_srl_split(
            InventedPieceTokenizer(),
            [record],
            vocabulary,
            "train",
            max_length=8,
        )
        example: AlignedModelExample = prepared.examples[0]

        with self.assertRaisesRegex(ValueError, "duplicate example IDs"):
            collate_srl_batch(
                [example, example], pad_token_id=0, predicate_signal=True
            )

    @staticmethod
    def _example(example_id, split, words, tags):
        document_id, _, _ = example_id.partition(":")
        return WordLevelSRLExample(
            example_id=example_id,
            document_id=document_id,
            sentence_id=f"{document_id}:0",
            split=split,
            words=words,
            predicate_index=tags.index("B-V"),
            tags=tags,
            predicate_roleset="invent.01",
        )


if __name__ == "__main__":
    unittest.main()
