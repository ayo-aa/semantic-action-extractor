import unittest
from dataclasses import FrozenInstanceError

from semantic_action_extractor.srl.example import (
    PreparedWordLevelSRLExample,
    WordLevelSRLExample,
)
from semantic_action_extractor.srl.label_vocabulary import (
    SRLLabelVocabulary,
    build_training_label_vocabulary,
)


class SRLLabelVocabularyTests(unittest.TestCase):
    def test_builds_deterministic_training_vocabulary_with_continuations(self) -> None:
        first = self._example(
            "invented-a:0:1",
            ("Nia", "mapped", "routes", "again"),
            ("B-ARG0", "B-V", "B-C-V", "O"),
            1,
        )
        second = self._example(
            "invented-b:0:2",
            ("Yesterday", "Lee", "revised"),
            ("B-ARGM-TMP", "I-ARGM-TMP", "B-V"),
            2,
        )

        vocabulary = build_training_label_vocabulary([first, second])
        reversed_vocabulary = build_training_label_vocabulary([second, first])

        self.assertEqual(
            vocabulary.labels,
            (
                "O",
                "B-ARG0",
                "B-ARGM-TMP",
                "B-C-V",
                "B-V",
                "I-ARG0",
                "I-ARGM-TMP",
                "I-C-V",
                "I-V",
            ),
        )
        self.assertEqual(vocabulary.labels, reversed_vocabulary.labels)
        self.assertIn("I-C-V", vocabulary.labels)

    def test_exposes_immutable_inverse_mappings(self) -> None:
        vocabulary = self._vocabulary()

        self.assertEqual(vocabulary.encode("B-V"), vocabulary.label_to_id["B-V"])
        self.assertEqual(vocabulary.decode(vocabulary.encode("I-V")), "I-V")
        self.assertEqual(len(vocabulary), len(vocabulary.labels))
        self.assertIsInstance(hash(vocabulary), int)
        with self.assertRaises(TypeError):
            vocabulary.label_to_id["B-ARG9"] = 50
        with self.assertRaises(TypeError):
            vocabulary.id_to_label[50] = "B-ARG9"
        with self.assertRaises(FrozenInstanceError):
            vocabulary.labels = ("O",)

    def test_encode_rejects_invalid_and_unknown_labels(self) -> None:
        vocabulary = self._vocabulary()

        with self.assertRaisesRegex(ValueError, "invalid BIO tag"):
            vocabulary.encode("ARG0")
        with self.assertRaisesRegex(KeyError, "unknown BIO label"):
            vocabulary.encode("B-ARG9")

    def test_decode_rejects_noninteger_and_out_of_range_ids(self) -> None:
        vocabulary = self._vocabulary()

        with self.assertRaisesRegex(TypeError, "integer"):
            vocabulary.decode(True)
        with self.assertRaisesRegex(IndexError, "outside"):
            vocabulary.decode(-1)
        with self.assertRaisesRegex(IndexError, "outside"):
            vocabulary.decode(len(vocabulary))

    def test_requires_nonempty_training_records_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_training_label_vocabulary([])
        for split in ("development", "test"):
            with self.subTest(split=split):
                record = self._example(
                    f"invented-{split}:0:1",
                    ("Ira", "sorted"),
                    ("B-ARG0", "B-V"),
                    1,
                    split=split,
                )
                with self.assertRaisesRegex(ValueError, "training records only"):
                    build_training_label_vocabulary([record])

    def test_rejects_nonfinal_records_and_duplicate_example_ids(self) -> None:
        prepared = PreparedWordLevelSRLExample(
            example_id="invented-prepared:0:1",
            document_id="invented-prepared",
            sentence_id="invented-prepared:0",
            words=("Uma", "packed"),
            predicate_index=1,
            tags=("B-ARG0", "B-V"),
            predicate_roleset="pack.01",
        )
        with self.assertRaisesRegex(TypeError, "WordLevelSRLExample"):
            build_training_label_vocabulary([prepared])

        record = self._example(
            "invented-duplicate:0:1",
            ("Uma", "packed"),
            ("B-ARG0", "B-V"),
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate training example ID"):
            build_training_label_vocabulary([record, record])

    def test_constructor_rejects_invalid_duplicate_or_incomplete_labels(self) -> None:
        with self.assertRaisesRegex(TypeError, "tuple"):
            SRLLabelVocabulary(labels=["O"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            SRLLabelVocabulary(labels=("O", "B-V", "B-V", "I-V"))
        with self.assertRaisesRegex(ValueError, "O must be the first"):
            SRLLabelVocabulary(labels=("B-V", "I-V", "O"))
        with self.assertRaisesRegex(ValueError, "invalid BIO tag"):
            SRLLabelVocabulary(labels=("O", "ARG0"))
        with self.assertRaisesRegex(ValueError, "sorted order"):
            SRLLabelVocabulary(labels=("O", "I-V", "B-V"))
        with self.assertRaisesRegex(ValueError, "missing continuation"):
            SRLLabelVocabulary(labels=("O", "B-V"))

    @classmethod
    def _vocabulary(cls) -> SRLLabelVocabulary:
        return build_training_label_vocabulary(
            [
                cls._example(
                    "invented-vocabulary:0:1",
                    ("Ari", "cataloged"),
                    ("B-ARG0", "B-V"),
                    1,
                )
            ]
        )

    @staticmethod
    def _example(
        example_id,
        words,
        tags,
        predicate_index,
        *,
        split="train",
    ):
        document_id, _, _ = example_id.partition(":")
        return WordLevelSRLExample(
            example_id=example_id,
            document_id=document_id,
            sentence_id=f"{document_id}:0",
            split=split,
            words=words,
            predicate_index=predicate_index,
            tags=tags,
            predicate_roleset="invent.01",
        )


if __name__ == "__main__":
    unittest.main()
