import unittest

from semantic_action_extractor.srl.example import (
    PreparedWordLevelSRLExample,
    WordLevelSRLExample,
)


class WordLevelSRLExampleTests(unittest.TestCase):
    def test_assigns_split_only_after_preparation(self) -> None:
        prepared = PreparedWordLevelSRLExample(
            example_id="invented-doc:0:1",
            document_id="invented-doc",
            sentence_id="invented-doc:0",
            words=("Mira", "mailed"),
            predicate_index=1,
            tags=("B-ARG0", "B-V"),
            predicate_roleset="mail.01",
        )

        assigned = prepared.assign_split("development")

        self.assertIsInstance(assigned, WordLevelSRLExample)
        self.assertEqual(assigned.split, "development")
        self.assertEqual(assigned.words, prepared.words)

    def test_accepts_a_strict_source_neutral_example(self) -> None:
        example = WordLevelSRLExample(
            example_id="invented-doc:0:1",
            document_id="invented-doc",
            sentence_id="invented-doc:0",
            split="train",
            words=("Mira", "mailed", "a", "parcel"),
            predicate_index=1,
            tags=("B-ARG0", "B-V", "B-ARG1", "I-ARG1"),
            predicate_roleset="mail.01",
        )

        self.assertEqual(example.tags[example.predicate_index], "B-V")

    def test_rejects_invalid_split_or_identifiers(self) -> None:
        values = self._valid_values()
        values["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "split"):
            WordLevelSRLExample(**values)

        values = self._valid_values()
        values["document_id"] = " "
        with self.assertRaisesRegex(ValueError, "document_id"):
            WordLevelSRLExample(**values)

    def test_rejects_malformed_gold_bio(self) -> None:
        values = self._valid_values()
        values["tags"] = ("I-ARG0", "B-V")
        with self.assertRaisesRegex(ValueError, "word 0"):
            WordLevelSRLExample(**values)

    def test_requires_one_predicate_anchor_at_supplied_index(self) -> None:
        values = self._valid_values()
        values["tags"] = ("B-V", "B-V")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            WordLevelSRLExample(**values)

        values = self._valid_values()
        values["tags"] = ("B-V", "O")
        with self.assertRaisesRegex(ValueError, "supplied predicate"):
            WordLevelSRLExample(**values)

    def test_requires_tuple_words_and_matching_tags(self) -> None:
        values = self._valid_values()
        values["words"] = ["Mira", "mailed"]
        with self.assertRaisesRegex(TypeError, "tuple"):
            WordLevelSRLExample(**values)

        values = self._valid_values()
        values["tags"] = ("B-ARG0",)
        with self.assertRaisesRegex(ValueError, "same length"):
            WordLevelSRLExample(**values)

    @staticmethod
    def _valid_values():
        return {
            "example_id": "invented-doc:0:1",
            "document_id": "invented-doc",
            "sentence_id": "invented-doc:0",
            "split": "train",
            "words": ("Mira", "mailed"),
            "predicate_index": 1,
            "tags": ("B-ARG0", "B-V"),
            "predicate_roleset": "mail.01",
        }


if __name__ == "__main__":
    unittest.main()
