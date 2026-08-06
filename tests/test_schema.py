import unittest

from semantic_action_extractor.schema import ActionFrame, TextSpan


class SchemaTests(unittest.TestCase):
    def test_rejects_invalid_span(self) -> None:
        with self.assertRaises(ValueError):
            TextSpan(text="bad", start=4, end=3)

    def test_rejects_invalid_confidence(self) -> None:
        with self.assertRaises(ValueError):
            ActionFrame(
                predicate=TextSpan(text="sent", start=0, end=4),
                predicate_lemma="send",
                sentence_index=0,
                confidence=1.1,
            )


if __name__ == "__main__":
    unittest.main()
