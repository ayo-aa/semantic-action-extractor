import unittest

from semantic_action_extractor.schema import (
    ActionFrame,
    ExtractionResult,
    PredicateCandidate,
    Qualifier,
    TextSpan,
)


class SchemaTests(unittest.TestCase):
    def test_rejects_invalid_span(self) -> None:
        with self.assertRaises(ValueError):
            TextSpan(text="bad", start=4, end=3)
        with self.assertRaises(ValueError):
            TextSpan(text="", start=0, end=0)
        with self.assertRaisesRegex(ValueError, "length"):
            TextSpan(text="bad", start=0, end=2)
        with self.assertRaisesRegex(TypeError, "integers"):
            TextSpan(text="bad", start=False, end=3)

    def test_rejects_invalid_confidence(self) -> None:
        with self.assertRaises(ValueError):
            ActionFrame(
                predicate=TextSpan(text="sent", start=0, end=4),
                predicate_lemma="send",
                sentence_index=0,
                confidence=1.1,
            )
        with self.assertRaisesRegex(TypeError, "real number"):
            ActionFrame(
                predicate=TextSpan(text="sent", start=0, end=4),
                predicate_lemma="send",
                sentence_index=0,
                confidence=True,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            ActionFrame(
                predicate=TextSpan(text="sent", start=0, end=4),
                predicate_lemma="send",
                sentence_index=0,
                confidence=float("nan"),
            )

    def test_rejects_candidate_index_outside_words(self) -> None:
        with self.assertRaises(ValueError):
            PredicateCandidate(
                words=(TextSpan(text="sent", start=0, end=4),),
                predicate_index=1,
                predicate_lemma="send",
                sentence_index=0,
            )
        with self.assertRaisesRegex(TypeError, "integer"):
            PredicateCandidate(
                words=(TextSpan(text="sent", start=0, end=4),),
                predicate_index=False,
                predicate_lemma="send",
                sentence_index=0,
            )

    def test_rejects_candidate_words_out_of_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordered"):
            PredicateCandidate(
                words=(
                    TextSpan(text="sent", start=5, end=9),
                    TextSpan(text="Ava", start=0, end=3),
                ),
                predicate_index=0,
                predicate_lemma="send",
                sentence_index=0,
            )

    def test_rejects_non_string_schema_labels(self) -> None:
        span = TextSpan(text="sent", start=0, end=4)
        with self.assertRaisesRegex(TypeError, "relation"):
            Qualifier(relation=True, value=span)
        with self.assertRaisesRegex(TypeError, "lemma"):
            ActionFrame(
                predicate=span,
                predicate_lemma=True,
                sentence_index=0,
            )

    def test_result_rejects_span_that_does_not_match_source(self) -> None:
        action = ActionFrame(
            predicate=TextSpan(text="sent", start=4, end=8),
            predicate_lemma="send",
            sentence_index=0,
        )
        with self.assertRaisesRegex(ValueError, "source text"):
            ExtractionResult(text="Ava mailed it.", actions=(action,))


if __name__ == "__main__":
    unittest.main()
