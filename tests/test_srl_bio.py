import unittest

from semantic_action_extractor.srl.bio import (
    LabeledSpan,
    continuation_tag,
    decode_bio,
    repair_bio,
)


class BioTests(unittest.TestCase):
    def test_repairs_orphaned_and_role_mismatched_i_tags(self) -> None:
        tags = ("I-ARG0", "I-ARG0", "I-ARG1", "O")

        self.assertEqual(
            repair_bio(tags), ("B-ARG0", "I-ARG0", "B-ARG1", "O")
        )

    def test_decodes_exact_end_exclusive_spans(self) -> None:
        tags = ("B-ARG0", "I-ARG0", "B-V", "B-ARGM-TMP", "I-ARGM-TMP")

        self.assertEqual(
            decode_bio(tags),
            (
                LabeledSpan("ARG0", 0, 2),
                LabeledSpan("V", 2, 3),
                LabeledSpan("ARGM-TMP", 3, 5),
            ),
        )

    def test_strict_decode_rejects_invalid_transition(self) -> None:
        with self.assertRaisesRegex(ValueError, "word 0"):
            decode_bio(("I-ARG0",), repair=False)

    def test_wordpiece_continuation_preserves_role(self) -> None:
        self.assertEqual(continuation_tag("B-ARGM-TMP"), "I-ARGM-TMP")
        self.assertEqual(continuation_tag("O"), "O")

    def test_rejects_malformed_tag(self) -> None:
        with self.assertRaises(ValueError):
            decode_bio(("ARG0",))
        with self.assertRaisesRegex(TypeError, "string"):
            decode_bio((True,))

    def test_rejects_non_integer_span_indexes(self) -> None:
        with self.assertRaisesRegex(TypeError, "integers"):
            LabeledSpan("ARG0", False, 1)


if __name__ == "__main__":
    unittest.main()
