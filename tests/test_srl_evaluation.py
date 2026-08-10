import unittest

from semantic_action_extractor.srl.evaluation import score_labeled_spans


class EvaluationTests(unittest.TestCase):
    def test_scores_exact_labeled_spans_and_excludes_predicate(self) -> None:
        gold = [
            ("B-ARG0", "B-V", "B-ARG1", "I-ARG1"),
            ("B-ARG0", "B-V", "O"),
        ]
        predicted = [
            ("B-ARG0", "B-V", "B-ARG1", "I-ARG1"),
            ("B-ARG1", "O", "B-V"),
        ]

        metrics = score_labeled_spans(gold, predicted)

        self.assertEqual(metrics.true_positives, 2)
        self.assertEqual(metrics.predicted, 3)
        self.assertEqual(metrics.gold, 3)
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 2 / 3)
        self.assertAlmostEqual(metrics.f1, 2 / 3)

    def test_requires_boundaries_and_label_to_match(self) -> None:
        metrics = score_labeled_spans(
            [("B-ARG1", "I-ARG1")],
            [("B-ARG1", "B-ARG1")],
        )

        self.assertEqual(metrics.true_positives, 0)
        self.assertEqual(metrics.predicted, 2)
        self.assertEqual(metrics.gold, 1)

    def test_repairs_invalid_predictions_before_scoring(self) -> None:
        metrics = score_labeled_spans(
            [("B-ARG0", "I-ARG0")],
            [("I-ARG0", "I-ARG0")],
        )

        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.repaired_prediction_tags, 1)

    def test_rejects_invalid_gold_bio(self) -> None:
        with self.assertRaisesRegex(ValueError, "word 0"):
            score_labeled_spans([("I-ARG0",)], [("B-ARG0",)])

    def test_can_reject_invalid_prediction_bio(self) -> None:
        with self.assertRaisesRegex(ValueError, "word 0"):
            score_labeled_spans(
                [("B-ARG0",)],
                [("I-ARG0",)],
                repair_predictions=False,
            )

    def test_excludes_continuation_predicate_spans(self) -> None:
        metrics = score_labeled_spans(
            [("B-C-V", "B-ARG0")],
            [("B-C-V", "B-ARG0")],
        )

        self.assertEqual(metrics.gold, 1)
        self.assertEqual(metrics.predicted, 1)

    def test_empty_examples_have_zero_metrics(self) -> None:
        metrics = score_labeled_spans([("O",)], [("O",)])

        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1, 0.0)

    def test_rejects_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "example 0"):
            score_labeled_spans([("O",)], [("O", "O")])


if __name__ == "__main__":
    unittest.main()
