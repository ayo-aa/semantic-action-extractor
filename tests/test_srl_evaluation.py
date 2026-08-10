import unittest

from semantic_action_extractor.srl.evaluation import (
    evaluate_supplied_predicate_srl,
    score_labeled_spans,
)


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


class SuppliedPredicateEvaluationTests(unittest.TestCase):
    def test_matches_hand_calculated_micro_and_per_role_oracle(self) -> None:
        result = evaluate_supplied_predicate_srl(
            [
                ("B-ARG0", "B-V", "B-ARG1", "I-ARG1"),
                ("O", "B-V", "B-ARGM-TMP"),
            ],
            [
                ("B-ARG0", "B-V", "B-ARG1", "I-ARG1"),
                ("O", "B-V", "B-ARGM-LOC"),
            ],
            [1, 1],
        )

        self.assertEqual(result.arguments.true_positives, 2)
        self.assertEqual(result.arguments.predicted, 3)
        self.assertEqual(result.arguments.gold, 3)
        self.assertAlmostEqual(result.arguments.precision, 2 / 3)
        self.assertAlmostEqual(result.arguments.recall, 2 / 3)
        self.assertAlmostEqual(result.arguments.f1, 2 / 3)
        self.assertEqual(
            [metrics.label for metrics in result.per_role],
            ["ARG0", "ARG1", "ARGM-LOC", "ARGM-TMP"],
        )
        by_role = {metrics.label: metrics for metrics in result.per_role}
        self.assertEqual(by_role["ARG0"].true_positives, 1)
        self.assertEqual(by_role["ARG0"].f1, 1.0)
        self.assertEqual(by_role["ARG1"].true_positives, 1)
        self.assertEqual(by_role["ARG1"].f1, 1.0)
        self.assertEqual(by_role["ARGM-LOC"].predicted, 1)
        self.assertEqual(by_role["ARGM-LOC"].gold, 0)
        self.assertEqual(by_role["ARGM-LOC"].precision, 0.0)
        self.assertEqual(by_role["ARGM-LOC"].recall, 0.0)
        self.assertEqual(by_role["ARGM-LOC"].f1, 0.0)
        self.assertEqual(by_role["ARGM-TMP"].predicted, 0)
        self.assertEqual(by_role["ARGM-TMP"].gold, 1)
        self.assertEqual(by_role["ARGM-TMP"].precision, 0.0)
        self.assertEqual(by_role["ARGM-TMP"].recall, 0.0)
        self.assertEqual(by_role["ARGM-TMP"].f1, 0.0)
        self.assertEqual(
            sum(metrics.true_positives for metrics in result.per_role),
            result.arguments.true_positives,
        )
        self.assertEqual(
            sum(metrics.predicted for metrics in result.per_role),
            result.arguments.predicted,
        )
        self.assertEqual(
            sum(metrics.gold for metrics in result.per_role),
            result.arguments.gold,
        )
        self.assertEqual(result.token_accuracy.correct, 6)
        self.assertEqual(result.token_accuracy.total, 7)
        self.assertAlmostEqual(result.token_accuracy.accuracy, 6 / 7)
        self.assertEqual(result.predicate.correct_anchors, 2)
        self.assertEqual(result.predicate.missing_anchors, 0)

    def test_argument_at_supplied_predicate_is_a_false_positive(self) -> None:
        result = evaluate_supplied_predicate_srl(
            [("B-ARG0", "B-V", "O")],
            [("B-ARG0", "B-ARG1", "O")],
            [1],
        )

        self.assertEqual(result.arguments.true_positives, 1)
        self.assertEqual(result.arguments.predicted, 2)
        self.assertEqual(result.arguments.gold, 1)
        self.assertAlmostEqual(result.arguments.precision, 0.5)
        self.assertEqual(result.arguments.recall, 1.0)
        self.assertEqual(result.predicate.correct_anchors, 0)
        self.assertEqual(result.predicate.missing_anchors, 1)
        self.assertEqual(
            result.predicate.argument_spans_overlapping_predicate, 1
        )

    def test_spurious_v_replacing_gold_argument_remains_a_false_negative(self) -> None:
        result = evaluate_supplied_predicate_srl(
            [("B-ARG0", "B-V", "B-ARG1")],
            [("B-ARG0", "B-V", "B-V")],
            [1],
        )

        self.assertEqual(result.arguments.true_positives, 1)
        self.assertEqual(result.arguments.predicted, 1)
        self.assertEqual(result.arguments.gold, 2)
        self.assertEqual(result.arguments.precision, 1.0)
        self.assertAlmostEqual(result.arguments.recall, 0.5)
        self.assertEqual(result.predicate.correct_anchors, 1)
        self.assertEqual(result.predicate.spurious_predicate_words, 1)

    def test_excludes_all_gold_multiword_predicate_pieces(self) -> None:
        result = evaluate_supplied_predicate_srl(
            [("B-ARG0", "B-V", "B-C-V", "B-ARG1")],
            [("B-ARG0", "B-V", "B-C-V", "B-ARG1")],
            [1],
        )

        self.assertEqual(result.arguments.true_positives, 2)
        self.assertEqual(result.arguments.predicted, 2)
        self.assertEqual(result.arguments.gold, 2)
        self.assertEqual(result.arguments.f1, 1.0)
        self.assertEqual(result.predicate.spurious_predicate_words, 0)
        self.assertEqual(result.token_accuracy.accuracy, 1.0)

    def test_repairs_predictions_before_all_metrics_and_diagnostics(self) -> None:
        result = evaluate_supplied_predicate_srl(
            [("B-ARG0", "B-V")],
            [("I-ARG0", "I-V")],
            [1],
        )

        self.assertEqual(result.arguments.f1, 1.0)
        self.assertEqual(result.arguments.repaired_prediction_tags, 2)
        self.assertEqual(result.predicate.correct_anchors, 1)
        self.assertEqual(result.token_accuracy.accuracy, 1.0)

    def test_validates_predicate_indexes_and_strict_gold_anchors(self) -> None:
        with self.assertRaisesRegex(ValueError, "predicate index and gold"):
            evaluate_supplied_predicate_srl(
                [("B-V",)], [("B-V",)], []
            )
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            evaluate_supplied_predicate_srl(
                [("B-V",)], [("B-V",)], [False]
            )
        with self.assertRaisesRegex(IndexError, "outside example 0"):
            evaluate_supplied_predicate_srl(
                [("B-V",)], [("B-V",)], [1]
            )
        with self.assertRaisesRegex(ValueError, "must have B-V"):
            evaluate_supplied_predicate_srl(
                [("O", "B-V")], [("O", "B-V")], [0]
            )
        with self.assertRaisesRegex(ValueError, "exactly one B-V"):
            evaluate_supplied_predicate_srl(
                [("B-V", "B-V")], [("B-V", "B-V")], [0]
            )
        with self.assertRaisesRegex(ValueError, "word 0"):
            evaluate_supplied_predicate_srl(
                [("I-ARG0", "B-V")], [("B-ARG0", "B-V")], [1]
            )

    def test_can_reject_invalid_prediction_bio_without_repair(self) -> None:
        with self.assertRaisesRegex(ValueError, "word 0"):
            evaluate_supplied_predicate_srl(
                [("B-ARG0", "B-V")],
                [("I-ARG0", "B-V")],
                [1],
                repair_predictions=False,
            )

    def test_empty_batch_has_explicit_zero_metrics(self) -> None:
        result = evaluate_supplied_predicate_srl([], [], [])

        self.assertEqual(result.arguments.f1, 0.0)
        self.assertEqual(result.per_role, ())
        self.assertEqual(result.predicate.examples, 0)
        self.assertEqual(result.token_accuracy.total, 0)
        self.assertEqual(result.token_accuracy.accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
