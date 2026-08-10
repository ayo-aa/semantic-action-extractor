import unittest

from semantic_action_extractor.schema import (
    ActionArgument,
    ActionFrame,
    ExtractionResult,
    MentionQualifier,
    TextSpan,
)


class SchemaTests(unittest.TestCase):
    def test_rejects_invalid_span(self) -> None:
        with self.assertRaises(ValueError):
            TextSpan(text="bad", start=4, end=3)

        with self.assertRaises(ValueError):
            TextSpan(text="bad", start=0, end=4)

        with self.assertRaises(ValueError):
            TextSpan(text="", start=0, end=0)

    def test_uses_unicode_code_point_offsets_for_astral_characters(self) -> None:
        text = "🤖 approved it."
        action = ActionFrame(
            predicate=TextSpan(text="approved", start=2, end=10),
            predicate_lemma="approve",
            predicate_type="verbal",
            sentence_index=0,
            score=0.7,
            score_type="model_probability",
            extractor="neural-test-double",
            arguments=(
                ActionArgument(
                    role="before_predicate",
                    span=TextSpan(text="🤖", start=0, end=1),
                ),
            ),
        )

        payload = ExtractionResult(text=text, actions=(action,)).to_dict()

        self.assertEqual(payload["actions"][0]["predicate"]["start"], 2)
        self.assertEqual(payload["actions"][0]["arguments"][0]["span"]["end"], 1)

    def test_rejects_invalid_score(self) -> None:
        with self.assertRaises(ValueError):
            ActionFrame(
                predicate=TextSpan(text="sent", start=0, end=4),
                predicate_lemma="send",
                predicate_type="verbal",
                sentence_index=0,
                score=1.1,
                score_type="heuristic_completeness",
                extractor="rule-based-test-double",
            )

    def test_rejects_invalid_argument_metadata(self) -> None:
        span = TextSpan(text="Maya", start=0, end=4)

        with self.assertRaises(ValueError):
            ActionArgument(role="", role_scheme="surface", span=span)

        with self.assertRaises(ValueError):
            ActionArgument(role="who?", role_scheme="unsupported", span=span)

        with self.assertRaisesRegex(ValueError, "provided together"):
            ActionArgument(
                role="who?",
                role_scheme="qa_srl",
                span=span,
                confidence=0.8,
            )

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            ActionArgument(
                role="who?",
                role_scheme="qa_srl",
                span=span,
                confidence=-0.1,
                confidence_type="model_probability",
            )

    def test_rejects_invalid_predicate_type(self) -> None:
        with self.assertRaises(ValueError):
            ActionFrame(
                predicate=TextSpan(text="sent", start=0, end=4),
                predicate_lemma="send",
                predicate_type="adjective",
                sentence_index=0,
                score=0.5,
                score_type="heuristic_completeness",
                extractor="rule-based-test-double",
            )

    def test_accepts_nominal_predicate_and_qasrl_role(self) -> None:
        action = ActionFrame(
            predicate=TextSpan(text="approval", start=8, end=16),
            predicate_lemma="approval",
            predicate_type="nominal",
            sentence_index=0,
            arguments=(
                ActionArgument(
                    role="who approved something?",
                    role_scheme="qa_srl",
                    span=TextSpan(text="Priya", start=0, end=5),
                ),
            ),
            score=0.8,
            score_type="model_probability",
            extractor="neural-test-double",
            related_verbal_form="approve",
            predicate_confidence=0.7123456789,
            predicate_confidence_type="model_probability",
        )

        self.assertEqual(action.predicate_type, "nominal")
        self.assertEqual(action.predicate_lemma, "approval")
        self.assertEqual(action.related_verbal_form, "approve")
        self.assertEqual(action.arguments[0].role_scheme, "qa_srl")
        self.assertEqual(action.to_dict()["predicate_confidence"], 0.7123456789)

    def test_preserves_explicit_multi_span_argument_group(self) -> None:
        action = ActionFrame(
            predicate=TextSpan(text="approved", start=17, end=25),
            predicate_lemma="approve",
            predicate_type="verbal",
            sentence_index=0,
            score=0.8,
            score_type="model_probability",
            extractor="neural-test-double",
            arguments=(
                ActionArgument(
                    role="who approved something?",
                    role_scheme="qa_srl",
                    span=TextSpan(text="Priya", start=0, end=5),
                    group_id="question-1",
                ),
                ActionArgument(
                    role="who approved something?",
                    role_scheme="qa_srl",
                    span=TextSpan(text="Jordan", start=10, end=16),
                    group_id="question-1",
                ),
            ),
        )

        payload = action.to_dict()

        self.assertEqual(
            [argument["group_id"] for argument in payload["arguments"]],
            ["question-1", "question-1"],
        )

    def test_preserves_orthogonal_source_grounded_mention_qualifiers(self) -> None:
        text = "Maya reportedly did not approve it."
        action = ActionFrame(
            predicate=TextSpan(text="approve", start=24, end=31),
            predicate_lemma="approve",
            predicate_type="verbal",
            sentence_index=0,
            score=0.8,
            score_type="model_probability",
            extractor="neural-test-double",
            mention_qualifiers=(
                MentionQualifier(
                    kind="reported",
                    evidence=(TextSpan(text="reportedly", start=5, end=15),),
                ),
                MentionQualifier(
                    kind="negated",
                    evidence=(TextSpan(text="not", start=20, end=23),),
                ),
            ),
        )

        payload = ExtractionResult(text=text, actions=(action,)).to_dict()

        self.assertEqual(
            [item["kind"] for item in payload["actions"][0]["mention_qualifiers"]],
            ["reported", "negated"],
        )
        self.assertEqual(
            payload["actions"][0]["mention_qualifiers"][1]["evidence"][0],
            {"text": "not", "start": 20, "end": 23},
        )

    def test_distinguishes_unassessed_from_assessed_without_qualifiers(self) -> None:
        common = {
            "predicate": TextSpan(text="approved", start=5, end=13),
            "predicate_lemma": "approve",
            "predicate_type": "verbal",
            "sentence_index": 0,
            "score": 0.8,
            "score_type": "model_probability",
            "extractor": "neural-test-double",
        }

        unassessed = ActionFrame(**common)
        assessed = ActionFrame(**common, mention_qualifiers=())

        self.assertIsNone(unassessed.to_dict()["mention_qualifiers"])
        self.assertEqual(assessed.to_dict()["mention_qualifiers"], [])

    def test_rejects_invalid_mention_qualifiers(self) -> None:
        cue = TextSpan(text="not", start=0, end=3)

        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            MentionQualifier(kind="completed", evidence=(cue,))

        with self.assertRaisesRegex(ValueError, "non-empty tuple"):
            MentionQualifier(kind="negated", evidence=())

        qualifier = MentionQualifier(kind="negated", evidence=(cue,))
        with self.assertRaisesRegex(ValueError, "kinds must be unique"):
            ActionFrame(
                predicate=TextSpan(text="approved", start=4, end=12),
                predicate_lemma="approve",
                predicate_type="verbal",
                sentence_index=0,
                score=0.8,
                score_type="model_probability",
                extractor="neural-test-double",
                mention_qualifiers=(qualifier, qualifier),
            )

    def test_result_rejects_qualifier_evidence_not_found_in_source(self) -> None:
        action = ActionFrame(
            predicate=TextSpan(text="approved", start=5, end=13),
            predicate_lemma="approve",
            predicate_type="verbal",
            sentence_index=0,
            score=0.8,
            score_type="model_probability",
            extractor="neural-test-double",
            mention_qualifiers=(
                MentionQualifier(
                    kind="negated",
                    evidence=(TextSpan(text="not", start=0, end=3),),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "map exactly"):
            ExtractionResult(text="Maya approved it.", actions=(action,))

    def test_grouped_arguments_must_share_one_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "share a role"):
            ActionFrame(
                predicate=TextSpan(text="approved", start=0, end=8),
                predicate_lemma="approve",
                predicate_type="verbal",
                sentence_index=0,
                score=0.8,
                score_type="model_probability",
                extractor="neural-test-double",
                arguments=(
                    ActionArgument(
                        role="who approved something?",
                        role_scheme="qa_srl",
                        span=TextSpan(text="Priya", start=10, end=15),
                        group_id="question-1",
                    ),
                    ActionArgument(
                        role="what was approved?",
                        role_scheme="qa_srl",
                        span=TextSpan(text="refund", start=20, end=26),
                        group_id="question-1",
                    ),
                ),
            )

    def test_rejects_invalid_argument_group_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "group_id"):
            ActionArgument(
                role="who approved something?",
                role_scheme="qa_srl",
                span=TextSpan(text="Priya", start=0, end=5),
                group_id=" ",
            )

    def test_requires_paired_predicate_confidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            ActionFrame(
                predicate=TextSpan(text="sent", start=0, end=4),
                predicate_lemma="send",
                predicate_type="verbal",
                sentence_index=0,
                score=0.7,
                score_type="heuristic_completeness",
                extractor="rule-based-test-double",
                predicate_confidence_type="model_probability",
            )

    def test_serializes_scores_without_rounding(self) -> None:
        raw_score = 0.123456789012345
        raw_argument_confidence = 0.987654321098765
        action = ActionFrame(
            predicate=TextSpan(text="sent", start=5, end=9),
            predicate_lemma="send",
            predicate_type="verbal",
            sentence_index=0,
            score=raw_score,
            score_type="model_probability",
            extractor="neural-test-double",
            arguments=(
                ActionArgument(
                    role="what was sent?",
                    role_scheme="qa_srl",
                    span=TextSpan(text="invoice", start=14, end=21),
                    confidence=raw_argument_confidence,
                    confidence_type="model_probability",
                ),
            ),
        )

        payload = action.to_dict()

        self.assertEqual(payload["score"], raw_score)
        self.assertEqual(
            payload["arguments"][0]["confidence"],
            raw_argument_confidence,
        )

    def test_result_rejects_span_that_does_not_match_source(self) -> None:
        action = ActionFrame(
            predicate=TextSpan(text="sent", start=5, end=9),
            predicate_lemma="send",
            predicate_type="verbal",
            sentence_index=0,
            score=0.5,
            score_type="heuristic_completeness",
            extractor="rule-based-test-double",
        )

        with self.assertRaisesRegex(ValueError, "map exactly"):
            ExtractionResult(text="Ava emailed the file.", actions=(action,))


if __name__ == "__main__":
    unittest.main()
