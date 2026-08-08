import tempfile
from pathlib import Path
import unittest

from semantic_action_extractor import BaselineConfig, RuleBasedExtractor


class BaselineTests(unittest.TestCase):
    def test_extracts_grounded_surface_arguments(self) -> None:
        text = "Maya emailed the signed contract to Jordan on Tuesday."

        result = RuleBasedExtractor().extract(text)

        self.assertEqual(result.schema_version, "0.2.0")
        self.assertEqual(len(result.actions), 1)
        action = result.actions[0]
        self.assertEqual(action.predicate.text, "emailed")
        self.assertEqual(action.predicate_lemma, "email")
        self.assertEqual(action.predicate_type, "verbal")
        self.assertEqual(action.score_type, "heuristic_completeness")
        self.assertEqual(
            [(item.role, item.span.text) for item in action.arguments],
            [
                ("before_predicate", "Maya"),
                ("after_predicate", "the signed contract"),
                ("to", "Jordan"),
                ("on", "Tuesday"),
            ],
        )

        self.assertEqual(text[action.predicate.start : action.predicate.end], "emailed")
        for argument in action.arguments:
            span = argument.span
            self.assertEqual(text[span.start : span.end], span.text)
            if argument.cue is not None:
                cue = argument.cue
                self.assertEqual(text[cue.start : cue.end], cue.text)

    def test_passive_voice_does_not_claim_actor_or_patient_roles(self) -> None:
        text = "The contract was emailed by Maya."

        action = RuleBasedExtractor().extract(text).actions[0]

        self.assertEqual(
            [(item.role, item.span.text) for item in action.arguments],
            [("before_predicate", "The contract"), ("by", "Maya")],
        )
        self.assertNotIn("actor", action.to_dict())
        self.assertNotIn("patient", action.to_dict())

    def test_preserves_unicode_source_span(self) -> None:
        text = "José emailed the signed form."

        action = RuleBasedExtractor().extract(text).actions[0]

        self.assertEqual(action.arguments[0].span.text, "José")
        self.assertEqual(
            text[action.arguments[0].span.start : action.arguments[0].span.end],
            "José",
        )

    def test_handles_irregular_verb_and_multiple_sentences(self) -> None:
        text = "Ava sent the invoice to Kai. Morgan approved the request."

        actions = RuleBasedExtractor().extract(text).actions

        self.assertEqual(
            [action.predicate_lemma for action in actions],
            ["send", "approve"],
        )
        self.assertEqual([action.sentence_index for action in actions], [0, 1])

    def test_returns_warning_when_no_predicate_matches(self) -> None:
        result = RuleBasedExtractor().extract("A quiet room with blue walls.")

        self.assertEqual(result.actions, ())
        self.assertIn("No action predicates", result.warnings[0])

    def test_warns_without_inventing_negation_status(self) -> None:
        result = RuleBasedExtractor().extract("Maya did not approve the refund.")

        self.assertIn("does not encode action polarity", result.warnings[0])

    def test_loads_domain_verb_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.toml"
            path.write_text(
                '[baseline]\nadditional_verbs = ["  TrIaGe  "]\nmin_score = 0.0\n',
                encoding="utf-8",
            )

            config = BaselineConfig.from_toml(path)
            result = RuleBasedExtractor(config).extract(
                "The operator triaged the alert."
            )

        self.assertEqual(len(result.actions), 1)
        self.assertEqual(config.additional_verbs, ("triage",))
        self.assertEqual(result.actions[0].predicate_lemma, "triage")

    def test_normalises_directly_configured_domain_verb(self) -> None:
        config = BaselineConfig(additional_verbs=("  TrIaGe  ",))

        result = RuleBasedExtractor(config).extract(
            "The operator triaged the alert."
        )

        self.assertEqual(config.additional_verbs, ("triage",))
        self.assertEqual(result.actions[0].predicate_lemma, "triage")

    def test_does_not_inherit_left_context_across_hard_clause_boundary(self) -> None:
        cases = (
            "Maya emailed Jordan; call Ava.",
            "Maya emailed Jordan: call Ava.",
            "Maya emailed Jordan. Call Ava.",
        )

        for text in cases:
            with self.subTest(text=text):
                actions = RuleBasedExtractor().extract(text).actions
                call = next(
                    action for action in actions if action.predicate_lemma == "call"
                )

                self.assertEqual(
                    [(item.role, item.span.text) for item in call.arguments],
                    [("after_predicate", "Ava")],
                )

    def test_inherits_left_context_for_coordinated_predicate(self) -> None:
        text = "Maya emailed Jordan and called Ava."

        actions = RuleBasedExtractor().extract(text).actions
        call = next(action for action in actions if action.predicate_lemma == "call")

        self.assertEqual(
            [(item.role, item.span.text) for item in call.arguments],
            [
                ("before_predicate", "Maya"),
                ("after_predicate", "Ava"),
            ],
        )

    def test_min_score_filters_incomplete_frames(self) -> None:
        extractor = RuleBasedExtractor(BaselineConfig(min_score=0.8))

        incomplete = extractor.extract("Call.")
        complete = extractor.extract("Maya emailed the contract to Jordan.")

        self.assertEqual(incomplete.actions, ())
        self.assertEqual(len(complete.actions), 1)
        self.assertEqual(complete.actions[0].score, 0.85)

    def test_rejects_boolean_min_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            BaselineConfig(min_score=True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.toml"
            path.write_text("[baseline]\nmin_score = true\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be a number"):
                BaselineConfig.from_toml(path)


if __name__ == "__main__":
    unittest.main()
