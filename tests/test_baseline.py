import tempfile
from pathlib import Path
import unittest

from semantic_action_extractor import BaselineConfig, RuleBasedExtractor


class BaselineTests(unittest.TestCase):
    def test_extracts_grounded_action_and_qualifiers(self) -> None:
        text = "Maya emailed the signed contract to Jordan on Tuesday."

        result = RuleBasedExtractor().extract(text)

        self.assertEqual(len(result.actions), 1)
        action = result.actions[0]
        self.assertEqual(action.actor.text, "Maya")
        self.assertEqual(action.predicate.text, "emailed")
        self.assertEqual(action.predicate_lemma, "email")
        self.assertEqual(action.patient.text, "the signed contract")
        self.assertEqual(
            [(item.relation, item.value.text) for item in action.qualifiers],
            [("to", "Jordan"), ("on", "Tuesday")],
        )

        for span in (action.actor, action.predicate, action.patient):
            self.assertEqual(text[span.start : span.end], span.text)
        for qualifier in action.qualifiers:
            span = qualifier.value
            self.assertEqual(text[span.start : span.end], span.text)

    def test_handles_irregular_verb_and_multiple_sentences(self) -> None:
        text = "Ava sent the invoice to Kai. Morgan approved the request."

        actions = RuleBasedExtractor().extract(text).actions

        self.assertEqual([action.predicate_lemma for action in actions], ["send", "approve"])
        self.assertEqual([action.sentence_index for action in actions], [0, 1])

    def test_exposes_sentence_local_predicate_candidates_for_srl(self) -> None:
        text = "Ava sent the invoice. Morgan approved it."

        candidates = RuleBasedExtractor().detect_predicates(text)

        self.assertEqual(
            [candidate.predicate.text for candidate in candidates], ["sent", "approved"]
        )
        self.assertEqual(
            [candidate.predicate_lemma for candidate in candidates], ["send", "approve"]
        )
        self.assertEqual([candidate.sentence_index for candidate in candidates], [0, 1])
        self.assertEqual(
            [candidate.words[candidate.predicate_index].text for candidate in candidates],
            ["sent", "approved"],
        )
        for candidate in candidates:
            for word in candidate.words:
                self.assertEqual(text[word.start : word.end], word.text)

    def test_returns_warning_when_no_predicate_matches(self) -> None:
        result = RuleBasedExtractor().extract("A quiet room with blue walls.")

        self.assertEqual(result.actions, ())
        self.assertIn("No action predicates", result.warnings[0])

    def test_loads_domain_verb_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.toml"
            path.write_text(
                '[baseline]\nadditional_verbs = ["triage"]\nmin_confidence = 0.0\n',
                encoding="utf-8",
            )

            config = BaselineConfig.from_toml(path)
            result = RuleBasedExtractor(config).extract("The operator triaged the alert.")

        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].predicate_lemma, "triage")

    def test_normalizes_direct_additional_verbs(self) -> None:
        config = BaselineConfig(additional_verbs=(" Triage ", "triage"))

        self.assertEqual(config.additional_verbs, ("triage",))
        result = RuleBasedExtractor(config).extract("The operator triaged the alert.")
        self.assertEqual(result.actions[0].predicate_lemma, "triage")

    def test_rejects_boolean_confidence_threshold(self) -> None:
        with self.assertRaisesRegex(TypeError, "real number"):
            BaselineConfig(min_confidence=True)


if __name__ == "__main__":
    unittest.main()
