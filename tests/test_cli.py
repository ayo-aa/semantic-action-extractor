from contextlib import redirect_stdout
from io import StringIO
import json
import unittest

from semantic_action_extractor.cli import main


class CliTests(unittest.TestCase):
    def test_cli_emits_json(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            exit_code = main(["Ava", "sent", "the", "invoice", "to", "Kai."])

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["actions"][0]["predicate_lemma"], "send")
        self.assertEqual(payload["actions"][0]["qualifiers"][0]["value"]["text"], "Kai")


if __name__ == "__main__":
    unittest.main()
