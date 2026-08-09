import gzip
import json
from pathlib import Path
import tempfile
import unittest

from semantic_action_extractor.datasets.common import DatasetFormatError
from semantic_action_extractor.datasets.qasrl import (
    adapt_qasrl_record,
    iter_qasrl_records,
    load_qasrl_index,
)


def _question(
    question_string="Who sent something?",
    *,
    spans=None,
    valid=True,
    source_id="validator-1",
):
    judgment = {"sourceId": source_id, "isValid": valid}
    if spans is not None:
        judgment["spans"] = spans
    return {
        "questionString": question_string,
        "questionSources": ["writer-1"],
        "answerJudgments": [judgment],
        "questionSlots": {
            "wh": "who",
            "aux": "_",
            "subj": "_",
            "verb": "past",
            "obj": "something",
            "prep": "_",
            "obj2": "_",
        },
        "tense": "past",
        "isPerfect": False,
        "isProgressive": False,
        "isNegated": False,
        "isPassive": False,
    }


def _record():
    question_string = "Who sent something?"
    return {
        "sentenceId": "Wiki1k:wikipedia:42:3:0",
        "sentenceTokens": ["Ari", "sent", "blue", "files", "."],
        "verbEntries": {
            "1": {
                "verbIndex": 1,
                "verbInflectedForms": {
                    "stem": "send",
                    "presentSingular3rd": "sends",
                    "presentParticiple": "sending",
                    "past": "sent",
                    "pastParticiple": "sent",
                },
                "questionLabels": {
                    question_string: _question(spans=[[0, 1], [2, 4]])
                },
            }
        },
        "nonPredicates": {},
    }


def _index_payload():
    return {
        "documents": {
            "dev": [
                {
                    "part": "dev",
                    "idString": "Wiki1k:wikipedia:42",
                    "domain": "wikipedia",
                    "id": "42",
                    "title": "Synthetic document",
                }
            ]
        },
        "denseIds": ["Wiki1k:wikipedia:42:3:0"],
    }


class QASRLAdapterTests(unittest.TestCase):
    def test_preserves_questions_judgments_and_separate_answer_alternatives(self):
        record = adapt_qasrl_record(
            _record(),
            release="2.1",
            split="dev",
            layer="dense",
        )

        self.assertEqual(record.text, "Ari sent blue files .")
        self.assertEqual(record.provenance.dataset, "qa-srl")
        self.assertEqual(record.provenance.document_id, "Wiki1k:wikipedia:42")
        self.assertEqual(record.provenance.metadata["domain"], "wikipedia")
        self.assertEqual(record.metadata["non_predicates"], {})

        candidate = record.candidates[0]
        self.assertEqual(
            candidate.candidate_id,
            "qa-srl:Wiki1k:wikipedia:42:3:0:verb:1",
        )
        self.assertEqual(candidate.span.span.text, "sent")
        self.assertEqual(candidate.lemma, "send")
        self.assertEqual(candidate.verb_inflected_forms.past_participle, "sent")

        question = candidate.questions[0]
        self.assertEqual(question.surface_form, "Who sent something?")
        self.assertEqual(question.slots.verb, "past")
        self.assertEqual(question.question_sources, ("writer-1",))
        self.assertEqual(len(question.judgments), 1)
        answers = question.judgments[0].answers
        self.assertEqual(len(answers), 2)
        self.assertEqual(
            [answer.surface_form for answer in answers],
            ["Ari", "blue files"],
        )
        self.assertEqual([len(answer.spans) for answer in answers], [1, 1])
        self.assertEqual(
            [
                (answer.spans[0].token_start, answer.spans[0].token_end)
                for answer in answers
            ],
            [(0, 1), (2, 4)],
        )

    def test_preserves_internal_nonbreaking_space_in_upstream_token(self):
        raw = _record()
        raw["sentenceTokens"][2] = "blue\N{NO-BREAK SPACE}files"
        question = next(iter(raw["verbEntries"]["1"]["questionLabels"].values()))
        question["answerJudgments"][0]["spans"] = [[2, 3]]

        record = adapt_qasrl_record(raw, release="2.1", split="dev")

        answer = record.candidates[0].questions[0].judgments[0].answers[0]
        self.assertEqual(answer.surface_form, "blue\N{NO-BREAK SPACE}files")
        self.assertEqual(record.tokens[2].span.text, "blue\N{NO-BREAK SPACE}files")

    def test_ids_are_stable_and_do_not_depend_on_mapping_order(self):
        first = _record()
        first_question = next(
            iter(first["verbEntries"]["1"]["questionLabels"].values())
        )
        second_question = _question(
            "What did someone send?",
            spans=[[2, 4]],
            source_id="validator-2",
        )
        second_question["questionSlots"].update(
            {"wh": "what", "aux": "did", "subj": "someone", "verb": "stem", "obj": "_"}
        )
        first["verbEntries"]["1"]["questionLabels"] = {
            "What did someone send?": second_question,
            "Who sent something?": first_question,
        }
        reordered = _record()
        reordered["verbEntries"]["1"]["questionLabels"] = {
            "Who sent something?": first_question,
            "What did someone send?": second_question,
        }

        one = adapt_qasrl_record(first, release="2.1", split="dev")
        two = adapt_qasrl_record(reordered, release="2.1", split="dev")

        self.assertEqual(
            [question.question_id for question in one.candidates[0].questions],
            [question.question_id for question in two.candidates[0].questions],
        )
        self.assertEqual(one.to_dict(), two.to_dict())

    def test_preserves_invalid_judgment_without_inventing_answers(self):
        raw = _record()
        question = next(iter(raw["verbEntries"]["1"]["questionLabels"].values()))
        question["answerJudgments"].append(
            {"sourceId": "validator-2", "isValid": False}
        )

        record = adapt_qasrl_record(raw, release="2.1", split="dev")

        invalid = record.candidates[0].questions[0].judgments[1]
        self.assertFalse(invalid.is_valid)
        self.assertEqual(invalid.answers, ())

    def test_preserves_upstream_valid_judgment_with_empty_spans_as_anomaly(self):
        raw = _record()
        question = next(iter(raw["verbEntries"]["1"]["questionLabels"].values()))
        question["answerJudgments"][0]["spans"] = []

        record = adapt_qasrl_record(raw, release="2.1", split="dev")

        judgment = record.candidates[0].questions[0].judgments[0]
        self.assertTrue(judgment.is_valid)
        self.assertEqual(judgment.answers, ())
        self.assertTrue(judgment.metadata["upstream_empty_valid_spans"])
        self.assertEqual(
            record.metadata["upstream_anomaly_counts"]["empty_valid_spans"],
            1,
        )

    def test_loads_index_and_validates_document_domain_split_and_dense_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index.json.gz"
            with gzip.open(index_path, "wt", encoding="utf-8") as handle:
                json.dump(_index_payload(), handle)

            index = load_qasrl_index(index_path)
            record = adapt_qasrl_record(
                _record(),
                release="2.1",
                split="dev",
                layer="dense",
                document_index=index,
            )

        self.assertEqual(
            record.provenance.metadata["document_title"],
            "Synthetic document",
        )
        self.assertEqual(record.provenance.metadata["document_source_id"], "42")

        with self.assertRaisesRegex(DatasetFormatError, "belongs to 'dev'"):
            adapt_qasrl_record(
                _record(),
                release="2.1",
                split="test",
                document_index=index,
            )

        raw = _record()
        raw["sentenceId"] = "Wiki1k:wikinews:42:3:0"
        with self.assertRaisesRegex(DatasetFormatError, "absent from the QA-SRL index"):
            adapt_qasrl_record(
                raw,
                release="2.1",
                split="dev",
                document_index=index,
            )

    def test_streams_plain_and_gzipped_jsonl_and_reports_line_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = root / "records.jsonl"
            compressed = root / "records.jsonl.gz"
            serialized = json.dumps(_record()) + "\n"
            plain.write_text(serialized, encoding="utf-8")
            with gzip.open(compressed, "wt", encoding="utf-8") as handle:
                handle.write(serialized)

            for path in (plain, compressed):
                iterator = iter_qasrl_records(
                    path,
                    release="2.1",
                    split="dev",
                )
                self.assertNotIsInstance(iterator, list)
                self.assertEqual(next(iterator).candidates[0].lemma, "send")
                with self.assertRaises(StopIteration):
                    next(iterator)

            plain.write_text(serialized + "\n", encoding="utf-8")
            iterator = iter_qasrl_records(plain, release="2.1", split="dev")
            next(iterator)
            with self.assertRaisesRegex(DatasetFormatError, r"records\.jsonl:2"):
                next(iterator)

    def test_rejects_upstream_key_mismatches_and_contradictory_judgments(self):
        cases = []

        wrong_verb_key = _record()
        wrong_verb_key["verbEntries"]["1"]["verbIndex"] = 2
        cases.append((wrong_verb_key, "expected 1"))

        wrong_question_key = _record()
        question = wrong_question_key["verbEntries"]["1"]["questionLabels"].pop(
            "Who sent something?"
        )
        wrong_question_key["verbEntries"]["1"]["questionLabels"][
            "Different key?"
        ] = question
        cases.append((wrong_question_key, "does not match"))

        invalid_with_answer = _record()
        judgment = next(
            iter(invalid_with_answer["verbEntries"]["1"]["questionLabels"].values())
        )["answerJudgments"][0]
        judgment["isValid"] = False
        cases.append((invalid_with_answer, "invalid but contains answer spans"))

        unknown_field = _record()
        unknown_field["invented"] = True
        cases.append((unknown_field, "unknown"))

        string_boolean = _record()
        question = next(
            iter(string_boolean["verbEntries"]["1"]["questionLabels"].values())
        )
        question["isPassive"] = "False"
        cases.append((string_boolean, "must be a JSON boolean"))

        realized_verb_slot = _record()
        question = next(
            iter(
                realized_verb_slot["verbEntries"]["1"]["questionLabels"].values()
            )
        )
        question["questionSlots"]["verb"] = "sent"
        cases.append((realized_verb_slot, "abstract release form"))

        missing_valid_spans = _record()
        judgment = next(
            iter(
                missing_valid_spans["verbEntries"]["1"]["questionLabels"].values()
            )
        )["answerJudgments"][0]
        del judgment["spans"]
        cases.append((missing_valid_spans, "valid but omits its spans field"))

        for raw, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(DatasetFormatError, message):
                    adapt_qasrl_record(raw, release="2.1", split="dev")

    def test_rejects_out_of_bounds_duplicate_and_malformed_spans(self):
        span_values = (
            [[0, 7]],
            [[2, 2]],
            [[0, 1], [0, 1]],
            [[0]],
            [[True, 1]],
        )
        for spans in span_values:
            raw = _record()
            judgment = next(
                iter(raw["verbEntries"]["1"]["questionLabels"].values())
            )["answerJudgments"][0]
            judgment["spans"] = spans
            with self.subTest(spans=spans):
                with self.assertRaises(DatasetFormatError):
                    adapt_qasrl_record(raw, release="2.1", split="dev")

    def test_rejects_invalid_index_identity_and_duplicate_documents(self):
        payload = _index_payload()
        payload["documents"]["dev"][0]["domain"] = "wikinews"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DatasetFormatError, "inconsistent"):
                load_qasrl_index(path)

            duplicate = _index_payload()
            duplicate["documents"]["dev"].append(
                dict(duplicate["documents"]["dev"][0])
            )
            path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(DatasetFormatError, "repeats document id"):
                load_qasrl_index(path)


if __name__ == "__main__":
    unittest.main()
