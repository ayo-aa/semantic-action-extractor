import csv
import tempfile
import unittest
from pathlib import Path

from semantic_action_extractor.datasets.common import DatasetFormatError
from semantic_action_extractor.datasets.qanom import (
    QANOM_RELEASE,
    QANOM_SHARED_FIELDS,
    iter_qanom_csv,
    iter_qanom_rows,
    load_qanom_csv,
)


SENTENCE_ID = "Wiki1k:wikipedia:42:1:0"
SENTENCE = "The approval by Priya and Jordan followed debate ."


def _row(**updates):
    values = {
        "qasrl_id": SENTENCE_ID,
        "sentence": SENTENCE,
        "target_idx": "1",
        "key": f"{SENTENCE_ID}_1",
        "noun": "approval",
        "worker_id": "Worker-final",
        "source_worker_ids": "Worker-source-1~!~Worker-source-2",
        "source_assign_id": "assignment-17",
        "is_verbal": "True",
        "verb_form": "approve",
        "question": "Who approved something?",
        "answer_range": "3:4~!~5:6",
        "answer": "Priya~!~Jordan",
        "wh": "who",
        "subj": "",
        "obj": "something",
        "obj2": "",
        "aux": "",
        "prep": "",
        "verb_prefix": "",
        "is_passive": "False",
        "is_negated": "False",
        "verb_slot_inflection": "Past",
    }
    values.update(updates)
    return values


def _negative_row(**updates):
    values = _row(
        target_idx="7",
        key=f"{SENTENCE_ID}_7",
        noun="debate",
        worker_id="Worker-negative",
        source_worker_ids="",
        source_assign_id="None",
        is_verbal="False",
        verb_form="debate",
        question="",
        answer_range="",
        answer="",
        wh="",
        subj="",
        obj="",
        obj2="",
        aux="",
        prep="",
        verb_prefix="",
        is_passive="False",
        is_negated="False",
        verb_slot_inflection="",
    )
    values.update(updates)
    return values


class QANomAdapterTests(unittest.TestCase):
    def test_reconstructs_evidence_preserving_record_and_deduplicates_exact_qa(self):
        first_question = _row()
        duplicate = dict(first_question)
        second_question = _row(
            question="What did someone approve?",
            answer_range="7:8",
            answer="debate",
            wh="what",
            subj="someone",
            obj="",
            aux="did",
            source_worker_ids="Worker-source-3",
            source_assign_id="assignment-18",
        )

        records = tuple(
            iter_qanom_rows(
                [first_question, duplicate, second_question, _negative_row()],
                split="train",
            )
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.provenance.dataset, "qanom")
        self.assertEqual(record.provenance.release, QANOM_RELEASE)
        self.assertEqual(record.provenance.split, "train")
        self.assertEqual(record.provenance.source_id, SENTENCE_ID)
        self.assertEqual(
            record.provenance.document_id,
            "Wiki1k:wikipedia:42",
        )
        self.assertEqual(len(record.candidates), 2)

        positive, negative = record.candidates
        self.assertEqual(positive.span.span.text, "approval")
        self.assertEqual(positive.lemma, "approval")
        self.assertEqual(positive.metadata["lemma_source"], "raw_upstream_noun")
        self.assertEqual(positive.metadata["lemma_source_field"], "noun")
        self.assertEqual(positive.related_verbal_form, "approve")
        self.assertTrue(positive.eventivity_judgments[0].is_eventive)
        self.assertEqual(
            positive.eventivity_judgments[0].annotator_id,
            "Worker-final",
        )
        self.assertEqual(positive.metadata["duplicate_qa_row_count"], 1)
        self.assertEqual(record.metadata["duplicate_qa_row_count"], 1)
        self.assertEqual(
            positive.metadata["source_assignment_ids"],
            ("assignment-17", "assignment-18"),
        )
        self.assertEqual(
            positive.metadata["source_worker_ids"],
            ("Worker-source-1", "Worker-source-2", "Worker-source-3"),
        )
        self.assertEqual(len(positive.questions), 2)

        who = positive.questions[0]
        self.assertEqual(
            who.question_sources,
            ("Worker-source-1", "Worker-source-2"),
        )
        self.assertEqual(who.judgments[0].source_id, "Worker-final")
        self.assertEqual(
            who.judgments[0].metadata["source_assignment_ids"],
            ("assignment-17",),
        )
        self.assertEqual(len(who.judgments[0].answers), 2)
        self.assertEqual(
            [answer.surface_form for answer in who.judgments[0].answers],
            ["Priya", "Jordan"],
        )
        self.assertEqual(
            [answer.spans[0].token_start for answer in who.judgments[0].answers],
            [3, 5],
        )
        self.assertEqual(who.slots.verb, "past")
        self.assertEqual(who.slots.verb_prefix, "")
        self.assertEqual(who.slots.verb_slot_inflection, "Past")

        self.assertEqual(negative.lemma, "debate")
        self.assertFalse(negative.eventivity_judgments[0].is_eventive)
        self.assertEqual(negative.questions, ())
        self.assertEqual(negative.metadata["source_assignment_ids"], ("None",))

    def test_train_dev_and_test_header_variants_are_parsed_by_name(self):
        train = _row()
        train.pop("source_assign_id")
        dev = _row()
        test = _row()
        test.pop("source_assign_id")
        test["Unnamed: 0"] = "395.0"

        train_record = tuple(iter_qanom_rows([train], split="train"))[0]
        dev_record = tuple(iter_qanom_rows([dev], split="dev"))[0]
        test_record = tuple(iter_qanom_rows([test], split="test"))[0]

        self.assertNotIn(
            "discarded_release_artifact_fields",
            train_record.metadata,
        )
        self.assertEqual(
            dev_record.candidates[0].metadata["source_assignment_ids"],
            ("assignment-17",),
        )
        self.assertEqual(
            test_record.metadata["discarded_release_artifact_fields"],
            ("Unnamed: 0",),
        )
        self.assertEqual(
            test_record.metadata["nonempty_release_artifact_value_counts"],
            {"Unnamed: 0": 1},
        )
        self.assertNotIn("395.0", repr(test_record.to_dict()["candidates"]))

    def test_file_api_accepts_reordered_columns_and_multiple_sentences(self):
        first = _row()
        second_id = "Wiki1k:wikinews:99:2:1"
        second = _negative_row(
            qasrl_id=second_id,
            sentence="A discussion ended .",
            target_idx="1",
            key=f"{second_id}_1",
            noun="discussion",
            verb_form="discuss",
        )
        fieldnames = list(reversed(tuple(first.keys())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annot.dev.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([first, second])

            streamed = tuple(iter_qanom_csv(path, split="dev"))
            loaded = load_qanom_csv(path, split="dev")

        self.assertEqual(len(streamed), 2)
        self.assertEqual(
            [record.provenance.source_id for record in streamed],
            [SENTENCE_ID, second_id],
        )
        self.assertEqual(
            [record.to_dict() for record in loaded],
            [record.to_dict() for record in streamed],
        )

    def test_question_ids_are_stable_when_qa_rows_are_reordered(self):
        first = _row()
        second = _row(
            question="What did someone approve?",
            answer_range="7:8",
            answer="debate",
            wh="what",
            subj="someone",
            obj="",
            aux="did",
        )

        forward = tuple(iter_qanom_rows([first, second], split="train"))[0]
        reverse = tuple(iter_qanom_rows([second, first], split="train"))[0]
        forward_ids = {
            question.surface_form: question.question_id
            for question in forward.candidates[0].questions
        }
        reverse_ids = {
            question.surface_form: question.question_id
            for question in reverse.candidates[0].questions
        }

        self.assertEqual(forward_ids, reverse_ids)

    def test_rejects_missing_and_unknown_header_fields(self):
        missing = _row()
        missing.pop("noun")
        unknown = _row(extra_column="not part of QANom")

        with self.assertRaisesRegex(DatasetFormatError, "missing=.*noun"):
            tuple(iter_qanom_rows([missing], split="train"))
        with self.assertRaisesRegex(DatasetFormatError, "unknown=.*extra_column"):
            tuple(iter_qanom_rows([unknown], split="train"))

    def test_rejects_a_row_whose_fields_differ_from_the_header(self):
        first = _row()
        second = _row(question="What did someone approve?")
        second.pop("source_assign_id")

        with self.assertRaisesRegex(DatasetFormatError, "fields differ"):
            tuple(iter_qanom_rows([first, second], split="train"))

    def test_rejects_changed_candidate_constants(self):
        second = _row(
            question="What did someone approve?",
            answer_range="7:8",
            answer="debate",
            wh="what",
            subj="someone",
            obj="",
            aux="did",
            worker_id="Different-final-worker",
        )

        with self.assertRaisesRegex(DatasetFormatError, "worker_id changed"):
            tuple(iter_qanom_rows([_row(), second], split="train"))

    def test_rejects_invalid_target_key_and_booleans(self):
        cases = (
            (_row(noun="approvals"), "does not match target token"),
            (_row(key="wrong"), "key must be"),
            (_row(target_idx="not-an-index"), "target_idx must be an integer"),
            (_row(is_verbal="yes"), "is_verbal must be True or False"),
            (_row(is_passive="0"), "is_passive must be True or False"),
            (_row(is_negated="FALSE"), "is_negated must be True or False"),
        )
        for row, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(DatasetFormatError, message):
                    tuple(iter_qanom_rows([row], split="train"))

    def test_validates_case_normalized_noun_against_target_token(self):
        sentence_id = "Wiki1k:wikipedia:77:2:0"
        capitalized_target = _row(
            qasrl_id=sentence_id,
            sentence="Approval followed debate .",
            target_idx="0",
            key=f"{sentence_id}_0",
            noun="approval",
            question="What was approved?",
            answer_range="2:3",
            answer="debate",
            wh="what",
            subj="",
            obj="",
            aux="was",
            is_passive="True",
        )

        candidate = tuple(
            iter_qanom_rows([capitalized_target], split="train")
        )[0].candidates[0]

        self.assertEqual(candidate.lemma, "approval")
        self.assertEqual(candidate.span.span.text, "Approval")
        self.assertEqual(candidate.metadata["target_surface_form"], "Approval")
        self.assertFalse(candidate.metadata["noun_matches_target_case_sensitive"])

    def test_rejects_invalid_or_nonparallel_answers(self):
        cases = (
            (_row(answer_range="3-4", answer="Priya"), "must be START:END"),
            (_row(answer_range="3:99", answer="Priya"), "token range"),
            (_row(answer="Priya"), "same number"),
            (_row(answer="Jordan~!~Priya"), "does not match range"),
            (_row(answer_range="3:4~!~", answer="Priya~!~Jordan"), "empty separated"),
        )
        for row, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(DatasetFormatError, message):
                    tuple(iter_qanom_rows([row], split="train"))

    def test_rejects_inconsistent_question_fields(self):
        cases = (
            (_negative_row(wh="who"), "empty question must also have empty"),
            (_row(wh=""), "wh cannot be empty"),
            (_row(wh="which"), "unsupported wh"),
            (_row(verb_slot_inflection="Future"), "unsupported verb_slot"),
            (_row(source_worker_ids="Worker-1~!~~!~Worker-2"), "empty source ID"),
        )
        for row, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(DatasetFormatError, message):
                    tuple(iter_qanom_rows([row], split="train"))

    def test_preserves_upstream_eventivity_question_conflict(self):
        conflicted = _row(is_verbal="False")

        candidate = tuple(iter_qanom_rows([conflicted], split="dev"))[0].candidates[0]

        self.assertFalse(candidate.eventivity_judgments[0].is_eventive)
        self.assertEqual(len(candidate.questions), 1)
        self.assertTrue(
            candidate.metadata["upstream_eventivity_question_conflict"]
        )

    def test_reconstructs_missing_answer_text_from_verified_range(self):
        missing_text = _row(
            answer_range="3:4",
            answer="",
            source_worker_ids="Worker-source-1",
        )

        candidate = tuple(iter_qanom_rows([missing_text], split="dev"))[0].candidates[0]
        answer = candidate.questions[0].judgments[0].answers[0]

        self.assertEqual(answer.surface_form, "Priya")
        self.assertEqual(answer.spans[0].span.text, "Priya")
        self.assertTrue(answer.metadata["upstream_answer_text_missing"])

    def test_rejects_sentence_groups_that_reappear(self):
        other_id = "Wiki1k:wikinews:99:2:1"
        other = _negative_row(
            qasrl_id=other_id,
            sentence="A discussion ended .",
            target_idx="1",
            key=f"{other_id}_1",
            noun="discussion",
            verb_form="discuss",
        )

        with self.assertRaisesRegex(DatasetFormatError, "reappears"):
            tuple(iter_qanom_rows([_row(), other, _negative_row()], split="train"))

    def test_declared_shared_schema_matches_every_synthetic_row(self):
        self.assertEqual(QANOM_SHARED_FIELDS, frozenset(_row()) - {"source_assign_id"})


if __name__ == "__main__":
    unittest.main()
