import json
import unittest

from semantic_action_extractor.annotation_schema import (
    ANNOTATION_SCHEMA_VERSION,
    AnnotationProvenance,
    AnnotationRecord,
    AnnotationToken,
    AnswerAlternative,
    EventivityJudgment,
    PredicateCandidate,
    QASRLQuestion,
    QASRLQuestionSlots,
    QuestionJudgment,
    TokenAlignedSpan,
    VerbInflectionParadigm,
)
from semantic_action_extractor.schema import TextSpan


def _token(index: int, text: str, start: int, end: int) -> AnnotationToken:
    return AnnotationToken(index=index, span=TextSpan(text=text, start=start, end=end))


def _aligned(text: str, start: int, end: int, token_start: int, token_end: int):
    return TokenAlignedSpan(
        span=TextSpan(text=text, start=start, end=end),
        token_start=token_start,
        token_end=token_end,
    )


class AnnotationSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = "The approval by Priya and Jordan of the merger was unanimous."
        self.tokens = (
            _token(0, "The", 0, 3),
            _token(1, "approval", 4, 12),
            _token(2, "by", 13, 15),
            _token(3, "Priya", 16, 21),
            _token(4, "and", 22, 25),
            _token(5, "Jordan", 26, 32),
            _token(6, "of", 33, 35),
            _token(7, "the", 36, 39),
            _token(8, "merger", 40, 46),
            _token(9, "was", 47, 50),
            _token(10, "unanimous", 51, 60),
            _token(11, ".", 60, 61),
        )
        self.provenance = AnnotationProvenance(
            dataset="qanom",
            release="1.0",
            split="train",
            source_id="wiki:42",
            record_id="qanom-train-42",
            document_id="wiki-document-7",
            metadata={"original_partition": "train", "review_rounds": (1, 2)},
        )

    def test_grounded_nominal_record_preserves_full_annotation_structure(self) -> None:
        joint_answer = AnswerAlternative(
            alternative_id="joint-approvers",
            spans=(
                _aligned("Priya", 16, 21, 3, 4),
                _aligned("Jordan", 26, 32, 5, 6),
            ),
            surface_form="Priya and Jordan",
        )
        priya_only = AnswerAlternative(
            alternative_id="priya-only",
            spans=(_aligned("Priya", 16, 21, 3, 4),),
        )
        who_question = QASRLQuestion(
            question_id="q-who",
            slots=QASRLQuestionSlots(
                wh="who",
                aux="_",
                subj="_",
                verb="past",
                obj="something",
                prep="_",
                obj2="_",
                verb_prefix="",
                verb_slot_inflection="Past",
            ),
            surface_form="Who approved something?",
            question_sources=("qanom-writer-1",),
            judgments=(
                QuestionJudgment(
                    source_id="qanom-validator-1",
                    is_valid=True,
                    answers=(joint_answer, priya_only),
                    confidence=0.812345678901234,
                    confidence_type="annotator_self_report",
                ),
                QuestionJudgment(
                    source_id="qanom-validator-2",
                    is_valid=True,
                    answers=(joint_answer,),
                ),
            ),
            is_negated=False,
            is_passive=False,
        )
        what_question = QASRLQuestion(
            question_id="q-what",
            slots=QASRLQuestionSlots(
                wh="what",
                aux="did",
                subj="someone",
                verb="stem",
                obj="_",
                prep="_",
                obj2="_",
                verb_prefix="",
                verb_slot_inflection="Stem",
            ),
            surface_form="What did someone approve?",
            question_sources=("qanom-writer-2",),
            judgments=(
                QuestionJudgment(
                    source_id="qanom-validator-1",
                    is_valid=True,
                    answers=(
                        AnswerAlternative(
                            alternative_id="approved-object",
                            spans=(_aligned("the merger", 36, 46, 7, 9),),
                        ),
                    ),
                ),
            ),
            is_negated=False,
            is_passive=False,
        )
        candidate = PredicateCandidate(
            candidate_id="predicate-approval",
            span=_aligned("approval", 4, 12, 1, 2),
            lemma="approval",
            predicate_type="nominal",
            related_verbal_form="approve",
            eventivity_judgments=(
                EventivityJudgment(
                    judgment_id="eventive-1",
                    annotator_id="anon-1",
                    is_eventive=True,
                ),
            ),
            questions=(who_question, what_question),
        )
        record = AnnotationRecord(
            text=self.text,
            tokens=self.tokens,
            provenance=self.provenance,
            candidates=(candidate,),
            metadata={"adapter": "pending", "flags": ["multi-answer"]},
        )

        payload = record.to_dict()

        self.assertEqual(payload["schema_version"], ANNOTATION_SCHEMA_VERSION)
        self.assertEqual(
            payload["candidates"][0]["related_verbal_form"],
            "approve",
        )
        self.assertEqual(len(payload["candidates"][0]["questions"]), 2)
        judgments = payload["candidates"][0]["questions"][0]["judgments"]
        self.assertEqual(len(judgments), 2)
        self.assertEqual(len(judgments[0]["answers"]), 2)
        self.assertEqual(len(judgments[0]["answers"][0]["spans"]), 2)
        self.assertEqual(judgments[0]["confidence"], 0.812345678901234)
        nominal_question = payload["candidates"][0]["questions"][0]
        self.assertIsNone(nominal_question["tense"])
        self.assertIsNone(nominal_question["is_perfect"])
        self.assertIsNone(nominal_question["is_progressive"])
        self.assertFalse(nominal_question["is_passive"])
        self.assertFalse(nominal_question["is_negated"])
        self.assertEqual(payload["provenance"]["document_id"], "wiki-document-7")
        self.assertEqual(payload["provenance"]["metadata"]["review_rounds"], [1, 2])
        json.dumps(payload)

    def test_preserves_negative_nominal_eventivity_candidate(self) -> None:
        text = "The approval rating rose."
        tokens = (
            _token(0, "The", 0, 3),
            _token(1, "approval", 4, 12),
            _token(2, "rating", 13, 19),
            _token(3, "rose", 20, 24),
            _token(4, ".", 24, 25),
        )
        negative = PredicateCandidate(
            candidate_id="approval-non-event",
            span=_aligned("approval", 4, 12, 1, 2),
            lemma="approval",
            predicate_type="nominal",
            related_verbal_form="approve",
            eventivity_judgments=(
                EventivityJudgment(
                    judgment_id="eventive-1",
                    is_eventive=False,
                    confidence=0.95,
                    confidence_type="annotator_self_report",
                ),
            ),
        )
        verbal = PredicateCandidate(
            candidate_id="predicate-rose",
            span=_aligned("rose", 20, 24, 3, 4),
            lemma="rise",
            predicate_type="verbal",
            verb_inflected_forms=VerbInflectionParadigm(
                stem="rise",
                present_singular_3rd="rises",
                present_participle="rising",
                past="rose",
                past_participle="risen",
            ),
        )
        record = AnnotationRecord(
            text=text,
            tokens=tokens,
            provenance=self.provenance,
            candidates=(negative, verbal),
        )

        candidate = record.to_dict()["candidates"][0]

        self.assertFalse(candidate["eventivity_judgments"][0]["is_eventive"])
        self.assertEqual(candidate["questions"], [])
        self.assertEqual(record.to_dict()["candidates"][1]["predicate_type"], "verbal")

    def test_qasrl_bank_release_fields_round_trip_without_realizing_slots(self) -> None:
        text = "Ava had been sending invoices."
        tokens = (
            _token(0, "Ava", 0, 3),
            _token(1, "had", 4, 7),
            _token(2, "been", 8, 12),
            _token(3, "sending", 13, 20),
            _token(4, "invoices", 21, 29),
            _token(5, ".", 29, 30),
        )
        inflections = VerbInflectionParadigm(
            stem="send",
            present_singular_3rd="sends",
            present_participle="sending",
            past="sent",
            past_participle="sent",
        )
        question = QASRLQuestion(
            question_id="What had someone been sending?",
            slots=QASRLQuestionSlots(
                wh="what",
                aux="had",
                subj="someone",
                verb="been presentParticiple",
                obj="_",
                prep="_",
                obj2="_",
            ),
            surface_form="What had someone been sending?",
            question_sources=(
                "turk-qasrl2-generator-17",
                "model-qasrl2-expansion",
            ),
            judgments=(
                QuestionJudgment(
                    source_id="turk-qasrl2-validator-9-eval",
                    is_valid=True,
                    answers=(
                        AnswerAlternative(
                            alternative_id="span-4-5",
                            spans=(_aligned("invoices", 21, 29, 4, 5),),
                        ),
                    ),
                ),
            ),
            tense="past",
            is_perfect=True,
            is_progressive=True,
            is_negated=False,
            is_passive=False,
        )
        record = AnnotationRecord(
            text=text,
            tokens=tokens,
            provenance=AnnotationProvenance(
                dataset="qa-srl-bank",
                release="2.1",
                split="train",
                source_id="WIKI1_DOC1_1",
                record_id="WIKI1_DOC1_1",
                document_id="WIKI1_DOC1",
                metadata={"layer": "expanded"},
            ),
            candidates=(
                PredicateCandidate(
                    candidate_id="WIKI1_DOC1_1:3",
                    span=_aligned("sending", 13, 20, 3, 4),
                    lemma="send",
                    predicate_type="verbal",
                    verb_inflected_forms=inflections,
                    questions=(question,),
                ),
            ),
        )

        payload = record.to_dict()
        candidate = payload["candidates"][0]
        serialized_question = candidate["questions"][0]

        self.assertEqual(
            candidate["verb_inflected_forms"],
            {
                "stem": "send",
                "presentSingular3rd": "sends",
                "presentParticiple": "sending",
                "past": "sent",
                "pastParticiple": "sent",
            },
        )
        self.assertEqual(
            serialized_question["slots"],
            {
                "wh": "what",
                "aux": "had",
                "subj": "someone",
                "verb": "been presentParticiple",
                "obj": "_",
                "prep": "_",
                "obj2": "_",
                "verb_prefix": None,
                "verb_slot_inflection": None,
            },
        )
        self.assertEqual(
            serialized_question["question_sources"],
            ["turk-qasrl2-generator-17", "model-qasrl2-expansion"],
        )
        self.assertEqual(serialized_question["tense"], "past")
        self.assertTrue(serialized_question["is_perfect"])
        self.assertTrue(serialized_question["is_progressive"])
        self.assertFalse(serialized_question["is_negated"])
        self.assertFalse(serialized_question["is_passive"])
        self.assertEqual(
            serialized_question["judgments"][0]["source_id"],
            "turk-qasrl2-validator-9-eval",
        )
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_rejects_realized_verb_in_abstract_release_slot(self) -> None:
        with self.assertRaisesRegex(ValueError, "abstract release form"):
            QASRLQuestionSlots(
                wh="what",
                aux="did",
                subj="someone",
                verb="send",
                obj="_",
                prep="_",
                obj2="_",
            )

    def test_maps_qanom_prefix_and_inflection_to_canonical_bank_verb(self) -> None:
        slots = QASRLQuestionSlots(
            wh="what",
            aux="has",
            subj="someone",
            verb="have been presentParticiple",
            obj="_",
            prep="_",
            obj2="_",
            verb_prefix="have~!~been",
            verb_slot_inflection="PresentParticiple",
        )

        payload = slots.to_dict()

        self.assertEqual(payload["verb"], "have been presentParticiple")
        self.assertEqual(payload["verb_prefix"], "have~!~been")
        self.assertEqual(payload["verb_slot_inflection"], "PresentParticiple")

    def test_preserves_qanom_missing_inflection_without_invention(self) -> None:
        slots = QASRLQuestionSlots(
            wh="what",
            aux="was",
            subj="_",
            verb="_",
            obj="_",
            prep="_",
            obj2="_",
            verb_prefix="being",
            verb_slot_inflection="",
        )

        payload = slots.to_dict()

        self.assertEqual(payload["verb"], "_")
        self.assertEqual(payload["verb_prefix"], "being")
        self.assertEqual(payload["verb_slot_inflection"], "")

    def test_requires_release_inflections_and_question_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "preserve verb_inflected_forms"):
            PredicateCandidate(
                candidate_id="predicate-sent",
                span=_aligned("sent", 4, 8, 1, 2),
                lemma="send",
                predicate_type="verbal",
            )

        answer = AnswerAlternative(
            alternative_id="answer-ava",
            spans=(_aligned("Ava", 0, 3, 0, 1),),
        )
        with self.assertRaisesRegex(ValueError, "question sources"):
            QASRLQuestion(
                question_id="Who sent something?",
                slots=QASRLQuestionSlots(
                    wh="who",
                    aux="_",
                    subj="_",
                    verb="past",
                    obj="something",
                    prep="_",
                    obj2="_",
                ),
                surface_form="Who sent something?",
                question_sources=(),
                judgments=(
                    QuestionJudgment(
                        source_id="turk-validator-1",
                        is_valid=True,
                        answers=(answer,),
                    ),
                ),
                tense="past",
                is_perfect=False,
                is_progressive=False,
                is_negated=False,
                is_passive=False,
            )

        question_without_bank_fields = QASRLQuestion(
            question_id="Who sent something?",
            slots=QASRLQuestionSlots(
                wh="who",
                aux="_",
                subj="_",
                verb="past",
                obj="something",
                prep="_",
                obj2="_",
            ),
            surface_form="Who sent something?",
            question_sources=("turk-writer-1",),
            judgments=(
                QuestionJudgment(
                    source_id="turk-validator-1",
                    is_valid=True,
                    answers=(answer,),
                ),
            ),
            is_negated=False,
            is_passive=False,
        )
        with self.assertRaisesRegex(ValueError, "Bank grammar fields"):
            PredicateCandidate(
                candidate_id="predicate-sent",
                span=_aligned("sent", 4, 8, 1, 2),
                lemma="send",
                predicate_type="verbal",
                verb_inflected_forms=VerbInflectionParadigm(
                    stem="send",
                    present_singular_3rd="sends",
                    present_participle="sending",
                    past="sent",
                    past_participle="sent",
                ),
                questions=(question_without_bank_fields,),
            )

    def test_rejects_invalid_metadata_and_confidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            AnnotationProvenance(
                dataset="qanom",
                release="1.0",
                split="train",
                source_id="source",
                record_id="record",
                metadata={"bad": object()},
            )

        with self.assertRaisesRegex(ValueError, "provided together"):
            EventivityJudgment(
                judgment_id="eventive-1",
                is_eventive=True,
                confidence=0.9,
            )

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            EventivityJudgment(
                judgment_id="eventive-1",
                is_eventive=True,
                confidence=float("nan"),
                confidence_type="model_probability",
            )

    def test_rejects_unrecognized_explicit_schema_version(self) -> None:
        with self.assertRaisesRegex(ValueError, ANNOTATION_SCHEMA_VERSION):
            AnnotationRecord(
                text=self.text,
                tokens=self.tokens,
                provenance=self.provenance,
                schema_version="9.9.9",
            )

    def test_rejects_source_and_token_boundary_mismatch(self) -> None:
        candidate = PredicateCandidate(
            candidate_id="misaligned",
            span=_aligned("approval", 4, 12, 0, 1),
            lemma="approval",
            predicate_type="nominal",
        )

        with self.assertRaisesRegex(ValueError, "boundaries do not align"):
            AnnotationRecord(
                text=self.text,
                tokens=self.tokens,
                provenance=self.provenance,
                candidates=(candidate,),
            )

        invalid_tokens = (
            _token(0, "Tho", 0, 3),
            *self.tokens[1:],
        )
        with self.assertRaisesRegex(ValueError, "map exactly"):
            AnnotationRecord(
                text=self.text,
                tokens=invalid_tokens,
                provenance=self.provenance,
            )


if __name__ == "__main__":
    unittest.main()
