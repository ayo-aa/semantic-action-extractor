import unittest

from semantic_action_extractor.annotation_schema import (
    AnnotationProvenance,
    AnnotationRecord,
    AnswerAlternative,
    EventivityJudgment,
    PredicateCandidate,
    QASRLQuestion,
    QASRLQuestionSlots,
    QuestionJudgment,
    VerbInflectionParadigm,
)
from semantic_action_extractor.datasets.common import (
    canonicalize_tokens,
    token_aligned_span,
)
from semantic_action_extractor.evaluation.consolidation import (
    consolidate_annotations,
)
from semantic_action_extractor.evaluation.matching import (
    greedy_weight_matching,
    maximum_cardinality_weight_matching,
)
from semantic_action_extractor.evaluation.question_equivalence import (
    EXACT_SLOTS_V1,
    QANOM_COARSE_ROLE_V1,
    QASRL_GS_FIVE_FIELD_V1,
    qanom_coarse_role,
    questions_equivalent,
)
from semantic_action_extractor.evaluation.scorers import (
    PRIMARY_END_TO_END_V1,
    QANOM_REFERENCE_V1,
    QASRL_GS_COMPATIBLE_V1,
    score_corpora,
)
from semantic_action_extractor.evaluation.types import (
    EvaluationArgument,
    EvaluationCorpus,
    EvaluationPredicate,
    EvaluationQAPair,
    EvaluationQuestion,
    PredicateKey,
)


def _question(
    *,
    surface: str = "Who approved something?",
    wh: str = "who",
    aux: str = "_",
    subj: str = "_",
    verb: str = "past",
    obj: str = "something",
    prep: str = "_",
    obj2: str = "_",
    passive: bool = False,
    negated: bool = False,
) -> EvaluationQuestion:
    return EvaluationQuestion(
        surface_form=surface,
        wh=wh,
        aux=aux,
        subj=subj,
        verb=verb,
        obj=obj,
        prep=prep,
        obj2=obj2,
        is_passive=passive,
        is_negated=negated,
    )


def _pair(
    pair_id: str,
    token_span: tuple[int, int],
    *,
    character_span: tuple[int, int] | None = None,
    question: EvaluationQuestion | None = None,
    role_id: str | None = None,
) -> EvaluationQAPair:
    return EvaluationQAPair(
        pair_id=pair_id,
        role_id=role_id,
        question=question or _question(),
        argument=EvaluationArgument(
            token_spans=(token_span,),
            character_spans=(character_span,) if character_span else None,
        ),
    )


def _predicate(
    source_id: str,
    token_start: int,
    *pairs: EvaluationQAPair,
    eventive: bool = True,
    predicate_type: str = "verbal",
    lemma: str = "approve",
) -> EvaluationPredicate:
    return EvaluationPredicate(
        key=PredicateKey(source_id, token_start, token_start + 1, predicate_type),
        is_eventive=eventive,
        lemma=lemma,
        pairs=tuple(pairs),
    )


def _score(
    gold: EvaluationCorpus,
    predicted: EvaluationCorpus,
    *,
    mode: str = PRIMARY_END_TO_END_V1,
):
    return score_corpora(
        gold,
        predicted,
        mode=mode,
        predicate_source="synthetic-fixture",
        consolidation_rule="synthetic-fixture-v1",
    )


class MatchingTests(unittest.TestCase):
    def test_optimal_matching_preserves_cardinality_when_greedy_does_not(self) -> None:
        matrix = ((0.9, 0.8), (0.85, None))
        edge = lambda predicted, gold: matrix[predicted][gold]

        greedy = greedy_weight_matching(2, 2, edge)
        optimal = maximum_cardinality_weight_matching(2, 2, edge)

        self.assertEqual(len(greedy), 1)
        self.assertEqual(len(optimal), 2)
        self.assertEqual(
            {(match.predicted_index, match.gold_index) for match in optimal},
            {(0, 1), (1, 0)},
        )

    def test_equal_weight_ties_are_stable(self) -> None:
        edge = lambda predicted, gold: 1.0

        first = maximum_cardinality_weight_matching(3, 3, edge)
        second = maximum_cardinality_weight_matching(3, 3, edge)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)


class QuestionEquivalenceTests(unittest.TestCase):
    def test_exact_contract_uses_all_slots(self) -> None:
        left = _question(prep="to")
        right = _question(prep="for")

        self.assertFalse(questions_equivalent(left, right, mode=EXACT_SLOTS_V1))
        self.assertTrue(
            questions_equivalent(left, right, mode=QASRL_GS_FIVE_FIELD_V1)
        )

    def test_qasrl_surface_identity_is_compatible(self) -> None:
        left = _question(surface="WHO APPROVED SOMETHING?", subj="someone")
        right = _question(surface="who approved something?", subj="_")

        self.assertTrue(
            questions_equivalent(left, right, mode=QASRL_GS_FIVE_FIELD_V1)
        )

    def test_qanom_role_mapping_handles_active_and_passive_core_roles(self) -> None:
        active_subject = _question()
        passive_subject = _question(
            surface="Who was something approved by?",
            subj="something",
            obj="_",
            prep="by",
            passive=True,
        )

        self.assertEqual(qanom_coarse_role(active_subject), "R0")
        self.assertEqual(qanom_coarse_role(passive_subject), "R0")
        self.assertTrue(
            questions_equivalent(
                active_subject,
                passive_subject,
                mode=QANOM_COARSE_ROLE_V1,
            )
        )


class ScorerTests(unittest.TestCase):
    def test_primary_threshold_is_inclusive_at_one_half(self) -> None:
        key_args = (
            _pair("gold", (0, 4), character_span=(0, 8)),
        )
        prediction_args = (
            _pair("predicted", (0, 2), character_span=(0, 4)),
        )
        result = _score(
            EvaluationCorpus((_predicate("sentence", 5, *key_args),)),
            EvaluationCorpus((_predicate("sentence", 5, *prediction_args),)),
        )

        self.assertEqual(result.unlabeled_arguments.true_positive, 1)

    def test_qanom_threshold_is_strictly_greater_than_point_three(self) -> None:
        gold = EvaluationCorpus(
            (_predicate("sentence", 5, _pair("gold", (0, 10))),)
        )
        predicted = EvaluationCorpus(
            (_predicate("sentence", 5, _pair("predicted", (0, 3))),)
        )

        result = _score(gold, predicted, mode=QANOM_REFERENCE_V1)

        self.assertEqual(result.unlabeled_arguments.true_positive, 0)
        self.assertEqual(result.unlabeled_arguments.false_positive, 1)
        self.assertEqual(result.unlabeled_arguments.false_negative, 1)

    def test_primary_union_penalizes_missed_and_spurious_predicates(self) -> None:
        gold = EvaluationCorpus(
            (_predicate("gold-sentence", 2, _pair("gold", (0, 1))),)
        )
        predicted = EvaluationCorpus(
            (_predicate("other-sentence", 2, _pair("predicted", (0, 1))),)
        )

        primary = _score(gold, predicted, mode=PRIMARY_END_TO_END_V1)
        compatible = _score(
            gold, predicted, mode=QASRL_GS_COMPATIBLE_V1
        )
        qanom = _score(gold, predicted, mode=QANOM_REFERENCE_V1)

        self.assertEqual(primary.evaluated_predicates, 2)
        self.assertEqual(primary.labeled_arguments.false_positive, 1)
        self.assertEqual(primary.labeled_arguments.false_negative, 1)
        self.assertEqual(compatible.evaluated_predicates, 1)
        self.assertEqual(compatible.labeled_arguments.false_positive, 0)
        self.assertEqual(compatible.labeled_arguments.false_negative, 1)
        self.assertEqual(qanom.evaluated_predicates, 0)
        self.assertEqual(qanom.labeled_arguments.f1, 1.0)
        self.assertEqual(primary.candidate_detection.false_positive, 1)
        self.assertEqual(primary.candidate_detection.false_negative, 1)

    def test_prediction_only_negative_is_not_an_eventivity_true_negative(self) -> None:
        predicted = EvaluationCorpus(
            (
                _predicate(
                    "prediction-only",
                    2,
                    eventive=False,
                    predicate_type="nominal",
                ),
            )
        )

        result = _score(EvaluationCorpus(), predicted)

        self.assertEqual(result.candidate_detection.false_positive, 1)
        self.assertEqual(result.eventivity.true_negative, 0)
        self.assertEqual(result.shared_eventivity_candidates, 0)

    def test_qanom_omits_arguments_when_eventivity_disagrees(self) -> None:
        gold_predicate = _predicate(
            "sentence",
            2,
            _pair("gold", (0, 1)),
            predicate_type="nominal",
        )
        predicted_predicate = _predicate(
            "sentence",
            2,
            eventive=False,
            predicate_type="nominal",
        )

        result = _score(
            EvaluationCorpus((gold_predicate,)),
            EvaluationCorpus((predicted_predicate,)),
            mode=QANOM_REFERENCE_V1,
        )

        self.assertEqual(result.skipped_eventivity_mismatches, 1)
        self.assertEqual(result.eventivity.false_negative, 1)
        self.assertEqual(result.unlabeled_arguments, type(result.unlabeled_arguments)())

    def test_duplicate_prediction_is_a_false_positive(self) -> None:
        gold = EvaluationCorpus(
            (_predicate("sentence", 2, _pair("gold", (0, 1))),)
        )
        predicted = EvaluationCorpus(
            (
                _predicate(
                    "sentence",
                    2,
                    _pair("first", (0, 1)),
                    _pair("duplicate", (0, 1)),
                ),
            )
        )

        result = _score(gold, predicted)

        self.assertEqual(result.labeled_arguments.true_positive, 1)
        self.assertEqual(result.labeled_arguments.false_positive, 1)
        self.assertEqual(result.labeled_arguments.false_negative, 0)

    def test_qanom_reference_matches_duplicate_span_values_only_once(self) -> None:
        gold = EvaluationCorpus(
            (
                _predicate(
                    "sentence",
                    2,
                    _pair("gold-1", (0, 1)),
                    _pair("gold-2", (0, 1)),
                ),
            )
        )
        predicted = EvaluationCorpus(
            (
                _predicate(
                    "sentence",
                    2,
                    _pair("predicted-1", (0, 1)),
                    _pair("predicted-2", (0, 1)),
                ),
            )
        )

        result = _score(gold, predicted, mode=QANOM_REFERENCE_V1)

        self.assertEqual(result.unlabeled_arguments.true_positive, 1)
        self.assertEqual(result.unlabeled_arguments.false_positive, 1)
        self.assertEqual(result.unlabeled_arguments.false_negative, 1)

    def test_qanom_reference_reports_grouped_role_alignment(self) -> None:
        gold = EvaluationCorpus(
            (
                _predicate(
                    "sentence",
                    2,
                    _pair("gold-1", (0, 1), role_id="gold-role"),
                    _pair("gold-2", (3, 4), role_id="gold-role"),
                ),
            )
        )
        predicted = EvaluationCorpus(
            (
                _predicate(
                    "sentence",
                    2,
                    _pair("pred-1", (0, 1), role_id="pred-role"),
                    _pair("pred-2", (3, 4), role_id="pred-role"),
                ),
            )
        )

        result = _score(gold, predicted, mode=QANOM_REFERENCE_V1)

        self.assertEqual(result.reference_roles.true_positive, 1)
        self.assertEqual(result.reference_roles.false_positive, 0)
        self.assertEqual(result.reference_roles.false_negative, 0)
        self.assertEqual(
            result.settings["reference_revision"],
            "2bce70e8a39b40157ba97f38e1a8ae7619b30162",
        )

    def test_exact_token_and_character_metrics_are_separate(self) -> None:
        gold = EvaluationCorpus(
            (
                _predicate(
                    "sentence",
                    2,
                    _pair("gold", (0, 1), character_span=(0, 4)),
                ),
            )
        )
        predicted = EvaluationCorpus(
            (
                _predicate(
                    "sentence",
                    2,
                    _pair("predicted", (0, 1), character_span=(1, 5)),
                ),
            )
        )

        result = _score(gold, predicted)

        self.assertEqual(result.exact_token_arguments.true_positive, 1)
        self.assertEqual(result.exact_character_arguments.true_positive, 0)

    def test_empty_corpora_have_declared_zero_denominator_behavior(self) -> None:
        result = _score(EvaluationCorpus(), EvaluationCorpus())

        self.assertEqual(result.eventivity.accuracy, 1.0)
        self.assertEqual(result.labeled_arguments.precision, 1.0)
        self.assertEqual(result.labeled_arguments.recall, 1.0)
        self.assertEqual(result.labeled_arguments.f1, 1.0)


class ConsolidationTests(unittest.TestCase):
    def _record(
        self,
        *,
        eventive: bool = True,
        empty_valid: bool = False,
        duplicate_valid: bool = True,
    ) -> AnnotationRecord:
        text, tokens = canonicalize_tokens(
            ("Maya", "approved", "refunds", "."),
            label="synthetic tokens",
        )
        answer = AnswerAlternative(
            alternative_id="refunds",
            spans=(token_aligned_span(text, tokens, 2, 3, label="answer"),),
        )
        judgments = [
            QuestionJudgment(
                source_id="validator-1",
                is_valid=True,
                answers=() if empty_valid else (answer,),
                metadata=(
                    {"upstream_empty_valid_spans": True}
                    if empty_valid
                    else {}
                ),
            ),
            QuestionJudgment(
                source_id="validator-invalid",
                is_valid=False,
            ),
        ]
        if duplicate_valid and not empty_valid:
            judgments.append(
                QuestionJudgment(
                    source_id="validator-2",
                    is_valid=True,
                    answers=(answer,),
                )
            )
        question = QASRLQuestion(
            question_id="what-approved",
            slots=QASRLQuestionSlots(
                wh="what",
                aux="did",
                subj="someone",
                verb="stem",
                obj="_",
                prep="_",
                obj2="_",
            ),
            surface_form="What did someone approve?",
            question_sources=("writer",),
            judgments=tuple(judgments),
            tense="past" if eventive else None,
            is_perfect=False if eventive else None,
            is_progressive=False if eventive else None,
            is_negated=False,
            is_passive=False,
        )
        if eventive:
            candidate = PredicateCandidate(
                candidate_id="approved",
                span=token_aligned_span(text, tokens, 1, 2, label="predicate"),
                lemma="approve",
                predicate_type="verbal",
                verb_inflected_forms=VerbInflectionParadigm(
                    stem="approve",
                    present_singular_3rd="approves",
                    present_participle="approving",
                    past="approved",
                    past_participle="approved",
                ),
                questions=(question,),
            )
        else:
            candidate = PredicateCandidate(
                candidate_id="approval",
                span=token_aligned_span(text, tokens, 1, 2, label="predicate"),
                lemma="approval",
                predicate_type="nominal",
                related_verbal_form="approve",
                eventivity_judgments=(
                    EventivityJudgment(
                        judgment_id="eventivity",
                        is_eventive=False,
                    ),
                ),
                questions=(question,),
                metadata={"upstream_eventivity_question_conflict": True},
            )
        return AnnotationRecord(
            text=text,
            tokens=tokens,
            provenance=AnnotationProvenance(
                dataset="synthetic",
                release="1",
                split="development",
                source_id="synthetic:1",
                record_id="synthetic:development:1",
                document_id="synthetic-document",
            ),
            candidates=(candidate,),
        )

    def test_valid_union_deduplicates_answers_across_judgments(self) -> None:
        result = consolidate_annotations((self._record(),))

        self.assertEqual(len(result.corpus.predicates[0].pairs), 1)
        self.assertEqual(result.counts["duplicate_answer_alternatives"], 1)
        self.assertEqual(result.counts["qa_pairs"], 1)

    def test_valid_empty_answer_is_counted_without_inventing_a_pair(self) -> None:
        result = consolidate_annotations(
            (self._record(empty_valid=True, duplicate_valid=False),)
        )

        self.assertEqual(result.corpus.predicates[0].pairs, ())
        self.assertEqual(result.counts["valid_judgments_without_answers"], 1)

    def test_non_eventive_question_conflict_stays_out_of_gold_pairs(self) -> None:
        result = consolidate_annotations((self._record(eventive=False),))

        self.assertFalse(result.corpus.predicates[0].is_eventive)
        self.assertEqual(result.corpus.predicates[0].pairs, ())
        self.assertEqual(result.counts["eventivity_question_conflicts"], 1)


if __name__ == "__main__":
    unittest.main()
