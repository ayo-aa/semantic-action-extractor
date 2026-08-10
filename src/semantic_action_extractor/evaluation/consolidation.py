"""Named conversion from raw annotation evidence to scorer-ready gold pairs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

from ..annotation_schema import AnnotationRecord, PredicateCandidate, QASRLQuestion
from ..datasets.common import DatasetFormatError
from .types import (
    EvaluationArgument,
    EvaluationCorpus,
    EvaluationMentionQualifier,
    EvaluationPredicate,
    EvaluationQAPair,
    EvaluationQuestion,
    PredicateKey,
)


VALID_JUDGMENT_UNION_V1 = "valid-judgment-union-v1"


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    corpus: EvaluationCorpus
    rule: str
    counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "counts": dict(sorted(self.counts.items())),
            "predicate_count": len(self.corpus.predicates),
        }


def consolidate_annotations(
    records: Iterable[AnnotationRecord],
    *,
    rule: str = VALID_JUDGMENT_UNION_V1,
) -> ConsolidationResult:
    """Create an evaluation view without flattening the stored raw judgments.

    Version 1 retains a question when at least one judgment marks it valid and
    takes the union of distinct answer alternatives from all valid judgments.
    Invalid judgments remain in the annotation record but do not create gold QA
    pairs. Exact duplicate alternatives across workers are counted once.
    """

    if rule != VALID_JUDGMENT_UNION_V1:
        raise ValueError(f"unsupported consolidation rule: {rule}")

    counts = {
        "records": 0,
        "predicates": 0,
        "eventive_predicates": 0,
        "non_eventive_predicates": 0,
        "questions_seen": 0,
        "questions_without_valid_judgment": 0,
        "valid_judgments_without_answers": 0,
        "qa_pairs": 0,
        "duplicate_answer_alternatives": 0,
        "eventivity_question_conflicts": 0,
    }
    predicates: list[EvaluationPredicate] = []
    for record in records:
        counts["records"] += 1
        for candidate in record.candidates:
            counts["predicates"] += 1
            eventive = _eventivity(candidate)
            counts[
                "eventive_predicates" if eventive else "non_eventive_predicates"
            ] += 1
            pairs: list[EvaluationQAPair] = []
            if candidate.questions and not eventive:
                counts["eventivity_question_conflicts"] += 1
            if eventive:
                for question in candidate.questions:
                    counts["questions_seen"] += 1
                    question_pairs, local_counts = _consolidate_question(question)
                    pairs.extend(question_pairs)
                    for key, value in local_counts.items():
                        counts[key] += value

            predicates.append(
                EvaluationPredicate(
                    key=PredicateKey(
                        source_id=record.provenance.source_id,
                        token_start=candidate.span.token_start,
                        token_end=candidate.span.token_end,
                        predicate_type=candidate.predicate_type,
                    ),
                    is_eventive=eventive,
                    lemma=candidate.lemma,
                    pairs=tuple(pairs),
                    mention_qualifiers=(
                        None
                        if candidate.mention_qualifiers is None
                        else tuple(
                            EvaluationMentionQualifier(
                                kind=qualifier.kind,
                                evidence=EvaluationArgument(
                                    token_spans=tuple(
                                        (span.token_start, span.token_end)
                                        for span in qualifier.evidence
                                    ),
                                    character_spans=tuple(
                                        (span.span.start, span.span.end)
                                        for span in qualifier.evidence
                                    ),
                                ),
                            )
                            for qualifier in candidate.mention_qualifiers
                        )
                    ),
                )
            )
    counts["qa_pairs"] = sum(len(predicate.pairs) for predicate in predicates)
    return ConsolidationResult(
        corpus=EvaluationCorpus(predicates=tuple(predicates)),
        rule=rule,
        counts=counts,
    )


def _eventivity(candidate: PredicateCandidate) -> bool:
    if candidate.predicate_type == "verbal":
        return True
    if not candidate.eventivity_judgments:
        raise DatasetFormatError(
            f"nominal candidate {candidate.candidate_id} has no eventivity judgment"
        )
    positive = sum(
        judgment.is_eventive for judgment in candidate.eventivity_judgments
    )
    negative = len(candidate.eventivity_judgments) - positive
    if positive == negative:
        raise DatasetFormatError(
            f"nominal candidate {candidate.candidate_id} has tied eventivity votes"
        )
    return positive > negative


def _consolidate_question(
    question: QASRLQuestion,
) -> tuple[list[EvaluationQAPair], dict[str, int]]:
    valid = tuple(judgment for judgment in question.judgments if judgment.is_valid)
    local_counts = {
        "questions_without_valid_judgment": 0,
        "valid_judgments_without_answers": 0,
        "duplicate_answer_alternatives": 0,
    }
    if not valid:
        local_counts["questions_without_valid_judgment"] = 1
        return [], local_counts

    signature_to_pair: dict[tuple[tuple[int, int, int, int], ...], EvaluationQAPair] = {}
    evaluation_question = EvaluationQuestion(
        surface_form=question.surface_form,
        wh=question.slots.wh,
        aux=question.slots.aux,
        subj=question.slots.subj,
        verb=question.slots.verb,
        obj=question.slots.obj,
        prep=question.slots.prep,
        obj2=question.slots.obj2,
        is_passive=question.is_passive,
        is_negated=question.is_negated,
    )
    for judgment in valid:
        if not judgment.answers:
            local_counts["valid_judgments_without_answers"] += 1
        for answer in judgment.answers:
            signature = tuple(
                (
                    span.token_start,
                    span.token_end,
                    span.span.start,
                    span.span.end,
                )
                for span in answer.spans
            )
            if signature in signature_to_pair:
                local_counts["duplicate_answer_alternatives"] += 1
                continue
            stable_key = repr(signature).encode("utf-8")
            digest = hashlib.sha256(stable_key).hexdigest()[:16]
            signature_to_pair[signature] = EvaluationQAPair(
                pair_id=f"{question.question_id}:answer:{digest}",
                role_id=question.question_id,
                question=evaluation_question,
                argument=EvaluationArgument(
                    token_spans=tuple(
                        (span.token_start, span.token_end) for span in answer.spans
                    ),
                    character_spans=tuple(
                        (span.span.start, span.span.end) for span in answer.spans
                    ),
                ),
                metadata={"consolidation_rule": VALID_JUDGMENT_UNION_V1},
            )
    return list(signature_to_pair.values()), local_counts
