"""Reference-compatible and corrected end-to-end scorer modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .matching import (
    Match,
    exact_character_match,
    exact_token_match,
    maximum_cardinality_weight_matching,
    qanom_reference_matching,
    thresholded_iou_edge,
)
from .question_equivalence import (
    EXACT_SLOTS_V1,
    QANOM_COARSE_ROLE_V1,
    QASRL_GS_FIVE_FIELD_V1,
    questions_equivalent,
)
from .types import (
    EvaluationArgument,
    EvaluationCorpus,
    EvaluationMentionQualifier,
    EvaluationPredicate,
    EvaluationQAPair,
    PredicateKey,
)


PRIMARY_END_TO_END_V1 = "primary-end-to-end-v1"
QASRL_GS_COMPATIBLE_V1 = "qasrl-gs-compatible-v1"
QANOM_REFERENCE_V1 = "qanom-reference-v1"
QASRL_GS_REFERENCE_REVISION = "f7c64ae9b6fe48ff3910c3e59850a12ec278bf83"
QANOM_REFERENCE_REVISION = "2bce70e8a39b40157ba97f38e1a8ae7619b30162"
PREDICATE_IDENTITY_V1 = "source-id+token-range+predicate-type-v1"
MENTION_QUALIFIER_METRIC_V1 = "mention-qualifier-label-and-exact-evidence-v1"

SCORER_MODES = {
    PRIMARY_END_TO_END_V1,
    QASRL_GS_COMPATIBLE_V1,
    QANOM_REFERENCE_V1,
}


@dataclass(frozen=True, slots=True)
class Counts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def __post_init__(self) -> None:
        if min(self.true_positive, self.false_positive, self.false_negative) < 0:
            raise ValueError("metric counts cannot be negative")

    @property
    def precision(self) -> float:
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 1.0

    @property
    def recall(self) -> float:
        expected = self.true_positive + self.false_negative
        return self.true_positive / expected if expected else 1.0

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0.0:
            return 0.0
        return 2.0 * self.precision * self.recall / (
            self.precision + self.recall
        )

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.true_positive + other.true_positive,
            self.false_positive + other.false_positive,
            self.false_negative + other.false_negative,
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class BinaryCounts:
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def retrieval(self) -> Counts:
        return Counts(self.true_positive, self.false_positive, self.false_negative)

    @property
    def accuracy(self) -> float:
        total = (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )
        return (
            (self.true_positive + self.true_negative) / total
            if total
            else 1.0
        )

    def __add__(self, other: "BinaryCounts") -> "BinaryCounts":
        return BinaryCounts(
            self.true_positive + other.true_positive,
            self.true_negative + other.true_negative,
            self.false_positive + other.false_positive,
            self.false_negative + other.false_negative,
        )

    def to_dict(self) -> dict[str, int | float]:
        payload = self.retrieval.to_dict()
        payload.update(
            {
                "true_negative": self.true_negative,
                "accuracy": self.accuracy,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class ScoreResult:
    scorer: str
    settings: dict[str, Any]
    candidate_detection: Counts
    eventivity: BinaryCounts
    unlabeled_arguments: Counts
    labeled_arguments: Counts
    exact_token_arguments: Counts
    exact_character_arguments: Counts
    reference_roles: Counts | None
    evaluated_predicates: int
    shared_eventivity_candidates: int
    skipped_eventivity_mismatches: int
    lemma_mismatches: int
    mention_qualifier_labels: Counts | None = None
    mention_qualifier_exact_evidence: Counts | None = None
    gold_qualifier_annotated_predicates: int = 0
    predicted_qualifier_annotated_predicates: int = 0
    shared_qualifier_annotated_predicates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorer": self.scorer,
            "settings": dict(self.settings),
            "evaluated_predicates": self.evaluated_predicates,
            "shared_eventivity_candidates": self.shared_eventivity_candidates,
            "skipped_eventivity_mismatches": self.skipped_eventivity_mismatches,
            "lemma_mismatches": self.lemma_mismatches,
            "candidate_detection": self.candidate_detection.to_dict(),
            "eventivity": self.eventivity.to_dict(),
            "unlabeled_arguments": self.unlabeled_arguments.to_dict(),
            "labeled_arguments": self.labeled_arguments.to_dict(),
            "exact_token_arguments": self.exact_token_arguments.to_dict(),
            "exact_character_arguments": self.exact_character_arguments.to_dict(),
            "reference_roles": (
                None if self.reference_roles is None else self.reference_roles.to_dict()
            ),
            "mention_qualifier_labels": (
                None
                if self.mention_qualifier_labels is None
                else self.mention_qualifier_labels.to_dict()
            ),
            "mention_qualifier_exact_evidence": (
                None
                if self.mention_qualifier_exact_evidence is None
                else self.mention_qualifier_exact_evidence.to_dict()
            ),
            "gold_qualifier_annotated_predicates": (
                self.gold_qualifier_annotated_predicates
            ),
            "predicted_qualifier_annotated_predicates": (
                self.predicted_qualifier_annotated_predicates
            ),
            "shared_qualifier_annotated_predicates": (
                self.shared_qualifier_annotated_predicates
            ),
        }


@dataclass(frozen=True, slots=True)
class _Mode:
    name: str
    predicate_scope: str
    threshold: float
    inclusive: bool
    matcher: str
    question_equivalence: str
    label_alignment: str
    skip_eventivity_mismatch: bool
    reference_revision: str | None
    reference_role_metric: str | None


_MODES = {
    PRIMARY_END_TO_END_V1: _Mode(
        name=PRIMARY_END_TO_END_V1,
        predicate_scope="union",
        threshold=0.5,
        inclusive=True,
        matcher="maximum-cardinality-then-maximum-iou",
        question_equivalence=EXACT_SLOTS_V1,
        label_alignment="joint",
        skip_eventivity_mismatch=False,
        reference_revision=None,
        reference_role_metric=None,
    ),
    QASRL_GS_COMPATIBLE_V1: _Mode(
        name=QASRL_GS_COMPATIBLE_V1,
        predicate_scope="gold",
        threshold=0.5,
        inclusive=True,
        matcher="maximum-cardinality-then-maximum-iou",
        question_equivalence=QASRL_GS_FIVE_FIELD_V1,
        label_alignment="after-unlabeled",
        skip_eventivity_mismatch=False,
        reference_revision=QASRL_GS_REFERENCE_REVISION,
        reference_role_metric=None,
    ),
    QANOM_REFERENCE_V1: _Mode(
        name=QANOM_REFERENCE_V1,
        predicate_scope="intersection",
        threshold=0.3,
        inclusive=False,
        matcher="greedy-descending-iou-by-span-value",
        question_equivalence=QANOM_COARSE_ROLE_V1,
        label_alignment="after-unlabeled",
        skip_eventivity_mismatch=True,
        reference_revision=QANOM_REFERENCE_REVISION,
        reference_role_metric="qanom-role-alignment-v1",
    ),
}


def score_corpora(
    gold: EvaluationCorpus,
    predicted: EvaluationCorpus,
    *,
    mode: str = PRIMARY_END_TO_END_V1,
    predicate_source: str,
    consolidation_rule: str,
) -> ScoreResult:
    try:
        contract = _MODES[mode]
    except KeyError as error:
        choices = ", ".join(sorted(_MODES))
        raise ValueError(f"unknown scorer mode {mode!r}; choose from: {choices}") from error
    for value, label in (
        (predicate_source, "predicate_source"),
        (consolidation_rule, "consolidation_rule"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} cannot be empty")

    gold_by_key = gold.by_key()
    predicted_by_key = predicted.by_key()
    keys = _predicate_keys(contract, gold_by_key, predicted_by_key)
    gold_keys = gold_by_key.keys()
    predicted_keys = predicted_by_key.keys()
    candidate_counts = Counts(
        true_positive=len(gold_keys & predicted_keys),
        false_positive=len(predicted_keys - gold_keys),
        false_negative=len(gold_keys - predicted_keys),
    )

    eventivity_counts = BinaryCounts()
    unlabeled = Counts()
    labeled = Counts()
    exact_token = Counts()
    exact_character = Counts()
    reference_roles = Counts() if contract.reference_role_metric else None
    qualifier_labels = Counts() if contract.name == PRIMARY_END_TO_END_V1 else None
    qualifier_evidence = Counts() if contract.name == PRIMARY_END_TO_END_V1 else None
    gold_qualifier_annotated = 0
    predicted_qualifier_annotated = 0
    shared_qualifier_annotated = 0
    skipped_eventivity = 0
    lemma_mismatches = 0

    for key in keys:
        gold_predicate = gold_by_key.get(key)
        predicted_predicate = predicted_by_key.get(key)
        if gold_predicate is not None and predicted_predicate is not None:
            eventivity_counts += _binary_decision(
                predicted_predicate.is_eventive,
                gold_predicate.is_eventive,
            )
            if (
                gold_predicate.lemma is not None
                and predicted_predicate.lemma is not None
                and gold_predicate.lemma.casefold()
                != predicted_predicate.lemma.casefold()
            ):
                lemma_mismatches += 1

        if (
            contract.skip_eventivity_mismatch
            and gold_predicate is not None
            and predicted_predicate is not None
            and gold_predicate.is_eventive != predicted_predicate.is_eventive
        ):
            skipped_eventivity += 1
            continue

        gold_pairs = (
            gold_predicate.pairs
            if gold_predicate is not None and gold_predicate.is_eventive
            else ()
        )
        predicted_pairs = (
            predicted_predicate.pairs
            if predicted_predicate is not None and predicted_predicate.is_eventive
            else ()
        )
        local = _score_pairs(gold_pairs, predicted_pairs, contract)
        unlabeled += local[0]
        labeled += local[1]
        exact_token += local[2]
        exact_character += local[3]
        if reference_roles is not None:
            reference_roles += local[4] or Counts()
        if qualifier_labels is not None and qualifier_evidence is not None:
            qualifier_local = _score_mention_qualifiers(
                gold_predicate,
                predicted_predicate,
            )
            qualifier_labels += qualifier_local[0]
            qualifier_evidence += qualifier_local[1]
            gold_qualifier_annotated += qualifier_local[2]
            predicted_qualifier_annotated += qualifier_local[3]
            shared_qualifier_annotated += qualifier_local[4]

    return ScoreResult(
        scorer=contract.name,
        settings={
            "scorer_version": contract.name,
            "reference_revision": contract.reference_revision,
            "predicate_source": predicate_source,
            "gold_consolidation_rule": consolidation_rule,
            "predicate_identity": PREDICATE_IDENTITY_V1,
            "predicate_scope": contract.predicate_scope,
            "token_iou_threshold": contract.threshold,
            "threshold_inclusive": contract.inclusive,
            "matching": contract.matcher,
            "question_equivalence": contract.question_equivalence,
            "label_alignment": contract.label_alignment,
            "reference_role_metric": contract.reference_role_metric,
            "mention_qualifier_metric": (
                MENTION_QUALIFIER_METRIC_V1
                if contract.name == PRIMARY_END_TO_END_V1
                else None
            ),
            "mention_qualifier_scope": (
                "union-of-predicate-keys-with-unavailable-gold-labels-skipped"
                if contract.name == PRIMARY_END_TO_END_V1
                else None
            ),
            "zero_denominator_value": 1.0,
        },
        candidate_detection=candidate_counts,
        eventivity=eventivity_counts,
        unlabeled_arguments=unlabeled,
        labeled_arguments=labeled,
        exact_token_arguments=exact_token,
        exact_character_arguments=exact_character,
        reference_roles=reference_roles,
        mention_qualifier_labels=qualifier_labels,
        mention_qualifier_exact_evidence=qualifier_evidence,
        gold_qualifier_annotated_predicates=gold_qualifier_annotated,
        predicted_qualifier_annotated_predicates=predicted_qualifier_annotated,
        shared_qualifier_annotated_predicates=shared_qualifier_annotated,
        evaluated_predicates=len(keys),
        shared_eventivity_candidates=len(gold_keys & predicted_keys),
        skipped_eventivity_mismatches=skipped_eventivity,
        lemma_mismatches=lemma_mismatches,
    )


def _score_mention_qualifiers(
    gold: EvaluationPredicate | None,
    predicted: EvaluationPredicate | None,
) -> tuple[Counts, Counts, int, int, int]:
    """Score qualifier kinds and their exact grouped evidence separately.

    ``None`` means that a predicate was not annotated or assessed for mention
    framing. An empty tuple means it was assessed and no supported qualifier
    was present.
    """

    gold_annotated = int(
        gold is not None
        and gold.is_eventive
        and gold.mention_qualifiers is not None
    )
    predicted_annotated = int(
        predicted is not None
        and predicted.is_eventive
        and predicted.mention_qualifiers is not None
    )
    if gold is not None and not gold_annotated:
        return Counts(), Counts(), 0, predicted_annotated, 0

    gold_items = (
        ()
        if gold is None or gold.mention_qualifiers is None
        else gold.mention_qualifiers
    )
    predicted_items = (
        ()
        if predicted is None or predicted.mention_qualifiers is None
        else predicted.mention_qualifiers
    )
    gold_labels = {item.kind for item in gold_items}
    predicted_labels = {item.kind for item in predicted_items}
    gold_evidence = {_qualifier_signature(item) for item in gold_items}
    predicted_evidence = {_qualifier_signature(item) for item in predicted_items}
    return (
        Counts(
            true_positive=len(gold_labels & predicted_labels),
            false_positive=len(predicted_labels - gold_labels),
            false_negative=len(gold_labels - predicted_labels),
        ),
        Counts(
            true_positive=len(gold_evidence & predicted_evidence),
            false_positive=len(predicted_evidence - gold_evidence),
            false_negative=len(gold_evidence - predicted_evidence),
        ),
        gold_annotated,
        predicted_annotated,
        int(gold_annotated and predicted_annotated),
    )


def _qualifier_signature(
    qualifier: EvaluationMentionQualifier,
) -> tuple[
    str,
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...] | None,
]:
    return (
        qualifier.kind,
        qualifier.evidence.token_spans,
        qualifier.evidence.character_spans,
    )


def _predicate_keys(
    mode: _Mode,
    gold: dict[PredicateKey, EvaluationPredicate],
    predicted: dict[PredicateKey, EvaluationPredicate],
) -> tuple[PredicateKey, ...]:
    if mode.predicate_scope == "union":
        keys = gold.keys() | predicted.keys()
    elif mode.predicate_scope == "gold":
        keys = gold.keys()
    elif mode.predicate_scope == "intersection":
        keys = gold.keys() & predicted.keys()
    else:
        raise AssertionError(f"unsupported predicate scope: {mode.predicate_scope}")
    return tuple(sorted(keys))


def _score_pairs(
    gold: tuple[EvaluationQAPair, ...],
    predicted: tuple[EvaluationQAPair, ...],
    mode: _Mode,
) -> tuple[Counts, Counts, Counts, Counts, Counts | None]:
    gold_arguments = tuple(pair.argument for pair in gold)
    predicted_arguments = tuple(pair.argument for pair in predicted)
    overlap_edge = thresholded_iou_edge(
        predicted_arguments,
        gold_arguments,
        threshold=mode.threshold,
        inclusive=mode.inclusive,
    )
    if mode.matcher == "greedy-descending-iou-by-span-value":
        overlap_matches = qanom_reference_matching(
            predicted_arguments,
            gold_arguments,
            overlap_edge,
        )
    else:
        overlap_matches = maximum_cardinality_weight_matching(
            len(predicted),
            len(gold),
            overlap_edge,
        )
    unlabeled = _counts_from_matches(len(predicted), len(gold), overlap_matches)

    if mode.label_alignment == "joint":
        labeled_edge = thresholded_iou_edge(
            predicted_arguments,
            gold_arguments,
            threshold=mode.threshold,
            inclusive=mode.inclusive,
            allowed=lambda predicted_index, gold_index: questions_equivalent(
                predicted[predicted_index].question,
                gold[gold_index].question,
                mode=mode.question_equivalence,
            ),
        )
        labeled_matches = maximum_cardinality_weight_matching(
            len(predicted),
            len(gold),
            labeled_edge,
        )
    else:
        labeled_matches = tuple(
            match
            for match in overlap_matches
            if questions_equivalent(
                predicted[match.predicted_index].question,
                gold[match.gold_index].question,
                mode=mode.question_equivalence,
            )
        )
    labeled = _counts_from_matches(len(predicted), len(gold), labeled_matches)

    exact_token_matches = maximum_cardinality_weight_matching(
        len(predicted),
        len(gold),
        lambda predicted_index, gold_index: (
            1.0
            if exact_token_match(
                predicted[predicted_index].argument,
                gold[gold_index].argument,
            )
            else None
        ),
    )
    exact_character_matches = maximum_cardinality_weight_matching(
        len(predicted),
        len(gold),
        lambda predicted_index, gold_index: (
            1.0
            if exact_character_match(
                predicted[predicted_index].argument,
                gold[gold_index].argument,
            )
            else None
        ),
    )
    return (
        unlabeled,
        labeled,
        _counts_from_matches(len(predicted), len(gold), exact_token_matches),
        _counts_from_matches(
            len(predicted), len(gold), exact_character_matches
        ),
        (
            _qanom_reference_role_counts(gold, predicted, overlap_matches)
            if mode.reference_role_metric == "qanom-role-alignment-v1"
            else None
        ),
    )


def _qanom_reference_role_counts(
    gold: tuple[EvaluationQAPair, ...],
    predicted: tuple[EvaluationQAPair, ...],
    matches: tuple[Match, ...],
) -> Counts:
    """Reproduce QANom's role-alignment count from matched arguments."""

    gold_roles = {_role_key(pair) for pair in gold}
    predicted_roles = {_role_key(pair) for pair in predicted}
    gold_to_predicted = {role: set() for role in gold_roles}
    predicted_to_gold = {role: set() for role in predicted_roles}
    for match in matches:
        predicted_role = _role_key(predicted[match.predicted_index])
        gold_role = _role_key(gold[match.gold_index])
        predicted_to_gold[predicted_role].add(gold_role)
        gold_to_predicted[gold_role].add(predicted_role)

    true_positive = sum(
        len(aligned_roles) == 1 for aligned_roles in gold_to_predicted.values()
    )
    return Counts(
        true_positive=true_positive,
        false_positive=sum(
            len(aligned_roles) != 1
            for aligned_roles in predicted_to_gold.values()
        ),
        false_negative=len(gold_roles) - true_positive,
    )


def _role_key(pair: EvaluationQAPair) -> object:
    return pair.role_id if pair.role_id is not None else pair.question


def _counts_from_matches(
    predicted_count: int,
    gold_count: int,
    matches: tuple[Match, ...],
) -> Counts:
    true_positive = len(matches)
    return Counts(
        true_positive=true_positive,
        false_positive=predicted_count - true_positive,
        false_negative=gold_count - true_positive,
    )


def _binary_decision(predicted: bool, gold: bool) -> BinaryCounts:
    return BinaryCounts(
        true_positive=int(predicted and gold),
        true_negative=int(not predicted and not gold),
        false_positive=int(predicted and not gold),
        false_negative=int(not predicted and gold),
    )
