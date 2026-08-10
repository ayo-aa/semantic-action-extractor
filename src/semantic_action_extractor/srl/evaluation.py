"""Exact labeled-span evaluation for predicate-conditioned SRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Sequence

from .bio import decode_bio, repair_bio


_PREDICATE_LABELS = frozenset({"V", "C-V"})


@dataclass(frozen=True, slots=True)
class SpanMetrics:
    """Micro-averaged exact labeled-span counts and metrics."""

    true_positives: int
    predicted: int
    gold: int
    precision: float
    recall: float
    f1: float
    repaired_prediction_tags: int


@dataclass(frozen=True, slots=True)
class RoleSpanMetrics:
    """Exact labeled-span counts and metrics for one argument role."""

    label: str
    true_positives: int
    predicted: int
    gold: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class PredicateDiagnostics:
    """Diagnostics excluded from the primary supplied-predicate role score.

    Predicate labels are not argument roles, but prediction mistakes at or
    away from the supplied predicate still matter.  Spurious predicate words
    count predicted ``V`` or ``C-V`` word positions outside the gold predicate
    pieces.  Argument overlaps count predicted argument spans that include at
    least one gold predicate word; those spans remain false positives in the
    primary argument metric.
    """

    examples: int
    correct_anchors: int
    missing_anchors: int
    spurious_predicate_words: int
    argument_spans_overlapping_predicate: int


@dataclass(frozen=True, slots=True)
class TokenAccuracy:
    """Exact repaired-prediction accuracy over all word-level BIO tags."""

    correct: int
    total: int
    accuracy: float


@dataclass(frozen=True, slots=True)
class SuppliedPredicateEvaluation:
    """Complete word-level evaluation for supplied-predicate SRL."""

    arguments: SpanMetrics
    per_role: tuple[RoleSpanMetrics, ...]
    predicate: PredicateDiagnostics
    token_accuracy: TokenAccuracy


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def score_labeled_spans(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
    *,
    excluded_labels: Collection[str] = ("V", "C-V"),
    repair_predictions: bool = True,
) -> SpanMetrics:
    """Score exact role label and boundary matches across examples.

    The score is micro-averaged over decoded spans.  Predicate spans (``V``)
    are excluded by default because the predicate index is supplied to this
    task rather than predicted. Gold BIO sequences are validated strictly.
    Invalid prediction transitions can be repaired deterministically, and the
    number of changed prediction tags is returned with the metrics.
    """

    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("gold and prediction example counts must match")

    excluded = set(excluded_labels)
    true_positives = 0
    predicted_count = 0
    gold_count = 0
    repaired_prediction_tags = 0

    for example_index, (gold_tags, predicted_tags) in enumerate(
        zip(gold_sequences, predicted_sequences, strict=True)
    ):
        if len(gold_tags) != len(predicted_tags):
            raise ValueError(
                f"gold and prediction lengths differ for example {example_index}"
            )
        gold_spans = {
            span
            for span in decode_bio(gold_tags, repair=False)
            if span.label not in excluded
        }
        normalized_predictions = tuple(predicted_tags)
        if repair_predictions:
            normalized_predictions = repair_bio(normalized_predictions)
            repaired_prediction_tags += sum(
                original != repaired
                for original, repaired in zip(
                    predicted_tags, normalized_predictions, strict=True
                )
            )
        predicted_spans = {
            span
            for span in decode_bio(normalized_predictions, repair=False)
            if span.label not in excluded
        }
        true_positives += len(gold_spans & predicted_spans)
        predicted_count += len(predicted_spans)
        gold_count += len(gold_spans)

    precision = _safe_ratio(true_positives, predicted_count)
    recall = _safe_ratio(true_positives, gold_count)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return SpanMetrics(
        true_positives=true_positives,
        predicted=predicted_count,
        gold=gold_count,
        precision=precision,
        recall=recall,
        f1=f1,
        repaired_prediction_tags=repaired_prediction_tags,
    )


def evaluate_supplied_predicate_srl(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
    predicate_indexes: Sequence[int],
    *,
    repair_predictions: bool = True,
) -> SuppliedPredicateEvaluation:
    """Evaluate word-level arguments while auditing supplied predicates.

    Gold BIO is always decoded strictly.  Predictions use the same optional
    orphan-``I`` repair policy as :func:`score_labeled_spans`.  The primary
    metric micro-averages exact labeled argument spans and excludes decoded
    ``V`` and ``C-V`` spans globally.  Predicted argument spans that overlap a
    gold predicate word are not discarded and therefore count as false
    positives.  Conversely, predicting ``V`` over a gold argument does not
    create a predicted argument and leaves the gold argument as a false
    negative.

    Token accuracy compares every repaired word-level prediction with its gold
    BIO tag, including predicate positions.  Predicate diagnostics are kept
    separate from the primary argument score.
    """

    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("gold and prediction example counts must match")
    if len(predicate_indexes) != len(gold_sequences):
        raise ValueError("predicate index and gold example counts must match")

    true_positives = 0
    predicted_count = 0
    gold_count = 0
    repaired_prediction_tags = 0
    token_correct = 0
    token_total = 0
    correct_anchors = 0
    spurious_predicate_words = 0
    argument_overlaps = 0
    role_counts: dict[str, list[int]] = {}

    for example_index, (gold_tags, predicted_tags, predicate_index) in enumerate(
        zip(
            gold_sequences,
            predicted_sequences,
            predicate_indexes,
            strict=True,
        )
    ):
        if len(gold_tags) != len(predicted_tags):
            raise ValueError(
                f"gold and prediction lengths differ for example {example_index}"
            )
        if type(predicate_index) is not int:
            raise TypeError(
                f"predicate index for example {example_index} must be an integer"
            )
        if predicate_index < 0 or predicate_index >= len(gold_tags):
            raise IndexError(
                f"predicate index is outside example {example_index}"
            )

        gold = tuple(gold_tags)
        gold_spans_all = decode_bio(gold, repair=False)
        if gold[predicate_index] != "B-V":
            raise ValueError(
                f"gold example {example_index} must have B-V at its supplied "
                "predicate index"
            )
        if gold.count("B-V") != 1:
            raise ValueError(
                f"gold example {example_index} must contain exactly one B-V anchor"
            )

        normalized_predictions = tuple(predicted_tags)
        if repair_predictions:
            normalized_predictions = repair_bio(normalized_predictions)
            repaired_prediction_tags += sum(
                original != repaired
                for original, repaired in zip(
                    predicted_tags, normalized_predictions, strict=True
                )
            )
        predicted_spans_all = decode_bio(
            normalized_predictions, repair=False
        )

        gold_arguments = {
            span
            for span in gold_spans_all
            if span.label not in _PREDICATE_LABELS
        }
        predicted_arguments = {
            span
            for span in predicted_spans_all
            if span.label not in _PREDICATE_LABELS
        }
        matches = gold_arguments & predicted_arguments
        true_positives += len(matches)
        predicted_count += len(predicted_arguments)
        gold_count += len(gold_arguments)

        for label in sorted(
            {span.label for span in gold_arguments | predicted_arguments}
        ):
            counts = role_counts.setdefault(label, [0, 0, 0])
            counts[0] += sum(span.label == label for span in matches)
            counts[1] += sum(span.label == label for span in predicted_arguments)
            counts[2] += sum(span.label == label for span in gold_arguments)

        gold_predicate_words = {
            word_index
            for span in gold_spans_all
            if span.label in _PREDICATE_LABELS
            for word_index in range(span.start, span.end)
        }
        if normalized_predictions[predicate_index] == "B-V":
            correct_anchors += 1
        spurious_predicate_words += sum(
            1
            for span in predicted_spans_all
            if span.label in _PREDICATE_LABELS
            for word_index in range(span.start, span.end)
            if word_index not in gold_predicate_words
        )
        argument_overlaps += sum(
            bool(gold_predicate_words.intersection(range(span.start, span.end)))
            for span in predicted_arguments
        )

        token_correct += sum(
            gold_tag == predicted_tag
            for gold_tag, predicted_tag in zip(
                gold, normalized_predictions, strict=True
            )
        )
        token_total += len(gold)

    precision = _safe_ratio(true_positives, predicted_count)
    recall = _safe_ratio(true_positives, gold_count)
    arguments = SpanMetrics(
        true_positives=true_positives,
        predicted=predicted_count,
        gold=gold_count,
        precision=precision,
        recall=recall,
        f1=_safe_ratio(2 * precision * recall, precision + recall),
        repaired_prediction_tags=repaired_prediction_tags,
    )
    per_role = tuple(
        RoleSpanMetrics(
            label=label,
            true_positives=counts[0],
            predicted=counts[1],
            gold=counts[2],
            precision=(role_precision := _safe_ratio(counts[0], counts[1])),
            recall=(role_recall := _safe_ratio(counts[0], counts[2])),
            f1=_safe_ratio(
                2 * role_precision * role_recall,
                role_precision + role_recall,
            ),
        )
        for label, counts in sorted(role_counts.items())
    )
    example_count = len(gold_sequences)
    return SuppliedPredicateEvaluation(
        arguments=arguments,
        per_role=per_role,
        predicate=PredicateDiagnostics(
            examples=example_count,
            correct_anchors=correct_anchors,
            missing_anchors=example_count - correct_anchors,
            spurious_predicate_words=spurious_predicate_words,
            argument_spans_overlapping_predicate=argument_overlaps,
        ),
        token_accuracy=TokenAccuracy(
            correct=token_correct,
            total=token_total,
            accuracy=_safe_ratio(token_correct, token_total),
        ),
    )
