"""Exact labeled-span evaluation for predicate-conditioned SRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Sequence

from .bio import decode_bio, repair_bio


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
