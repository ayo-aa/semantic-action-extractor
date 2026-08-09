"""Deterministic one-to-one span matching without third-party dependencies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

from .types import EvaluationArgument


@dataclass(frozen=True, slots=True)
class Match:
    predicted_index: int
    gold_index: int
    score: float


EdgeScore = Callable[[int, int], float | None]


def token_iou(left: EvaluationArgument, right: EvaluationArgument) -> float:
    left_tokens = _covered_indexes(left.token_spans)
    right_tokens = _covered_indexes(right.token_spans)
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union


def character_iou(left: EvaluationArgument, right: EvaluationArgument) -> float:
    if left.character_spans is None or right.character_spans is None:
        return 0.0
    left_characters = _covered_indexes(left.character_spans)
    right_characters = _covered_indexes(right.character_spans)
    intersection = len(left_characters & right_characters)
    union = len(left_characters | right_characters)
    return intersection / union


def exact_token_match(left: EvaluationArgument, right: EvaluationArgument) -> bool:
    return left.token_spans == right.token_spans


def exact_character_match(left: EvaluationArgument, right: EvaluationArgument) -> bool:
    return (
        left.character_spans is not None
        and right.character_spans is not None
        and left.character_spans == right.character_spans
    )


def maximum_cardinality_weight_matching(
    predicted_count: int,
    gold_count: int,
    edge_score: EdgeScore,
) -> tuple[Match, ...]:
    """Maximize eligible pair count, then total edge score.

    A square assignment matrix receives a cardinality bonus larger than the
    largest possible difference in total edge weight. The Hungarian algorithm
    then gives maximum cardinality first and maximum total score second.
    """

    if predicted_count < 0 or gold_count < 0:
        raise ValueError("matching counts cannot be negative")
    if not predicted_count or not gold_count:
        return ()

    size = max(predicted_count, gold_count)
    cardinality_bonus = float(size + 1)
    weights = [[0.0 for _ in range(size)] for _ in range(size)]
    eligible: dict[tuple[int, int], float] = {}
    for predicted_index in range(predicted_count):
        for gold_index in range(gold_count):
            score = edge_score(predicted_index, gold_index)
            if score is None:
                continue
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError("matching edge scores must be between 0 and 1")
            eligible[(predicted_index, gold_index)] = score
            weights[predicted_index][gold_index] = cardinality_bonus + score

    assignment = _hungarian_maximize(weights)
    matches = [
        Match(predicted_index=row, gold_index=column, score=eligible[(row, column)])
        for row, column in enumerate(assignment)
        if row < predicted_count
        and column < gold_count
        and (row, column) in eligible
    ]
    return tuple(sorted(matches, key=lambda item: item.predicted_index))


def greedy_weight_matching(
    predicted_count: int,
    gold_count: int,
    edge_score: EdgeScore,
) -> tuple[Match, ...]:
    """Reproduce QANom's descending-IoU greedy alignment deterministically."""

    candidates: list[Match] = []
    for predicted_index in range(predicted_count):
        for gold_index in range(gold_count):
            score = edge_score(predicted_index, gold_index)
            if score is None:
                continue
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError("matching edge scores must be between 0 and 1")
            candidates.append(Match(predicted_index, gold_index, score))
    candidates.sort(
        key=lambda item: (-item.score, item.predicted_index, item.gold_index)
    )

    used_predicted: set[int] = set()
    used_gold: set[int] = set()
    selected: list[Match] = []
    for candidate in candidates:
        if candidate.predicted_index in used_predicted:
            continue
        if candidate.gold_index in used_gold:
            continue
        used_predicted.add(candidate.predicted_index)
        used_gold.add(candidate.gold_index)
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: item.predicted_index))


def qanom_reference_matching(
    predicted: Sequence[EvaluationArgument],
    gold: Sequence[EvaluationArgument],
    edge_score: EdgeScore,
) -> tuple[Match, ...]:
    """Mirror QANom's greedy matching keyed by hashable span values.

    The reference implementation stores its alignment in a dictionary from a
    predicted span tuple to a gold span tuple. Repeated occurrences of the same
    token range can therefore contribute at most one match, even though every
    occurrence remains in the false-positive and false-negative denominators.
    """

    candidates: list[Match] = []
    for predicted_index in range(len(predicted)):
        for gold_index in range(len(gold)):
            score = edge_score(predicted_index, gold_index)
            if score is None:
                continue
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError("matching edge scores must be between 0 and 1")
            candidates.append(Match(predicted_index, gold_index, score))
    candidates.sort(
        key=lambda item: (-item.score, item.predicted_index, item.gold_index)
    )

    used_predicted_values: set[tuple[tuple[int, int], ...]] = set()
    used_gold_values: set[tuple[tuple[int, int], ...]] = set()
    selected: list[Match] = []
    for candidate in candidates:
        predicted_value = predicted[candidate.predicted_index].token_spans
        gold_value = gold[candidate.gold_index].token_spans
        if predicted_value in used_predicted_values:
            continue
        if gold_value in used_gold_values:
            continue
        used_predicted_values.add(predicted_value)
        used_gold_values.add(gold_value)
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: item.predicted_index))


def thresholded_iou_edge(
    predicted: Sequence[EvaluationArgument],
    gold: Sequence[EvaluationArgument],
    *,
    threshold: float,
    inclusive: bool,
    allowed: Callable[[int, int], bool] | None = None,
) -> EdgeScore:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1")

    def score(predicted_index: int, gold_index: int) -> float | None:
        if allowed is not None and not allowed(predicted_index, gold_index):
            return None
        overlap = token_iou(predicted[predicted_index], gold[gold_index])
        passes = overlap >= threshold if inclusive else overlap > threshold
        return overlap if passes else None

    return score


def _covered_indexes(ranges: Sequence[tuple[int, int]]) -> frozenset[int]:
    return frozenset(
        index
        for start, end in ranges
        for index in range(start, end)
    )


def _hungarian_maximize(weights: Sequence[Sequence[float]]) -> list[int]:
    """Return the selected column for each row of a square weight matrix."""

    size = len(weights)
    if any(len(row) != size for row in weights):
        raise ValueError("Hungarian assignment requires a square matrix")
    if not size:
        return []

    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    potentials_rows = [0.0] * (size + 1)
    potentials_columns = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    epsilon = 1e-12

    for row in range(1, size + 1):
        matched_row[0] = row
        column = 0
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    costs[current_row - 1][candidate_column - 1]
                    - potentials_rows[current_row]
                    - potentials_columns[candidate_column]
                )
                if reduced < minimum[candidate_column] - epsilon:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if (
                    minimum[candidate_column] < delta - epsilon
                    or (
                        abs(minimum[candidate_column] - delta) <= epsilon
                        and (next_column == 0 or candidate_column < next_column)
                    )
                ):
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    potentials_rows[matched_row[candidate_column]] += delta
                    potentials_columns[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = [-1] * size
    for column in range(1, size + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    return assignment
