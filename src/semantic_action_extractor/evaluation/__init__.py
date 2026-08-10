"""Versioned evaluation contracts for semantic action extraction."""

from .bundle import (
    EVALUATION_BUNDLE_VERSION,
    EvaluationBundle,
    load_evaluation_bundle,
)
from .consolidation import (
    VALID_JUDGMENT_UNION_V1,
    ConsolidationResult,
    consolidate_annotations,
)
from .scorers import (
    MENTION_QUALIFIER_METRIC_V1,
    PRIMARY_END_TO_END_V1,
    QANOM_REFERENCE_V1,
    QASRL_GS_COMPATIBLE_V1,
    ScoreResult,
    score_corpora,
)
from .types import (
    EvaluationArgument,
    EvaluationCorpus,
    EvaluationMentionQualifier,
    EvaluationPredicate,
    EvaluationQAPair,
    EvaluationQuestion,
    PredicateKey,
)

__all__ = [
    "PRIMARY_END_TO_END_V1",
    "MENTION_QUALIFIER_METRIC_V1",
    "QANOM_REFERENCE_V1",
    "QASRL_GS_COMPATIBLE_V1",
    "VALID_JUDGMENT_UNION_V1",
    "ConsolidationResult",
    "EVALUATION_BUNDLE_VERSION",
    "EvaluationArgument",
    "EvaluationBundle",
    "EvaluationCorpus",
    "EvaluationMentionQualifier",
    "EvaluationPredicate",
    "EvaluationQAPair",
    "EvaluationQuestion",
    "PredicateKey",
    "ScoreResult",
    "consolidate_annotations",
    "load_evaluation_bundle",
    "score_corpora",
]
