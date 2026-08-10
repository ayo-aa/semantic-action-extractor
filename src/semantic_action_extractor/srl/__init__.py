"""Predicate-conditioned semantic role labeling primitives.

The internal SRL contract is deliberately narrower than the product schema:
given sentence words and the index of one supplied predicate, predict one
PropBank-style BIO label per word.  Predicate discovery is a separate task.
"""

from .alignment import (
    AlignedSRLExample,
    align_word_labels,
    collapse_subword_predictions,
)
from .bio import LabeledSpan, decode_bio, repair_bio
from .dataset_io import (
    DATASET_SCHEMA_VERSION,
    DatasetFormatError,
    DatasetManifest,
    PreparedSRLDataset,
    compute_dataset_fingerprint,
    read_prepared_dataset,
    write_prepared_dataset,
)
from .evaluation import (
    PredicateDiagnostics,
    RoleSpanMetrics,
    SpanMetrics,
    SuppliedPredicateEvaluation,
    TokenAccuracy,
    evaluate_supplied_predicate_srl,
    score_labeled_spans,
)
from .example import DatasetSplit, PreparedWordLevelSRLExample, WordLevelSRLExample
from .label_vocabulary import SRLLabelVocabulary, build_training_label_vocabulary
from .model import OptionalMLDependencyError, build_predicate_conditioned_bert
from .propbank import (
    PropBankAdapterError,
    PropBankConversion,
    allow_all_sources,
    convert_propbank_record,
    deny_wsj_prefixed_document,
    parse_penn_tree,
    parse_penn_trees,
    parse_propbank_record,
)

__all__ = [
    "AlignedSRLExample",
    "DATASET_SCHEMA_VERSION",
    "DatasetFormatError",
    "DatasetManifest",
    "DatasetSplit",
    "LabeledSpan",
    "OptionalMLDependencyError",
    "PredicateDiagnostics",
    "PropBankAdapterError",
    "PropBankConversion",
    "PreparedSRLDataset",
    "PreparedWordLevelSRLExample",
    "RoleSpanMetrics",
    "SRLLabelVocabulary",
    "SpanMetrics",
    "SuppliedPredicateEvaluation",
    "TokenAccuracy",
    "WordLevelSRLExample",
    "allow_all_sources",
    "align_word_labels",
    "build_training_label_vocabulary",
    "build_predicate_conditioned_bert",
    "collapse_subword_predictions",
    "compute_dataset_fingerprint",
    "convert_propbank_record",
    "decode_bio",
    "deny_wsj_prefixed_document",
    "evaluate_supplied_predicate_srl",
    "repair_bio",
    "read_prepared_dataset",
    "parse_penn_tree",
    "parse_penn_trees",
    "parse_propbank_record",
    "score_labeled_spans",
    "write_prepared_dataset",
]
