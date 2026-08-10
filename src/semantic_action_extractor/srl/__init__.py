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
from .batching import (
    AlignedModelExample,
    OverlengthDrop,
    PaddedSRLBatch,
    PreparedSRLSplit,
    collate_srl_batch,
    prepare_srl_split,
)
from .dataset_io import (
    DATASET_SCHEMA_VERSION,
    DatasetFormatError,
    DatasetManifest,
    PreparedSRLDataset,
    compute_dataset_fingerprint,
    read_prepared_dataset,
    write_prepared_dataset,
)
from .checkpoint_bundle import (
    CheckpointLabelConfig,
    LoadedCheckpointBundle,
    ValidatedCheckpointBundle,
    load_checkpoint_bundle,
    save_checkpoint_bundle,
    validate_checkpoint_bundle,
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
from .experiment_config import (
    TrainingConfig,
    load_training_config,
    parse_training_config,
)
from .label_vocabulary import SRLLabelVocabulary, build_training_label_vocabulary
from .model import OptionalMLDependencyError, build_predicate_conditioned_bert
from .run_metadata import RunMetadata, require_exact_run_metadata
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
    "AlignedModelExample",
    "CheckpointLabelConfig",
    "DATASET_SCHEMA_VERSION",
    "DatasetFormatError",
    "DatasetManifest",
    "DatasetSplit",
    "LabeledSpan",
    "LoadedCheckpointBundle",
    "OptionalMLDependencyError",
    "OverlengthDrop",
    "PaddedSRLBatch",
    "PredicateDiagnostics",
    "PropBankAdapterError",
    "PropBankConversion",
    "PreparedSRLDataset",
    "PreparedSRLSplit",
    "PreparedWordLevelSRLExample",
    "RoleSpanMetrics",
    "RunMetadata",
    "SRLLabelVocabulary",
    "SpanMetrics",
    "SuppliedPredicateEvaluation",
    "TokenAccuracy",
    "TrainingConfig",
    "ValidatedCheckpointBundle",
    "WordLevelSRLExample",
    "allow_all_sources",
    "align_word_labels",
    "build_training_label_vocabulary",
    "build_predicate_conditioned_bert",
    "collapse_subword_predictions",
    "collate_srl_batch",
    "compute_dataset_fingerprint",
    "convert_propbank_record",
    "decode_bio",
    "deny_wsj_prefixed_document",
    "evaluate_supplied_predicate_srl",
    "load_checkpoint_bundle",
    "load_training_config",
    "repair_bio",
    "read_prepared_dataset",
    "parse_penn_tree",
    "parse_penn_trees",
    "parse_propbank_record",
    "parse_training_config",
    "prepare_srl_split",
    "require_exact_run_metadata",
    "save_checkpoint_bundle",
    "score_labeled_spans",
    "validate_checkpoint_bundle",
    "write_prepared_dataset",
]
