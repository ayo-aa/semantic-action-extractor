"""Predicate-conditioned semantic role labeling primitives.

The internal SRL contract is deliberately narrower than the product schema:
given sentence words and the index of one supplied predicate, predict one
PropBank-style BIO label per word.  Predicate discovery is a separate task.
"""

from .alignment import AlignedSRLExample, align_word_labels
from .bio import LabeledSpan, decode_bio, repair_bio
from .evaluation import SpanMetrics, score_labeled_spans
from .example import DatasetSplit, PreparedWordLevelSRLExample, WordLevelSRLExample
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
    "DatasetSplit",
    "LabeledSpan",
    "OptionalMLDependencyError",
    "PropBankAdapterError",
    "PropBankConversion",
    "PreparedWordLevelSRLExample",
    "SpanMetrics",
    "WordLevelSRLExample",
    "allow_all_sources",
    "align_word_labels",
    "build_predicate_conditioned_bert",
    "convert_propbank_record",
    "decode_bio",
    "deny_wsj_prefixed_document",
    "repair_bio",
    "parse_penn_tree",
    "parse_penn_trees",
    "parse_propbank_record",
    "score_labeled_spans",
]
