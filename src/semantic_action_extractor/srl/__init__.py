"""Predicate-conditioned semantic role labeling primitives.

The internal SRL contract is deliberately narrower than the product schema:
given sentence words and the index of one supplied predicate, predict one
PropBank-style BIO label per word.  Predicate discovery is a separate task.
"""

from .alignment import AlignedSRLExample, align_word_labels
from .bio import LabeledSpan, decode_bio, repair_bio
from .evaluation import SpanMetrics, score_labeled_spans
from .model import OptionalMLDependencyError, build_predicate_conditioned_bert

__all__ = [
    "AlignedSRLExample",
    "LabeledSpan",
    "OptionalMLDependencyError",
    "SpanMetrics",
    "align_word_labels",
    "build_predicate_conditioned_bert",
    "decode_bio",
    "repair_bio",
    "score_labeled_spans",
]
