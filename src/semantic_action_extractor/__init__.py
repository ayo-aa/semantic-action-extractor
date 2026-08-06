"""Public API for the semantic action extractor."""

from .baseline import BaselineConfig, RuleBasedExtractor
from .schema import ActionFrame, ExtractionResult, Qualifier, TextSpan

__all__ = [
    "ActionFrame",
    "BaselineConfig",
    "ExtractionResult",
    "Qualifier",
    "RuleBasedExtractor",
    "TextSpan",
]

__version__ = "0.1.0"
