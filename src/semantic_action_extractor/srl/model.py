"""Optional predicate-conditioned BERT model for wordpiece token labels.

Importing this module does not import PyTorch or Transformers.  The optional
libraries are loaded only when :func:`build_predicate_conditioned_bert` is
called, keeping the rule-based package dependency-free.
"""

from __future__ import annotations

import importlib
from typing import Any


class OptionalMLDependencyError(ImportError):
    """Raised when construction is requested without optional ML libraries."""


class SRLModelOutput(dict[str, Any]):
    """Dictionary-compatible token logits and optional loss.

    Dictionary compatibility keeps the boundary usable by consumers that
    expect Hugging Face-style mapping outputs, while the properties preserve
    convenient attribute access in direct calls.
    """

    def __init__(self, *, logits: Any, loss: Any | None = None) -> None:
        if loss is None:
            super().__init__(logits=logits)
        else:
            super().__init__(loss=loss, logits=logits)

    @property
    def logits(self) -> Any:
        return self["logits"]

    @property
    def loss(self) -> Any | None:
        return self.get("loss")


def _load_optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise OptionalMLDependencyError(
            "predicate-conditioned BERT requires the optional torch and "
            "transformers packages"
        ) from error


def build_predicate_conditioned_bert(
    model_name: str,
    num_labels: int,
    *,
    model_revision: str | None = None,
    encoder: Any | None = None,
    torch_module: Any | None = None,
    transformers_module: Any | None = None,
) -> Any:
    """Build a fully fine-tuned BERT encoder with one linear token head.

    ``token_type_ids`` carries the first-subword predicate indicator produced
    by :func:`semantic_action_extractor.srl.alignment.align_word_labels`.
    Encoder parameters remain trainable, so optimizing ``model.parameters()``
    fine-tunes the full encoder and the linear classifier.  Label ``-100`` is
    ignored by the cross-entropy loss.

    ``model_revision`` pins the Hugging Face revision used to construct the
    encoder.  Experiment runners should always provide it; ``None`` remains
    supported for injected/offline callers and backwards compatibility.

    The module arguments support dependency injection for offline tests and
    specialized runtimes; normal callers should leave them unset.
    """

    if not isinstance(model_name, str):
        raise TypeError("model name must be a string")
    if not model_name.strip():
        raise ValueError("model name cannot be empty")
    model_name = model_name.strip()
    if type(num_labels) is not int:
        raise TypeError("num_labels must be an integer")
    if num_labels <= 0:
        raise ValueError("num_labels must be positive")
    if model_revision is not None:
        if not isinstance(model_revision, str):
            raise TypeError("model revision must be a string or None")
        if not model_revision.strip():
            raise ValueError("model revision cannot be empty")
        model_revision = model_revision.strip()

    torch = torch_module if torch_module is not None else _load_optional_module("torch")
    transformers = transformers_module
    if encoder is None:
        if transformers is None:
            transformers = _load_optional_module("transformers")
        load_kwargs = (
            {"revision": model_revision} if model_revision is not None else {}
        )
        encoder = transformers.AutoModel.from_pretrained(model_name, **load_kwargs)

    try:
        module_base = torch.nn.Module
        linear_type = torch.nn.Linear
        loss_type = torch.nn.CrossEntropyLoss
        hidden_size = int(encoder.config.hidden_size)
        type_vocab_size = encoder.config.type_vocab_size
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("incompatible torch module or BERT-style encoder") from error
    if type(type_vocab_size) is not int or type_vocab_size < 2:
        raise ValueError(
            "predicate conditioning requires an encoder with at least two "
            "token-type embeddings"
        )

    class PredicateConditionedBert(module_base):
        """Runtime-defined module so PyTorch stays an optional dependency.

        Persist checkpoints as ``state_dict`` plus the base model name and
        label configuration. Full-object pickling is not a supported contract.
        """

        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.base_model_name = model_name
            self.base_model_revision = model_revision
            self.num_labels = num_labels
            self.classifier = linear_type(hidden_size, num_labels)
            self.loss_function = loss_type(ignore_index=-100)

        def forward(
            self,
            input_ids: Any,
            *,
            attention_mask: Any | None = None,
            token_type_ids: Any | None = None,
            labels: Any | None = None,
            **encoder_kwargs: Any,
        ) -> SRLModelOutput:
            if token_type_ids is None:
                raise ValueError(
                    "token_type_ids must explicitly contain the predicate "
                    "indicator or the all-zero ablation"
                )
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **encoder_kwargs,
            )
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = outputs[0]
            logits = self.classifier(hidden_states)
            loss = None
            if labels is not None:
                loss = self.loss_function(
                    logits.reshape(-1, self.num_labels), labels.reshape(-1)
                )
            return SRLModelOutput(logits=logits, loss=loss)

    PredicateConditionedBert.__name__ = "PredicateConditionedBert"
    return PredicateConditionedBert()
