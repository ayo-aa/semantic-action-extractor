"""Reproducible optional-dependency training for supplied-predicate SRL.

Importing this module does not import PyTorch or Transformers.  The concrete
runtime loads those packages only when a real experiment is requested, while
the orchestration boundary accepts an injected runtime for deterministic unit
tests and specialized execution environments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import random
import re
import shutil
import tempfile
from typing import Any, Protocol

from .alignment import collapse_subword_predictions
from .batching import (
    AlignedModelExample,
    PaddedSRLBatch,
    PreparedSRLSplit,
    collate_srl_batch,
    prepare_srl_split,
)
from .checkpoint_bundle import (
    CheckpointLabelConfig,
    load_checkpoint_bundle,
    save_checkpoint_bundle,
)
from .dataset_io import read_prepared_dataset
from .evaluation import (
    SuppliedPredicateEvaluation,
    evaluate_supplied_predicate_srl,
)
from .experiment_config import TrainingConfig
from .label_vocabulary import (
    SRLLabelVocabulary,
    build_training_label_vocabulary,
)
from .model import build_predicate_conditioned_bert
from .run_metadata import RunMetadata


EXPERIMENT_RESULT_VERSION = 1
PAIRED_RESULT_VERSION = 1

_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z"
)


class OptionalTrainingDependencyError(ImportError):
    """Raised when a real neural run lacks Torch or Transformers."""


@dataclass(frozen=True, slots=True)
class RunContext:
    """Caller-owned immutable revision and actual UTC start of one run."""

    git_revision: str
    started_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.git_revision, str):
            raise TypeError("git_revision must be a string")
        if _GIT_REVISION_RE.fullmatch(self.git_revision) is None:
            raise ValueError(
                "git_revision must be a 40-character lowercase commit"
            )
        _parse_utc_timestamp(self.started_at, field="started_at")


@dataclass(frozen=True, slots=True)
class BatchRuntimeOutput:
    """Framework-neutral evaluation output for one padded batch."""

    loss: float
    predicted_label_ids: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _require_finite_float(self.loss, field="batch evaluation loss")
        if not isinstance(self.predicted_label_ids, tuple):
            raise TypeError("predicted label rows must be a tuple")
        if any(
            not isinstance(row, tuple) for row in self.predicted_label_ids
        ):
            raise TypeError("each predicted label row must be a tuple")
        for row in self.predicted_label_ids:
            if any(type(label_id) is not int for label_id in row):
                raise TypeError("predicted label IDs must be integers")


class TrainingRuntime(Protocol):
    """Small runtime surface consumed by the dependency-free orchestrator."""

    def load_tokenizer(self, config: TrainingConfig) -> Any: ...

    def resolve_device(self, request: str) -> str: ...

    def seed_everything(self, seed: int) -> None: ...

    def build_model(
        self,
        config: TrainingConfig,
        num_labels: int,
        device: str,
    ) -> Any: ...

    def initial_state_fingerprint(self, model: Any) -> str: ...

    def create_optimizer(self, model: Any, config: TrainingConfig) -> Any: ...

    def create_scheduler(
        self,
        optimizer: Any,
        *,
        warmup_steps: int,
        total_steps: int,
    ) -> Any: ...

    def train_batch(
        self,
        model: Any,
        batch: PaddedSRLBatch,
        optimizer: Any,
        scheduler: Any,
        *,
        gradient_clip_norm: float,
        device: str,
    ) -> float: ...

    def evaluate_batch(
        self,
        model: Any,
        batch: PaddedSRLBatch,
        *,
        device: str,
    ) -> BatchRuntimeOutput: ...

    def capture_state_dict(self, model: Any) -> Any: ...

    def restore_state_dict(self, model: Any, state_dict: Any) -> None: ...

    def save_state_dict(self, state_dict: Any, path: Path) -> None: ...

    def load_state_dict(self, path: Path, *, device: str) -> Any: ...

    def package_versions(self) -> Mapping[str, str]: ...

    def hardware(self, *, device: str) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Canonical loss and exact word-level metrics for one frozen split."""

    loss: float
    evaluation: SuppliedPredicateEvaluation

    def __post_init__(self) -> None:
        _require_finite_float(self.loss, field="evaluation loss")
        if not isinstance(self.evaluation, SuppliedPredicateEvaluation):
            raise TypeError("evaluation must be SuppliedPredicateEvaluation")
        _require_finite_evaluation(self.evaluation)

    def to_dict(self) -> dict[str, object]:
        arguments = self.evaluation.arguments
        predicate = self.evaluation.predicate
        token_accuracy = self.evaluation.token_accuracy
        return {
            "loss": self.loss,
            "arguments": {
                "true_positives": arguments.true_positives,
                "predicted": arguments.predicted,
                "gold": arguments.gold,
                "precision": arguments.precision,
                "recall": arguments.recall,
                "f1": arguments.f1,
                "repaired_prediction_tags": (
                    arguments.repaired_prediction_tags
                ),
            },
            "per_role": [
                {
                    "label": role.label,
                    "true_positives": role.true_positives,
                    "predicted": role.predicted,
                    "gold": role.gold,
                    "precision": role.precision,
                    "recall": role.recall,
                    "f1": role.f1,
                }
                for role in self.evaluation.per_role
            ],
            "predicate": {
                "examples": predicate.examples,
                "correct_anchors": predicate.correct_anchors,
                "missing_anchors": predicate.missing_anchors,
                "spurious_predicate_words": (
                    predicate.spurious_predicate_words
                ),
                "argument_spans_overlapping_predicate": (
                    predicate.argument_spans_overlapping_predicate
                ),
            },
            "token_accuracy": {
                "correct": token_accuracy.correct,
                "total": token_accuracy.total,
                "accuracy": token_accuracy.accuracy,
            },
        }


@dataclass(frozen=True, slots=True)
class EpochSummary:
    """One training epoch and its development-only selection evidence."""

    epoch: int
    training_loss: float
    development: EvaluationSummary
    selection_value: float

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        _require_finite_float(self.training_loss, field="training loss")
        if not isinstance(self.development, EvaluationSummary):
            raise TypeError("development must be an EvaluationSummary")
        _require_finite_float(self.selection_value, field="selection value")

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "training_loss": self.training_loss,
            "development": self.development.to_dict(),
            "selection_value": self.selection_value,
        }


@dataclass(frozen=True, slots=True)
class SRLRunResult:
    """Canonical aggregate result for one seed and one model variant."""

    result_version: int
    metadata: RunMetadata
    best_epoch: int
    checkpoint_selection_metric: str
    best_development: EvaluationSummary
    test: EvaluationSummary
    epochs: tuple[EpochSummary, ...]

    def __post_init__(self) -> None:
        if type(self.result_version) is not int:
            raise TypeError("result_version must be an integer")
        if self.result_version != EXPERIMENT_RESULT_VERSION:
            raise ValueError(
                f"result_version must be {EXPERIMENT_RESULT_VERSION}"
            )
        if not isinstance(self.metadata, RunMetadata):
            raise TypeError("metadata must be RunMetadata")
        if not isinstance(self.epochs, tuple) or not self.epochs:
            raise ValueError("epochs must be a nonempty tuple")
        if any(not isinstance(item, EpochSummary) for item in self.epochs):
            raise TypeError("epochs must contain EpochSummary values")
        expected_epochs = tuple(range(1, len(self.epochs) + 1))
        if tuple(item.epoch for item in self.epochs) != expected_epochs:
            raise ValueError("epochs must be consecutive and one-indexed")
        if type(self.best_epoch) is not int:
            raise TypeError("best_epoch must be an integer")
        if self.best_epoch not in expected_epochs:
            raise ValueError("best_epoch must identify a recorded epoch")
        if not isinstance(self.best_development, EvaluationSummary):
            raise TypeError("best_development must be an EvaluationSummary")
        if self.best_development != self.epochs[self.best_epoch - 1].development:
            raise ValueError("best development summary must match best_epoch")
        if not isinstance(self.test, EvaluationSummary):
            raise TypeError("test must be an EvaluationSummary")
        if self.checkpoint_selection_metric not in {
            "development_argument_f1",
            "development_loss",
        }:
            raise ValueError("unsupported checkpoint selection metric")

    def to_dict(self) -> dict[str, object]:
        return {
            "result_version": self.result_version,
            "metadata": self.metadata.to_dict(),
            "best_epoch": self.best_epoch,
            "checkpoint_selection_metric": self.checkpoint_selection_metric,
            "best_development": self.best_development.to_dict(),
            "test": self.test.to_dict(),
            "epochs": [epoch.to_dict() for epoch in self.epochs],
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PairedSeedResult:
    """Conditioned and ablated results sharing one seed and initial state."""

    seed: int
    predicate_signal: SRLRunResult
    no_predicate_signal: SRLRunResult
    development_argument_f1_delta: float
    test_argument_f1_delta: float

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise TypeError("paired result seed must be an integer")
        for result in (self.predicate_signal, self.no_predicate_signal):
            if not isinstance(result, SRLRunResult):
                raise TypeError("paired runs must be SRLRunResult values")
            if result.metadata.seed != self.seed:
                raise ValueError("paired run seed does not match its result")
        if self.predicate_signal.metadata.variant != "predicate_signal":
            raise ValueError("conditioned result has the wrong variant")
        if self.no_predicate_signal.metadata.variant != "no_predicate_signal":
            raise ValueError("ablation result has the wrong variant")
        if (
            self.predicate_signal.metadata.initial_state_fingerprint
            != self.no_predicate_signal.metadata.initial_state_fingerprint
        ):
            raise ValueError("paired runs must share the initial state fingerprint")
        if (
            self.predicate_signal.metadata.resolved_device
            != self.no_predicate_signal.metadata.resolved_device
        ):
            raise ValueError("paired runs must use the same resolved device")
        if (
            self.predicate_signal.metadata.labels
            != self.no_predicate_signal.metadata.labels
        ):
            raise ValueError("paired runs must use the same label vocabulary")
        if (
            self.predicate_signal.metadata.counts
            != self.no_predicate_signal.metadata.counts
        ):
            raise ValueError("paired runs must use the same example and token counts")
        if (
            self.predicate_signal.metadata.drop_stats
            != self.no_predicate_signal.metadata.drop_stats
        ):
            raise ValueError("paired runs must use the same drop accounting")
        _require_finite_float(
            self.development_argument_f1_delta,
            field="development F1 delta",
        )
        _require_finite_float(
            self.test_argument_f1_delta,
            field="test F1 delta",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "predicate_signal": self.predicate_signal.to_dict(),
            "no_predicate_signal": self.no_predicate_signal.to_dict(),
            "development_argument_f1_delta": (
                self.development_argument_f1_delta
            ),
            "test_argument_f1_delta": self.test_argument_f1_delta,
        }


@dataclass(frozen=True, slots=True)
class VariantAggregate:
    """Mean and sample SD for one variant across the three paired seeds."""

    variant: str
    development_argument_f1_mean: float
    development_argument_f1_sample_stddev: float
    test_argument_f1_mean: float
    test_argument_f1_sample_stddev: float

    def __post_init__(self) -> None:
        if self.variant not in {
            "predicate_signal",
            "no_predicate_signal",
        }:
            raise ValueError("unsupported aggregate variant")
        for field in (
            "development_argument_f1_mean",
            "development_argument_f1_sample_stddev",
            "test_argument_f1_mean",
            "test_argument_f1_sample_stddev",
        ):
            _require_finite_float(getattr(self, field), field=field)

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "development_argument_f1_mean": (
                self.development_argument_f1_mean
            ),
            "development_argument_f1_sample_stddev": (
                self.development_argument_f1_sample_stddev
            ),
            "test_argument_f1_mean": self.test_argument_f1_mean,
            "test_argument_f1_sample_stddev": (
                self.test_argument_f1_sample_stddev
            ),
        }


@dataclass(frozen=True, slots=True)
class PairedEffectAggregate:
    """Mean and sample SD of conditioned-minus-ablation seed effects."""

    development_argument_f1_delta_mean: float
    development_argument_f1_delta_sample_stddev: float
    test_argument_f1_delta_mean: float
    test_argument_f1_delta_sample_stddev: float

    def __post_init__(self) -> None:
        for field in (
            "development_argument_f1_delta_mean",
            "development_argument_f1_delta_sample_stddev",
            "test_argument_f1_delta_mean",
            "test_argument_f1_delta_sample_stddev",
        ):
            _require_finite_float(getattr(self, field), field=field)

    def to_dict(self) -> dict[str, float]:
        return {
            "development_argument_f1_delta_mean": (
                self.development_argument_f1_delta_mean
            ),
            "development_argument_f1_delta_sample_stddev": (
                self.development_argument_f1_delta_sample_stddev
            ),
            "test_argument_f1_delta_mean": self.test_argument_f1_delta_mean,
            "test_argument_f1_delta_sample_stddev": (
                self.test_argument_f1_delta_sample_stddev
            ),
        }


@dataclass(frozen=True, slots=True)
class PairedExperimentResult:
    """Canonical three-seed aggregate for the predicate-signal ablation."""

    paired_result_version: int
    pairs: tuple[PairedSeedResult, ...]
    aggregates: tuple[VariantAggregate, ...]
    paired_effect: PairedEffectAggregate

    def __post_init__(self) -> None:
        if type(self.paired_result_version) is not int:
            raise TypeError("paired_result_version must be an integer")
        if self.paired_result_version != PAIRED_RESULT_VERSION:
            raise ValueError(
                f"paired_result_version must be {PAIRED_RESULT_VERSION}"
            )
        if not isinstance(self.pairs, tuple) or len(self.pairs) != 3:
            raise ValueError("paired result must contain exactly three seeds")
        if any(not isinstance(pair, PairedSeedResult) for pair in self.pairs):
            raise TypeError("pairs must contain PairedSeedResult values")
        seeds = tuple(pair.seed for pair in self.pairs)
        if len(set(seeds)) != 3:
            raise ValueError("paired result seeds must be unique")
        if not isinstance(self.aggregates, tuple) or len(self.aggregates) != 2:
            raise ValueError("aggregates must contain exactly two variants")
        if any(
            not isinstance(item, VariantAggregate) for item in self.aggregates
        ):
            raise TypeError("aggregates must contain VariantAggregate values")
        if tuple(item.variant for item in self.aggregates) != (
            "predicate_signal",
            "no_predicate_signal",
        ):
            raise ValueError("aggregates must use canonical variant order")
        if not isinstance(self.paired_effect, PairedEffectAggregate):
            raise TypeError("paired_effect must be PairedEffectAggregate")

    def to_dict(self) -> dict[str, object]:
        return {
            "paired_result_version": self.paired_result_version,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "aggregates": [item.to_dict() for item in self.aggregates],
            "paired_effect": self.paired_effect.to_dict(),
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


class TorchTrainingRuntime:
    """Real Torch/Transformers implementation of :class:`TrainingRuntime`."""

    def __init__(
        self,
        *,
        torch_module: Any | None = None,
        transformers_module: Any | None = None,
    ) -> None:
        self.torch = (
            torch_module
            if torch_module is not None
            else _load_optional_module("torch")
        )
        self.transformers = (
            transformers_module
            if transformers_module is not None
            else _load_optional_module("transformers")
        )

    def load_tokenizer(self, config: TrainingConfig) -> Any:
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_id,
            revision=config.tokenizer_revision,
            use_fast=True,
        )
        if getattr(tokenizer, "is_fast", None) is not True:
            raise ValueError("SRL alignment requires a pinned fast tokenizer")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("fast tokenizer must define a non-negative pad token ID")
        return tokenizer

    def resolve_device(self, request: str) -> str:
        if request not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device request must be auto, cpu, mps, or cuda")
        cuda_available = bool(self.torch.cuda.is_available())
        mps_backend = getattr(getattr(self.torch, "backends", None), "mps", None)
        mps_available = bool(
            mps_backend is not None and mps_backend.is_available()
        )
        if request == "cuda":
            if not cuda_available:
                raise RuntimeError("CUDA was requested but is unavailable")
            return "cuda"
        if request == "mps":
            if not mps_available:
                raise RuntimeError("MPS was requested but is unavailable")
            return "mps"
        if request == "cpu":
            return "cpu"
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    def seed_everything(self, seed: int) -> None:
        random.seed(seed)
        set_seed = getattr(self.transformers, "set_seed", None)
        if callable(set_seed):
            set_seed(seed)
        self.torch.manual_seed(seed)
        if bool(self.torch.cuda.is_available()):
            self.torch.cuda.manual_seed_all(seed)
        deterministic = getattr(
            self.torch, "use_deterministic_algorithms", None
        )
        if callable(deterministic):
            deterministic(True)
        cudnn = getattr(getattr(self.torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False

    def build_model(
        self,
        config: TrainingConfig,
        num_labels: int,
        device: str,
    ) -> Any:
        model = build_predicate_conditioned_bert(
            config.model_id,
            num_labels,
            model_revision=config.model_revision,
            torch_module=self.torch,
            transformers_module=self.transformers,
        )
        return model.to(self.torch.device(device))

    def initial_state_fingerprint(self, model: Any) -> str:
        digest = hashlib.sha256(b"srl-state-dict-v1\0")
        state_dict = model.state_dict()
        if not isinstance(state_dict, Mapping):
            raise TypeError("model state_dict must be a mapping")
        for name in sorted(state_dict):
            if not isinstance(name, str):
                raise TypeError("state_dict keys must be strings")
            tensor = state_dict[name]
            try:
                shape = tuple(int(size) for size in tensor.shape)
                dtype = str(tensor.dtype)
                flat = tensor.detach().cpu().contiguous().reshape(-1)
                payload = flat.view(self.torch.uint8).numpy().tobytes()
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise TypeError(
                    f"state_dict value {name!r} must be a dense tensor"
                ) from error
            _update_length_prefixed(digest, name.encode("utf-8"))
            _update_length_prefixed(digest, dtype.encode("ascii"))
            _update_length_prefixed(
                digest,
                json.dumps(shape, separators=(",", ":")).encode("ascii"),
            )
            _update_length_prefixed(digest, payload)
        return digest.hexdigest()

    def create_optimizer(self, model: Any, config: TrainingConfig) -> Any:
        return self.torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def create_scheduler(
        self,
        optimizer: Any,
        *,
        warmup_steps: int,
        total_steps: int,
    ) -> Any:
        del total_steps
        # Transformers implements a linear warmup followed by constant LR.
        return self.transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )

    def train_batch(
        self,
        model: Any,
        batch: PaddedSRLBatch,
        optimizer: Any,
        scheduler: Any,
        *,
        gradient_clip_norm: float,
        device: str,
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(**self._tensorize(batch, device=device))
        loss = getattr(output, "loss", None)
        if loss is None and isinstance(output, Mapping):
            loss = output.get("loss")
        loss_value = _torch_loss_value(loss)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        return loss_value

    def evaluate_batch(
        self,
        model: Any,
        batch: PaddedSRLBatch,
        *,
        device: str,
    ) -> BatchRuntimeOutput:
        model.eval()
        with self.torch.no_grad():
            output = model(**self._tensorize(batch, device=device))
        loss = getattr(output, "loss", None)
        logits = getattr(output, "logits", None)
        if isinstance(output, Mapping):
            if loss is None:
                loss = output.get("loss")
            if logits is None:
                logits = output.get("logits")
        loss_value = _torch_loss_value(loss)
        if logits is None:
            raise ValueError("model evaluation output is missing logits")
        predicted = logits.argmax(dim=-1).detach().cpu().tolist()
        return BatchRuntimeOutput(
            loss=loss_value,
            predicted_label_ids=tuple(tuple(row) for row in predicted),
        )

    def capture_state_dict(self, model: Any) -> Any:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    def restore_state_dict(self, model: Any, state_dict: Any) -> None:
        model.load_state_dict(state_dict, strict=True)

    def save_state_dict(self, state_dict: Any, path: Path) -> None:
        self.torch.save(state_dict, path)

    def load_state_dict(self, path: Path, *, device: str) -> Any:
        return self.torch.load(
            path,
            map_location=self.torch.device(device),
            weights_only=True,
        )

    def package_versions(self) -> Mapping[str, str]:
        return {
            "python": platform.python_version(),
            "torch": str(self.torch.__version__),
            "transformers": str(self.transformers.__version__),
        }

    def hardware(self, *, device: str) -> Mapping[str, str]:
        if device.startswith("cuda"):
            accelerator = str(self.torch.cuda.get_device_name(0))
        elif device == "mps":
            accelerator = "Apple Metal Performance Shaders"
        else:
            accelerator = platform.processor().strip() or "generic-cpu"
        return {
            "accelerator": accelerator,
            "machine": platform.machine() or "unknown-machine",
            "operating_system": platform.platform() or "unknown-platform",
        }

    def _tensorize(
        self, batch: PaddedSRLBatch, *, device: str
    ) -> dict[str, Any]:
        target = self.torch.device(device)
        return {
            name: self.torch.tensor(values, dtype=self.torch.long, device=target)
            for name, values in batch.to_model_inputs().items()
        }


def deterministic_batch_indices(
    item_count: int,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> tuple[tuple[int, ...], ...]:
    """Return stable complete batches without consulting global RNG state."""

    for field, value in (
        ("item_count", item_count),
        ("batch_size", batch_size),
        ("seed", seed),
        ("epoch", epoch),
    ):
        if type(value) is not int:
            raise TypeError(f"{field} must be an integer")
    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be between 0 and 4294967295")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if type(shuffle) is not bool:
        raise TypeError("shuffle must be a boolean")

    indexes = list(range(item_count))
    if shuffle:
        seed_material = f"srl-batches-v1\0{seed}\0{epoch}".encode("ascii")
        epoch_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest(), "big"
        )
        random.Random(epoch_seed).shuffle(indexes)
    return tuple(
        tuple(indexes[start : start + batch_size])
        for start in range(0, item_count, batch_size)
    )


def run_srl_experiment(
    dataset_directory: str | Path,
    config: TrainingConfig,
    checkpoint_directory: str | Path,
    *,
    seed: int,
    context: RunContext,
    runtime: TrainingRuntime | None = None,
    utc_now: Callable[[], str] | None = None,
) -> SRLRunResult:
    """Train, select on development, reload best state, and test once."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed not in config.paired_seeds:
        raise ValueError("seed must be one of config.paired_seeds")
    if not isinstance(context, RunContext):
        raise TypeError("context must be RunContext")
    clock = utc_now if utc_now is not None else _canonical_utc_now
    if not callable(clock):
        raise TypeError("utc_now must be callable")
    target = Path(checkpoint_directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"checkpoint destination already exists: {target}")

    dataset = read_prepared_dataset(dataset_directory)
    if dataset.manifest.dataset_fingerprint != config.prepared_data_fingerprint:
        raise ValueError(
            "prepared dataset fingerprint does not match training config"
        )
    records = dataset.examples
    training_records = tuple(
        record for record in records if record.split == "train"
    )
    vocabulary = build_training_label_vocabulary(training_records)

    active_runtime: TrainingRuntime
    active_runtime = runtime if runtime is not None else TorchTrainingRuntime()
    tokenizer = active_runtime.load_tokenizer(config)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("tokenizer must define a non-negative pad token ID")

    prepared = {
        split: prepare_srl_split(
            tokenizer,
            records,
            vocabulary,
            split,
            max_length=config.max_length,
        )
        for split in ("train", "development", "test")
    }
    for split, prepared_split in prepared.items():
        if prepared_split.retained_count == 0:
            raise ValueError(f"empty retained {split} split")

    resolved_device = active_runtime.resolve_device(config.device_request)
    active_runtime.seed_everything(seed)
    model = active_runtime.build_model(
        config, len(vocabulary), resolved_device
    )
    initial_fingerprint = active_runtime.initial_state_fingerprint(model)
    _require_sha256(
        initial_fingerprint, field="initial state fingerprint"
    )

    train_split = prepared["train"]
    steps_per_epoch = math.ceil(
        train_split.retained_count / config.batch_size
    )
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = math.floor(total_steps * config.warmup_ratio)
    optimizer = active_runtime.create_optimizer(model, config)
    scheduler = active_runtime.create_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    source_by_id = {record.example_id: record for record in records}
    epochs: list[EpochSummary] = []
    best_epoch: int | None = None
    best_development: EvaluationSummary | None = None
    best_state: Any | None = None
    optimizer_steps = 0

    for epoch in range(1, config.epochs + 1):
        weighted_train_loss = 0.0
        trained_tokens = 0
        for indexes in deterministic_batch_indices(
            train_split.retained_count,
            batch_size=config.batch_size,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        ):
            batch = collate_srl_batch(
                tuple(train_split.examples[index] for index in indexes),
                pad_token_id=pad_token_id,
                predicate_signal=config.predicate_signal,
            )
            active_labels = _active_label_count(batch)
            batch_loss = active_runtime.train_batch(
                model,
                batch,
                optimizer,
                scheduler,
                gradient_clip_norm=config.gradient_clip_norm,
                device=resolved_device,
            )
            _require_finite_float(batch_loss, field="training batch loss")
            weighted_train_loss += batch_loss * active_labels
            trained_tokens += active_labels
            optimizer_steps += 1
        if trained_tokens == 0:
            raise ValueError("training epoch contains no active labels")
        training_loss = weighted_train_loss / trained_tokens
        _require_finite_float(training_loss, field="training epoch loss")

        development = _evaluate_split(
            active_runtime,
            model,
            prepared["development"],
            source_by_id,
            vocabulary,
            batch_size=config.batch_size,
            pad_token_id=pad_token_id,
            predicate_signal=config.predicate_signal,
            device=resolved_device,
        )
        selection_value = _selection_value(config, development)
        summary = EpochSummary(
            epoch=epoch,
            training_loss=training_loss,
            development=development,
            selection_value=selection_value,
        )
        epochs.append(summary)
        if best_development is None or _is_better(
            config,
            candidate=development,
            incumbent=best_development,
        ):
            best_epoch = epoch
            best_development = development
            best_state = active_runtime.capture_state_dict(model)

    if best_epoch is None or best_development is None or best_state is None:
        raise RuntimeError("training did not produce a development checkpoint")
    if optimizer_steps != total_steps:
        raise RuntimeError("optimizer step count does not match the schedule")

    active_runtime.restore_state_dict(model, best_state)
    test_summary = _evaluate_split(
        active_runtime,
        model,
        prepared["test"],
        source_by_id,
        vocabulary,
        batch_size=config.batch_size,
        pad_token_id=pad_token_id,
        predicate_signal=config.predicate_signal,
        device=resolved_device,
    )
    package_versions = active_runtime.package_versions()
    hardware = active_runtime.hardware(device=resolved_device)
    counts = _run_counts(
        prepared, config.epochs, optimizer_steps, warmup_steps
    )
    drop_stats = _drop_stats(prepared)
    recorded_at = clock()
    _parse_utc_timestamp(recorded_at, field="recorded_at")

    metadata = RunMetadata.create(
        git_revision=context.git_revision,
        config_digest=config.digest,
        dataset_fingerprint=dataset.manifest.dataset_fingerprint,
        labels=vocabulary.labels,
        package_versions=package_versions,
        hardware=hardware,
        started_at=context.started_at,
        recorded_at=recorded_at,
        seed=seed,
        variant=config.variant,
        requested_device=config.device_request,
        resolved_device=resolved_device,
        counts=counts,
        drop_stats=drop_stats,
        initial_state_fingerprint=initial_fingerprint,
    )
    label_config = CheckpointLabelConfig.from_labels(vocabulary.labels)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.", suffix=".run-partial", dir=target.parent
        )
    )
    staged_checkpoint = staging_root / "checkpoint"
    try:
        save_checkpoint_bundle(
            staged_checkpoint,
            metadata=metadata,
            config=config,
            label_config=label_config,
            state_dict=best_state,
            save_state_dict=active_runtime.save_state_dict,
        )
        loaded = load_checkpoint_bundle(
            staged_checkpoint,
            expected_metadata=metadata,
            expected_config=config,
            expected_label_config=label_config,
            load_state_dict=lambda path: active_runtime.load_state_dict(
                path, device=resolved_device
            ),
        )
        active_runtime.restore_state_dict(model, loaded.state_dict)
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"checkpoint destination already exists: {target}"
            )
        staged_checkpoint.rename(target)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    return SRLRunResult(
        result_version=EXPERIMENT_RESULT_VERSION,
        metadata=metadata,
        best_epoch=best_epoch,
        checkpoint_selection_metric=config.checkpoint_selection_metric,
        best_development=best_development,
        test=test_summary,
        epochs=tuple(epochs),
    )


def run_paired_srl_experiments(
    predicate_signal_config: TrainingConfig,
    no_predicate_signal_config: TrainingConfig,
    *,
    run_one: Callable[[TrainingConfig, int], SRLRunResult],
) -> PairedExperimentResult:
    """Run three matched seeds and reject any non-variant config difference."""

    validate_paired_configs(
        predicate_signal_config, no_predicate_signal_config
    )
    if not callable(run_one):
        raise TypeError("run_one must be callable")

    pairs: list[PairedSeedResult] = []
    for seed in predicate_signal_config.paired_seeds:
        conditioned = run_one(predicate_signal_config, seed)
        ablated = run_one(no_predicate_signal_config, seed)
        if not isinstance(conditioned, SRLRunResult):
            raise TypeError("run_one must return SRLRunResult")
        if not isinstance(ablated, SRLRunResult):
            raise TypeError("run_one must return SRLRunResult")
        conditioned.metadata.assert_config_compatible(
            predicate_signal_config
        )
        ablated.metadata.assert_config_compatible(
            no_predicate_signal_config
        )
        development_delta = (
            conditioned.best_development.evaluation.arguments.f1
            - ablated.best_development.evaluation.arguments.f1
        )
        test_delta = (
            conditioned.test.evaluation.arguments.f1
            - ablated.test.evaluation.arguments.f1
        )
        pairs.append(
            PairedSeedResult(
                seed=seed,
                predicate_signal=conditioned,
                no_predicate_signal=ablated,
                development_argument_f1_delta=development_delta,
                test_argument_f1_delta=test_delta,
            )
        )

    return PairedExperimentResult(
        paired_result_version=PAIRED_RESULT_VERSION,
        pairs=tuple(pairs),
        aggregates=(
            _aggregate_variant(tuple(pairs), "predicate_signal"),
            _aggregate_variant(tuple(pairs), "no_predicate_signal"),
        ),
        paired_effect=_aggregate_paired_effect(tuple(pairs)),
    )


def _evaluate_split(
    runtime: TrainingRuntime,
    model: Any,
    prepared: PreparedSRLSplit,
    source_by_id: Mapping[str, Any],
    vocabulary: SRLLabelVocabulary,
    *,
    batch_size: int,
    pad_token_id: int,
    predicate_signal: bool,
    device: str,
) -> EvaluationSummary:
    gold_sequences: list[tuple[str, ...]] = []
    predicted_sequences: list[tuple[str, ...]] = []
    predicate_indexes: list[int] = []
    weighted_loss = 0.0
    active_tokens = 0

    for indexes in deterministic_batch_indices(
        prepared.retained_count,
        batch_size=batch_size,
        seed=0,
        epoch=0,
        shuffle=False,
    ):
        batch = collate_srl_batch(
            tuple(prepared.examples[index] for index in indexes),
            pad_token_id=pad_token_id,
            predicate_signal=predicate_signal,
        )
        output = runtime.evaluate_batch(model, batch, device=device)
        if not isinstance(output, BatchRuntimeOutput):
            raise TypeError("runtime evaluation must return BatchRuntimeOutput")
        if len(output.predicted_label_ids) != len(batch.example_ids):
            raise ValueError("prediction batch size does not match input batch")
        width = len(batch.input_ids[0])
        if any(len(row) != width for row in output.predicted_label_ids):
            raise ValueError("prediction token width does not match input batch")

        batch_active_tokens = _active_label_count(batch)
        weighted_loss += output.loss * batch_active_tokens
        active_tokens += batch_active_tokens
        for row_index, example_id in enumerate(batch.example_ids):
            record = source_by_id[example_id]
            collapsed_ids = collapse_subword_predictions(
                output.predicted_label_ids[row_index],
                batch.word_ids[row_index],
                batch.word_counts[row_index],
                attention_mask=batch.attention_mask[row_index],
            )
            predicted_sequences.append(
                tuple(vocabulary.decode(label_id) for label_id in collapsed_ids)
            )
            gold_sequences.append(record.tags)
            predicate_indexes.append(record.predicate_index)

    if active_tokens == 0:
        raise ValueError(f"{prepared.split} split contains no active labels")
    loss = weighted_loss / active_tokens
    _require_finite_float(loss, field=f"{prepared.split} loss")
    evaluation = evaluate_supplied_predicate_srl(
        gold_sequences,
        predicted_sequences,
        predicate_indexes,
        repair_predictions=True,
    )
    return EvaluationSummary(loss=loss, evaluation=evaluation)


def _active_label_count(batch: PaddedSRLBatch) -> int:
    count = sum(label != -100 for row in batch.labels for label in row)
    if count <= 0:
        raise ValueError("batch contains no active labels")
    return count


def _selection_value(
    config: TrainingConfig, summary: EvaluationSummary
) -> float:
    if config.checkpoint_selection_metric == "development_argument_f1":
        value = summary.evaluation.arguments.f1
    else:
        value = summary.loss
    _require_finite_float(value, field="checkpoint selection value")
    return value


def _is_better(
    config: TrainingConfig,
    *,
    candidate: EvaluationSummary,
    incumbent: EvaluationSummary,
) -> bool:
    candidate_value = _selection_value(config, candidate)
    incumbent_value = _selection_value(config, incumbent)
    if config.checkpoint_selection_metric == "development_argument_f1":
        return candidate_value > incumbent_value
    return candidate_value < incumbent_value


def _run_counts(
    prepared: Mapping[str, PreparedSRLSplit],
    epochs: int,
    optimizer_steps: int,
    warmup_steps: int,
) -> dict[str, int]:
    counts = {
        "epochs_completed": epochs,
        "optimizer_steps": optimizer_steps,
        "test_evaluations": 1,
        "training_active_tokens": (
            epochs * _prepared_active_label_count(prepared["train"])
        ),
        "warmup_steps": warmup_steps,
    }
    for split in ("train", "development", "test"):
        counts[f"{split}_active_tokens"] = _prepared_active_label_count(
            prepared[split]
        )
        counts[f"{split}_records"] = prepared[split].selected_records
        counts[f"{split}_retained"] = prepared[split].retained_count
    return counts


def _prepared_active_label_count(prepared: PreparedSRLSplit) -> int:
    return sum(
        label != -100
        for example in prepared.examples
        for label in example.alignment.labels
    )


def _drop_stats(
    prepared: Mapping[str, PreparedSRLSplit],
) -> dict[str, int]:
    stats = {
        f"{split}_overlength": prepared[split].overlength_drop_count
        for split in ("train", "development", "test")
    }
    stats["total_overlength"] = sum(stats.values())
    return stats


def validate_paired_configs(
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
) -> None:
    """Require paired configs to differ only by the declared variant."""

    if not isinstance(predicate_config, TrainingConfig):
        raise TypeError("predicate_signal_config must be TrainingConfig")
    if not isinstance(ablation_config, TrainingConfig):
        raise TypeError("no_predicate_signal_config must be TrainingConfig")
    if predicate_config.variant != "predicate_signal":
        raise ValueError("predicate_signal_config must enable predicate_signal")
    if ablation_config.variant != "no_predicate_signal":
        raise ValueError(
            "no_predicate_signal_config must enable no_predicate_signal"
        )
    predicate_values = predicate_config.to_dict()
    ablation_values = ablation_config.to_dict()
    predicate_values.pop("variant")
    ablation_values.pop("variant")
    differences = sorted(
        key
        for key in predicate_values
        if predicate_values[key] != ablation_values[key]
    )
    if differences:
        raise ValueError(
            "paired configs may differ only by variant; mismatches: "
            + ", ".join(differences)
        )


def _aggregate_variant(
    pairs: tuple[PairedSeedResult, ...], variant: str
) -> VariantAggregate:
    results = tuple(getattr(pair, variant) for pair in pairs)
    development_values = tuple(
        result.best_development.evaluation.arguments.f1 for result in results
    )
    test_values = tuple(result.test.evaluation.arguments.f1 for result in results)
    return VariantAggregate(
        variant=variant,
        development_argument_f1_mean=_mean(development_values),
        development_argument_f1_sample_stddev=_sample_stddev(
            development_values
        ),
        test_argument_f1_mean=_mean(test_values),
        test_argument_f1_sample_stddev=_sample_stddev(test_values),
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty value sequence")
    result = math.fsum(values) / len(values)
    return _require_finite_float(result, field="aggregate mean")


def _sample_stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("sample standard deviation requires at least two values")
    mean = _mean(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (
        len(values) - 1
    )
    result = math.sqrt(variance)
    return _require_finite_float(result, field="aggregate standard deviation")


def _aggregate_paired_effect(
    pairs: tuple[PairedSeedResult, ...],
) -> PairedEffectAggregate:
    development_deltas = tuple(
        pair.development_argument_f1_delta for pair in pairs
    )
    test_deltas = tuple(pair.test_argument_f1_delta for pair in pairs)
    return PairedEffectAggregate(
        development_argument_f1_delta_mean=_mean(development_deltas),
        development_argument_f1_delta_sample_stddev=_sample_stddev(
            development_deltas
        ),
        test_argument_f1_delta_mean=_mean(test_deltas),
        test_argument_f1_delta_sample_stddev=_sample_stddev(test_deltas),
    )


def _require_finite_evaluation(
    evaluation: SuppliedPredicateEvaluation,
) -> None:
    values = {
        "argument precision": evaluation.arguments.precision,
        "argument recall": evaluation.arguments.recall,
        "argument f1": evaluation.arguments.f1,
        "token accuracy": evaluation.token_accuracy.accuracy,
    }
    for role in evaluation.per_role:
        values[f"{role.label} precision"] = role.precision
        values[f"{role.label} recall"] = role.recall
        values[f"{role.label} f1"] = role.f1
    for field, value in values.items():
        _require_finite_float(value, field=field)


def _require_finite_float(value: object, *, field: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{field} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal digits")
    return value


def _parse_utc_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{field} must be UTC")
    return parsed


def _torch_loss_value(loss: Any) -> float:
    if loss is None:
        raise ValueError("model output is missing loss")
    try:
        value = float(loss.detach().cpu().item())
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise TypeError("model loss must be a scalar tensor") from error
    return _require_finite_float(value, field="model loss")


def _update_length_prefixed(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _load_optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise OptionalTrainingDependencyError(
            "SRL training requires the optional torch and transformers packages"
        ) from error
