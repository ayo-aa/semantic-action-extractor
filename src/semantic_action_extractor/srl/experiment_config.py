"""Strict, dependency-free configuration for supplied-predicate SRL runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib


EXPERIMENT_CONFIG_VERSION = 1
EXPERIMENT_VARIANTS = frozenset(
    {"predicate_signal", "no_predicate_signal"}
)
CHECKPOINT_SELECTION_METRICS = frozenset(
    {"development_argument_f1", "development_loss"}
)
DEVICE_REQUESTS = frozenset({"auto", "cpu", "mps", "cuda"})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HUB_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_REQUIRED_KEYS = frozenset(
    {
        "config_version",
        "variant",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "prepared_data_fingerprint",
        "max_length",
        "batch_size",
        "learning_rate",
        "epochs",
        "weight_decay",
        "warmup_ratio",
        "gradient_clip_norm",
        "paired_seeds",
        "checkpoint_selection_metric",
        "device_request",
    }
)


def _require_exact_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    return value


def _require_exact_float(value: object, *, field: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{field} must be a TOML float")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field} must be nonempty without surrounding whitespace")
    return value


def _validate_model_id(value: object, *, field: str) -> str:
    identifier = _require_string(value, field=field)
    if _MODEL_ID_RE.fullmatch(identifier) is None:
        raise ValueError(f"{field} must be a model-hub identifier")
    unsafe_segment = any(
        part in {"", ".", ".."} for part in identifier.split("/")
    )
    if "//" in identifier or unsafe_segment:
        raise ValueError(f"{field} contains an unsafe path segment")
    return identifier


def _validate_revision(value: object, *, field: str) -> str:
    revision = _require_string(value, field=field)
    if _HUB_REVISION_RE.fullmatch(revision) is None:
        raise ValueError(f"{field} must be a pinned 40-character lowercase commit")
    return revision


def _validate_sha256(value: object, *, field: str) -> str:
    digest = _require_string(value, field=field)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must contain 64 lowercase hexadecimal digits")
    return digest


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """One fully explicit neural training configuration.

    Numeric fields deliberately require their exact TOML types.  For example,
    ``learning_rate = 1`` is rejected rather than coerced to ``1.0``.  Model
    and tokenizer revisions must be immutable 40-character repository commits;
    mutable names such as ``main`` are not accepted.
    """

    config_version: int
    variant: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    prepared_data_fingerprint: str
    max_length: int
    batch_size: int
    learning_rate: float
    epochs: int
    weight_decay: float
    warmup_ratio: float
    gradient_clip_norm: float
    paired_seeds: tuple[int, ...]
    checkpoint_selection_metric: str
    device_request: str

    def __post_init__(self) -> None:
        version = _require_exact_int(
            self.config_version, field="config_version"
        )
        if version != EXPERIMENT_CONFIG_VERSION:
            raise ValueError(
                f"config_version must be {EXPERIMENT_CONFIG_VERSION}"
            )

        variant = _require_string(self.variant, field="variant")
        if variant not in EXPERIMENT_VARIANTS:
            raise ValueError(
                "variant must be 'predicate_signal' or 'no_predicate_signal'"
            )
        _validate_model_id(self.model_id, field="model_id")
        _validate_revision(self.model_revision, field="model_revision")
        _validate_model_id(self.tokenizer_id, field="tokenizer_id")
        _validate_revision(self.tokenizer_revision, field="tokenizer_revision")
        _validate_sha256(
            self.prepared_data_fingerprint,
            field="prepared_data_fingerprint",
        )

        max_length = _require_exact_int(self.max_length, field="max_length")
        if not 3 <= max_length <= 512:
            raise ValueError("max_length must be between 3 and 512")
        batch_size = _require_exact_int(self.batch_size, field="batch_size")
        if not 1 <= batch_size <= 1024:
            raise ValueError("batch_size must be between 1 and 1024")
        learning_rate = _require_exact_float(
            self.learning_rate, field="learning_rate"
        )
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be greater than 0 and at most 1")
        epochs = _require_exact_int(self.epochs, field="epochs")
        if not 1 <= epochs <= 1000:
            raise ValueError("epochs must be between 1 and 1000")
        weight_decay = _require_exact_float(
            self.weight_decay, field="weight_decay"
        )
        if not 0.0 <= weight_decay <= 1.0:
            raise ValueError("weight_decay must be between 0 and 1")
        warmup_ratio = _require_exact_float(
            self.warmup_ratio, field="warmup_ratio"
        )
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be at least 0 and less than 1")
        gradient_clip = _require_exact_float(
            self.gradient_clip_norm, field="gradient_clip_norm"
        )
        if not 0.0 < gradient_clip <= 1_000_000.0:
            raise ValueError(
                "gradient_clip_norm must be greater than 0 and at most 1000000"
            )

        if not isinstance(self.paired_seeds, tuple):
            raise TypeError("paired_seeds must be a tuple")
        if len(self.paired_seeds) != 3:
            raise ValueError("paired_seeds must contain exactly three seeds")
        for seed in self.paired_seeds:
            _require_exact_int(seed, field="paired_seeds item")
            if not 0 <= seed <= 0xFFFFFFFF:
                raise ValueError(
                    "paired_seeds items must be between 0 and 4294967295"
                )
        if len(set(self.paired_seeds)) != len(self.paired_seeds):
            raise ValueError("paired_seeds must be unique")

        metric = _require_string(
            self.checkpoint_selection_metric,
            field="checkpoint_selection_metric",
        )
        if metric not in CHECKPOINT_SELECTION_METRICS:
            raise ValueError(
                "checkpoint_selection_metric must be "
                "'development_argument_f1' or 'development_loss'"
            )
        device = _require_string(self.device_request, field="device_request")
        if device not in DEVICE_REQUESTS:
            raise ValueError("device_request must be auto, cpu, mps, or cuda")

    @property
    def predicate_signal(self) -> bool:
        """Whether this configuration enables the predicate indicator."""

        return self.variant == "predicate_signal"

    def to_dict(self) -> dict[str, object]:
        """Return the complete configuration as plain canonical values."""

        return {
            "config_version": self.config_version,
            "variant": self.variant,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "prepared_data_fingerprint": self.prepared_data_fingerprint,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "gradient_clip_norm": self.gradient_clip_norm,
            "paired_seeds": list(self.paired_seeds),
            "checkpoint_selection_metric": self.checkpoint_selection_metric,
            "device_request": self.device_request,
        }

    def canonical_json_bytes(self) -> bytes:
        """Return a whitespace-independent canonical representation."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @property
    def digest(self) -> str:
        """SHA-256 of the parsed configuration's canonical JSON bytes."""

        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


def parse_training_config(source: str) -> TrainingConfig:
    """Parse one exact-schema TOML training configuration."""

    if not isinstance(source, str):
        raise TypeError("training configuration source must be a string")
    try:
        raw = tomllib.loads(source)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid training configuration TOML: {error}") from error

    actual_keys = frozenset(raw)
    missing = sorted(_REQUIRED_KEYS - actual_keys)
    unknown = sorted(actual_keys - _REQUIRED_KEYS)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError("invalid training configuration keys; " + "; ".join(details))

    seeds = raw["paired_seeds"]
    if not isinstance(seeds, list):
        raise TypeError("paired_seeds must be a TOML array")
    return TrainingConfig(
        config_version=raw["config_version"],
        variant=raw["variant"],
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        tokenizer_id=raw["tokenizer_id"],
        tokenizer_revision=raw["tokenizer_revision"],
        prepared_data_fingerprint=raw["prepared_data_fingerprint"],
        max_length=raw["max_length"],
        batch_size=raw["batch_size"],
        learning_rate=raw["learning_rate"],
        epochs=raw["epochs"],
        weight_decay=raw["weight_decay"],
        warmup_ratio=raw["warmup_ratio"],
        gradient_clip_norm=raw["gradient_clip_norm"],
        paired_seeds=tuple(seeds),
        checkpoint_selection_metric=raw["checkpoint_selection_metric"],
        device_request=raw["device_request"],
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    """Read UTF-8 TOML from a caller-supplied path and parse it strictly."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"training configuration is not valid UTF-8: {source}"
        ) from error
    return parse_training_config(text)
