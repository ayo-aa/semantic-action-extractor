"""Torch-independent checkpoint bundle layout and compatibility checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .experiment_config import TrainingConfig
from .label_vocabulary import SRLLabelVocabulary
from .run_metadata import RunMetadata, require_exact_run_metadata


CHECKPOINT_BUNDLE_VERSION = 1
LABEL_CONFIG_VERSION = 1
METADATA_FILENAME = "checkpoint_metadata.json"
LABEL_CONFIG_FILENAME = "labels.json"
STATE_DICT_FILENAME = "state_dict.pt"
_EXPECTED_MEMBERS = frozenset(
    {METADATA_FILENAME, LABEL_CONFIG_FILENAME, STATE_DICT_FILENAME}
)


@dataclass(frozen=True, slots=True)
class CheckpointLabelConfig:
    """Versioned immutable label order stored beside a model state dict."""

    label_config_version: int
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.label_config_version) is not int:
            raise TypeError("label_config_version must be an integer")
        if self.label_config_version != LABEL_CONFIG_VERSION:
            raise ValueError(
                f"label_config_version must be {LABEL_CONFIG_VERSION}"
            )
        SRLLabelVocabulary(labels=self.labels)

    @classmethod
    def from_labels(cls, labels: tuple[str, ...]) -> CheckpointLabelConfig:
        return cls(label_config_version=LABEL_CONFIG_VERSION, labels=labels)

    def to_dict(self) -> dict[str, object]:
        return {
            "label_config_version": self.label_config_version,
            "labels": list(self.labels),
        }

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_canonical_json_bytes(
        cls, payload: bytes
    ) -> CheckpointLabelConfig:
        if not isinstance(payload, bytes):
            raise TypeError("label configuration payload must be bytes")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("label configuration is not valid UTF-8") from error
        try:
            value = json.loads(text, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid label configuration JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise TypeError("label configuration must be a JSON object")
        required = {"label_config_version", "labels"}
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required)
            details: list[str] = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown keys: {', '.join(unknown)}")
            raise ValueError(
                "invalid label configuration keys; " + "; ".join(details)
            )
        labels = value["labels"]
        if not isinstance(labels, list):
            raise TypeError("label configuration labels must be a JSON array")
        result = cls(
            label_config_version=value["label_config_version"],
            labels=tuple(labels),
        )
        if result.canonical_json_bytes() != payload:
            raise ValueError("label configuration JSON is not in canonical form")
        return result


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Canonical run identity plus integrity hash of serialized state."""

    checkpoint_bundle_version: int
    run_metadata: RunMetadata
    state_dict_sha256: str

    def __post_init__(self) -> None:
        if type(self.checkpoint_bundle_version) is not int:
            raise TypeError("checkpoint_bundle_version must be an integer")
        if self.checkpoint_bundle_version != CHECKPOINT_BUNDLE_VERSION:
            raise ValueError(
                "checkpoint_bundle_version must be "
                f"{CHECKPOINT_BUNDLE_VERSION}"
            )
        if not isinstance(self.run_metadata, RunMetadata):
            raise TypeError("run_metadata must be RunMetadata")
        _validate_sha256(self.state_dict_sha256, field="state_dict_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_bundle_version": self.checkpoint_bundle_version,
            "run_metadata": self.run_metadata.to_dict(),
            "state_dict_sha256": self.state_dict_sha256,
        }

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> CheckpointMetadata:
        if not isinstance(payload, bytes):
            raise TypeError("checkpoint metadata payload must be bytes")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("checkpoint metadata is not valid UTF-8") from error
        try:
            value = json.loads(text, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid checkpoint metadata JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise TypeError("checkpoint metadata must be a JSON object")
        required = {
            "checkpoint_bundle_version",
            "run_metadata",
            "state_dict_sha256",
        }
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required)
            details: list[str] = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown keys: {', '.join(unknown)}")
            raise ValueError(
                "invalid checkpoint metadata keys; " + "; ".join(details)
            )
        result = cls(
            checkpoint_bundle_version=value["checkpoint_bundle_version"],
            run_metadata=RunMetadata.from_dict(value["run_metadata"]),
            state_dict_sha256=value["state_dict_sha256"],
        )
        if result.canonical_json_bytes() != payload:
            raise ValueError("checkpoint metadata JSON is not in canonical form")
        return result


@dataclass(frozen=True, slots=True)
class ValidatedCheckpointBundle:
    """A compatible checkpoint whose state file has not yet been loaded."""

    directory: Path
    metadata: RunMetadata
    label_config: CheckpointLabelConfig
    state_dict_path: Path
    state_dict_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedCheckpointBundle:
    """Validated checkpoint identity paired with callback-loaded state."""

    metadata: RunMetadata
    label_config: CheckpointLabelConfig
    state_dict: Any


def save_checkpoint_bundle(
    directory: str | Path,
    *,
    metadata: RunMetadata,
    config: TrainingConfig,
    label_config: CheckpointLabelConfig,
    state_dict: Any,
    save_state_dict: Callable[[Any, Path], None],
) -> Path:
    """Atomically save metadata, labels, and state through an injected callback.

    The callback has the same argument order as ``torch.save``: state first,
    destination path second.  This module never imports or interprets PyTorch.
    Existing destinations are refused rather than overwritten.
    """

    if not isinstance(metadata, RunMetadata):
        raise TypeError("metadata must be RunMetadata")
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig")
    metadata.assert_config_compatible(config)
    if not isinstance(label_config, CheckpointLabelConfig):
        raise TypeError("label_config must be CheckpointLabelConfig")
    if metadata.labels != label_config.labels:
        raise ValueError("run metadata labels do not match checkpoint labels")
    if not callable(save_state_dict):
        raise TypeError("save_state_dict must be callable")

    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"checkpoint destination already exists: {target}")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".partial", dir=parent)
    )
    try:
        (temporary / LABEL_CONFIG_FILENAME).write_bytes(
            label_config.canonical_json_bytes()
        )
        state_path = temporary / STATE_DICT_FILENAME
        save_state_dict(state_dict, state_path)
        _validate_regular_nonempty_file(state_path, STATE_DICT_FILENAME)
        checkpoint_metadata = CheckpointMetadata(
            checkpoint_bundle_version=CHECKPOINT_BUNDLE_VERSION,
            run_metadata=metadata,
            state_dict_sha256=_sha256_file(state_path),
        )
        (temporary / METADATA_FILENAME).write_bytes(
            checkpoint_metadata.canonical_json_bytes()
        )
        validate_checkpoint_bundle(
            temporary,
            expected_metadata=metadata,
            expected_config=config,
            expected_label_config=label_config,
        )
        temporary.replace(target)
    except Exception:
        if temporary.exists() and temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    return target


def validate_checkpoint_bundle(
    directory: str | Path,
    *,
    expected_metadata: RunMetadata,
    expected_config: TrainingConfig,
    expected_label_config: CheckpointLabelConfig,
) -> ValidatedCheckpointBundle:
    """Validate exact layout and compatibility without loading PyTorch."""

    if not isinstance(expected_metadata, RunMetadata):
        raise TypeError("expected_metadata must be RunMetadata")
    if not isinstance(expected_config, TrainingConfig):
        raise TypeError("expected_config must be TrainingConfig")
    expected_metadata.assert_config_compatible(expected_config)
    if not isinstance(expected_label_config, CheckpointLabelConfig):
        raise TypeError(
            "expected_label_config must be CheckpointLabelConfig"
        )
    source = Path(directory)
    _validate_layout(source)
    checkpoint_metadata = CheckpointMetadata.from_canonical_json_bytes(
        (source / METADATA_FILENAME).read_bytes()
    )
    metadata = checkpoint_metadata.run_metadata
    label_config = CheckpointLabelConfig.from_canonical_json_bytes(
        (source / LABEL_CONFIG_FILENAME).read_bytes()
    )
    if metadata.labels != label_config.labels:
        raise ValueError("checkpoint metadata labels do not match its label config")
    require_exact_run_metadata(metadata, expected_metadata)
    if label_config != expected_label_config:
        raise ValueError("checkpoint label configuration mismatch")
    if expected_metadata.labels != expected_label_config.labels:
        raise ValueError("expected metadata labels do not match expected labels")
    actual_state_digest = _sha256_file(source / STATE_DICT_FILENAME)
    if actual_state_digest != checkpoint_metadata.state_dict_sha256:
        raise ValueError("checkpoint state_dict SHA-256 mismatch")
    return ValidatedCheckpointBundle(
        directory=source,
        metadata=metadata,
        label_config=label_config,
        state_dict_path=source / STATE_DICT_FILENAME,
        state_dict_sha256=checkpoint_metadata.state_dict_sha256,
    )


def load_checkpoint_bundle(
    directory: str | Path,
    *,
    expected_metadata: RunMetadata,
    expected_config: TrainingConfig,
    expected_label_config: CheckpointLabelConfig,
    load_state_dict: Callable[[Path], Any],
) -> LoadedCheckpointBundle:
    """Load state only after exact metadata and label compatibility passes."""

    if not callable(load_state_dict):
        raise TypeError("load_state_dict must be callable")
    validated = validate_checkpoint_bundle(
        directory,
        expected_metadata=expected_metadata,
        expected_config=expected_config,
        expected_label_config=expected_label_config,
    )
    state_dict = load_state_dict(validated.state_dict_path)
    return LoadedCheckpointBundle(
        metadata=validated.metadata,
        label_config=validated.label_config,
        state_dict=state_dict,
    )


def _validate_layout(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"checkpoint path must be a real directory: {directory}")
    members = {path.name for path in directory.iterdir()}
    if members != _EXPECTED_MEMBERS:
        missing = sorted(_EXPECTED_MEMBERS - members)
        unknown = sorted(members - _EXPECTED_MEMBERS)
        details: list[str] = []
        if missing:
            details.append(f"missing files: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown files: {', '.join(unknown)}")
        raise ValueError("invalid checkpoint layout; " + "; ".join(details))
    for name in sorted(_EXPECTED_MEMBERS):
        path = directory / name
        _validate_regular_nonempty_file(path, name)


def _validate_regular_nonempty_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint member must be a regular file: {name}")
    if path.stat().st_size == 0:
        raise ValueError(f"checkpoint member cannot be empty: {name}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    valid = len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
    if not valid:
        raise ValueError(f"{field} must contain 64 lowercase hexadecimal digits")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result
