"""Canonical immutable provenance for supplied-predicate SRL runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

from .experiment_config import (
    DEVICE_REQUESTS,
    EXPERIMENT_VARIANTS,
    TrainingConfig,
)
from .label_vocabulary import SRLLabelVocabulary


RUN_METADATA_VERSION = 1

_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_KEY_RE = re.compile(r"[a-z][a-z0-9_]*")
_PACKAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RESOLVED_DEVICE_RE = re.compile(r"(?:cpu|mps|cuda(?::[0-9]+)?)")
_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z"
)
_REQUIRED_KEYS = frozenset(
    {
        "metadata_version",
        "git_revision",
        "config_digest",
        "dataset_fingerprint",
        "labels",
        "package_versions",
        "hardware",
        "started_at",
        "recorded_at",
        "seed",
        "variant",
        "requested_device",
        "resolved_device",
        "counts",
        "drop_stats",
        "initial_state_fingerprint",
    }
)


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field} must be nonempty without surrounding whitespace")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = _require_string(value, field=field)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must contain 64 lowercase hexadecimal digits")
    return digest


def _parse_timestamp(value: object, *, field: str) -> datetime:
    timestamp = _require_string(value, field=field)
    if _UTC_TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{field} must be a valid ISO-8601 UTC timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{field} must be in UTC")
    return parsed


def _freeze_string_mapping(
    value: Mapping[str, str],
    *,
    field: str,
    key_pattern: re.Pattern[str],
    require_nonempty: bool,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    pairs: list[tuple[str, str]] = []
    for key, item in value.items():
        if not isinstance(key, str) or key_pattern.fullmatch(key) is None:
            raise ValueError(f"{field} contains an invalid key: {key!r}")
        pairs.append((key, _require_string(item, field=f"{field}.{key}")))
    if require_nonempty and not pairs:
        raise ValueError(f"{field} cannot be empty")
    return tuple(sorted(pairs))


def _freeze_count_mapping(
    value: Mapping[str, int],
    *,
    field: str,
    require_nonempty: bool,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    pairs: list[tuple[str, int]] = []
    for key, item in value.items():
        if not isinstance(key, str) or _KEY_RE.fullmatch(key) is None:
            raise ValueError(f"{field} contains an invalid key: {key!r}")
        if type(item) is not int or item < 0:
            raise ValueError(
                f"{field}.{key} must be a non-negative integer"
            )
        pairs.append((key, item))
    if require_nonempty and not pairs:
        raise ValueError(f"{field} cannot be empty")
    return tuple(sorted(pairs))


def _validate_frozen_pairs(
    value: object,
    *,
    field: str,
    value_type: type,
    key_pattern: re.Pattern[str],
    require_nonempty: bool,
) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be an immutable tuple of pairs")
    if require_nonempty and not value:
        raise ValueError(f"{field} cannot be empty")
    keys: list[str] = []
    for pair in value:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{field} entries must be two-item tuples")
        key, item = pair
        if not isinstance(key, str) or key_pattern.fullmatch(key) is None:
            raise ValueError(f"{field} contains an invalid key: {key!r}")
        if value_type is str:
            _require_string(item, field=f"{field}.{key}")
        elif type(item) is not int or item < 0:
            raise ValueError(
                f"{field}.{key} must be a non-negative integer"
            )
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field} keys must be unique")
    if tuple(keys) != tuple(sorted(keys)):
        raise ValueError(f"{field} keys must be in canonical sorted order")


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Immutable provenance sufficient to identify one experiment run."""

    metadata_version: int
    git_revision: str
    config_digest: str
    dataset_fingerprint: str
    labels: tuple[str, ...]
    package_versions: tuple[tuple[str, str], ...]
    hardware: tuple[tuple[str, str], ...]
    started_at: str
    recorded_at: str
    seed: int
    variant: str
    requested_device: str
    resolved_device: str
    counts: tuple[tuple[str, int], ...]
    drop_stats: tuple[tuple[str, int], ...]
    initial_state_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.metadata_version) is not int:
            raise TypeError("metadata_version must be an integer")
        if self.metadata_version != RUN_METADATA_VERSION:
            raise ValueError(
                f"metadata_version must be {RUN_METADATA_VERSION}"
            )
        revision = _require_string(self.git_revision, field="git_revision")
        if _GIT_REVISION_RE.fullmatch(revision) is None:
            raise ValueError(
                "git_revision must be a 40-character lowercase commit"
            )
        _require_sha256(self.config_digest, field="config_digest")
        _require_sha256(
            self.dataset_fingerprint, field="dataset_fingerprint"
        )
        if not isinstance(self.labels, tuple):
            raise TypeError("labels must be a tuple")
        SRLLabelVocabulary(labels=self.labels)
        _validate_frozen_pairs(
            self.package_versions,
            field="package_versions",
            value_type=str,
            key_pattern=_PACKAGE_RE,
            require_nonempty=True,
        )
        _validate_frozen_pairs(
            self.hardware,
            field="hardware",
            value_type=str,
            key_pattern=_KEY_RE,
            require_nonempty=True,
        )
        started = _parse_timestamp(self.started_at, field="started_at")
        recorded = _parse_timestamp(self.recorded_at, field="recorded_at")
        if recorded < started:
            raise ValueError("recorded_at cannot precede started_at")
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= 0xFFFFFFFF:
            raise ValueError("seed must be between 0 and 4294967295")
        if self.variant not in EXPERIMENT_VARIANTS:
            raise ValueError(
                "variant must be 'predicate_signal' or 'no_predicate_signal'"
            )
        if self.requested_device not in DEVICE_REQUESTS:
            raise ValueError("requested_device must be auto, cpu, mps, or cuda")
        resolved_device = _require_string(
            self.resolved_device, field="resolved_device"
        )
        if _RESOLVED_DEVICE_RE.fullmatch(resolved_device) is None:
            raise ValueError(
                "resolved_device must be cpu, mps, cuda, or cuda:<index>"
            )
        resolved_kind = resolved_device.partition(":")[0]
        if (
            self.requested_device != "auto"
            and resolved_kind != self.requested_device
        ):
            raise ValueError(
                "resolved_device must satisfy the explicit requested_device"
            )
        _validate_frozen_pairs(
            self.counts,
            field="counts",
            value_type=int,
            key_pattern=_KEY_RE,
            require_nonempty=True,
        )
        _validate_frozen_pairs(
            self.drop_stats,
            field="drop_stats",
            value_type=int,
            key_pattern=_KEY_RE,
            require_nonempty=False,
        )
        _require_sha256(
            self.initial_state_fingerprint,
            field="initial_state_fingerprint",
        )

    @classmethod
    def create(
        cls,
        *,
        git_revision: str,
        config_digest: str,
        dataset_fingerprint: str,
        labels: tuple[str, ...],
        package_versions: Mapping[str, str],
        hardware: Mapping[str, str],
        started_at: str,
        recorded_at: str,
        seed: int,
        variant: str,
        requested_device: str,
        resolved_device: str,
        counts: Mapping[str, int],
        drop_stats: Mapping[str, int],
        initial_state_fingerprint: str,
    ) -> RunMetadata:
        """Freeze caller-supplied mappings into canonical sorted tuples."""

        return cls(
            metadata_version=RUN_METADATA_VERSION,
            git_revision=git_revision,
            config_digest=config_digest,
            dataset_fingerprint=dataset_fingerprint,
            labels=labels,
            package_versions=_freeze_string_mapping(
                package_versions,
                field="package_versions",
                key_pattern=_PACKAGE_RE,
                require_nonempty=True,
            ),
            hardware=_freeze_string_mapping(
                hardware,
                field="hardware",
                key_pattern=_KEY_RE,
                require_nonempty=True,
            ),
            started_at=started_at,
            recorded_at=recorded_at,
            seed=seed,
            variant=variant,
            requested_device=requested_device,
            resolved_device=resolved_device,
            counts=_freeze_count_mapping(
                counts, field="counts", require_nonempty=True
            ),
            drop_stats=_freeze_count_mapping(
                drop_stats, field="drop_stats", require_nonempty=False
            ),
            initial_state_fingerprint=initial_state_fingerprint,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the complete metadata as plain canonical JSON values."""

        return {
            "metadata_version": self.metadata_version,
            "git_revision": self.git_revision,
            "config_digest": self.config_digest,
            "dataset_fingerprint": self.dataset_fingerprint,
            "labels": list(self.labels),
            "package_versions": dict(self.package_versions),
            "hardware": dict(self.hardware),
            "started_at": self.started_at,
            "recorded_at": self.recorded_at,
            "seed": self.seed,
            "variant": self.variant,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "counts": dict(self.counts),
            "drop_stats": dict(self.drop_stats),
            "initial_state_fingerprint": self.initial_state_fingerprint,
        }

    def canonical_json_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON without insignificant whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @property
    def digest(self) -> str:
        """SHA-256 of :meth:`canonical_json_bytes`."""

        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    def assert_config_compatible(self, config: TrainingConfig) -> None:
        """Require every run-defining configuration field to match."""

        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig")
        mismatches: list[str] = []
        if self.config_digest != config.digest:
            mismatches.append("config_digest")
        if self.dataset_fingerprint != config.prepared_data_fingerprint:
            mismatches.append("dataset_fingerprint")
        if self.variant != config.variant:
            mismatches.append("variant")
        if self.seed not in config.paired_seeds:
            mismatches.append("seed")
        if self.requested_device != config.device_request:
            mismatches.append("requested_device")
        if mismatches:
            raise ValueError(
                "run metadata is incompatible with training config: "
                + ", ".join(mismatches)
            )

    @classmethod
    def from_dict(cls, value: object) -> RunMetadata:
        """Parse a strict plain-object metadata representation."""

        if not isinstance(value, dict):
            raise TypeError("run metadata must be a JSON object")
        actual_keys = frozenset(value)
        missing = sorted(_REQUIRED_KEYS - actual_keys)
        unknown = sorted(actual_keys - _REQUIRED_KEYS)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown keys: {', '.join(unknown)}")
            raise ValueError("invalid run metadata keys; " + "; ".join(details))

        if type(value["metadata_version"]) is not int:
            raise TypeError("metadata_version must be an integer")
        if value["metadata_version"] != RUN_METADATA_VERSION:
            raise ValueError(
                f"metadata_version must be {RUN_METADATA_VERSION}"
            )

        labels = value["labels"]
        if not isinstance(labels, list):
            raise TypeError("labels must be a JSON array")
        return cls.create(
            git_revision=value["git_revision"],
            config_digest=value["config_digest"],
            dataset_fingerprint=value["dataset_fingerprint"],
            labels=tuple(labels),
            package_versions=_require_json_mapping(
                value["package_versions"], field="package_versions"
            ),
            hardware=_require_json_mapping(value["hardware"], field="hardware"),
            started_at=value["started_at"],
            recorded_at=value["recorded_at"],
            seed=value["seed"],
            variant=value["variant"],
            requested_device=value["requested_device"],
            resolved_device=value["resolved_device"],
            counts=_require_json_mapping(value["counts"], field="counts"),
            drop_stats=_require_json_mapping(
                value["drop_stats"], field="drop_stats"
            ),
            initial_state_fingerprint=value["initial_state_fingerprint"],
        )

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> RunMetadata:
        """Parse canonical JSON and reject alternate or ambiguous encodings."""

        if not isinstance(payload, bytes):
            raise TypeError("run metadata payload must be bytes")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("run metadata is not valid UTF-8") from error
        try:
            raw = json.loads(text, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid run metadata JSON: {error.msg}") from error
        metadata = cls.from_dict(raw)
        if metadata.canonical_json_bytes() != payload:
            raise ValueError("run metadata JSON is not in canonical form")
        return metadata


def _require_json_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def require_exact_run_metadata(
    actual: RunMetadata, expected: RunMetadata
) -> None:
    """Reject any metadata difference before checkpoint state is loaded."""

    if not isinstance(actual, RunMetadata) or not isinstance(expected, RunMetadata):
        raise TypeError("actual and expected must be RunMetadata")
    mismatches = [
        field.name
        for field in fields(RunMetadata)
        if getattr(actual, field.name) != getattr(expected, field.name)
    ]
    if mismatches:
        raise ValueError(
            "run metadata mismatch: " + ", ".join(mismatches)
        )
