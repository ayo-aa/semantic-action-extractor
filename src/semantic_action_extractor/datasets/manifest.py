"""Deterministic preparation manifests for adapted annotation records."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ..annotation_schema import ANNOTATION_SCHEMA_VERSION, AnnotationRecord
from .common import DatasetFormatError


MANIFEST_VERSION = "1.0.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SourceArtifactIdentity:
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("source artifact name cannot be empty")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.sha256
        ):
            raise ValueError(
                "source artifact sha256 must contain 64 lowercase hex digits"
            )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PreparationManifest:
    dataset: str
    release: str
    split: str
    adapter: str
    adapter_version: str
    source_artifacts: tuple[SourceArtifactIdentity, ...]
    counts: Mapping[str, int]
    anomalies: Mapping[str, int]
    exclusions: Mapping[str, int]
    record_fingerprint: str
    schema_version: str = ANNOTATION_SCHEMA_VERSION
    manifest_version: str = MANIFEST_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "dataset",
            "release",
            "split",
            "adapter",
            "adapter_version",
            "schema_version",
            "manifest_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"manifest {field_name} cannot be empty")
        if not self.source_artifacts:
            raise ValueError("manifest must identify at least one source artifact")
        artifact_names = [artifact.name for artifact in self.source_artifacts]
        if len(artifact_names) != len(set(artifact_names)):
            raise ValueError("manifest source artifact names must be unique")
        for label, counts in (
            ("counts", self.counts),
            ("anomalies", self.anomalies),
            ("exclusions", self.exclusions),
        ):
            _validate_counts(counts, label=label)
        if not isinstance(
            self.record_fingerprint, str
        ) or not _SHA256_PATTERN.fullmatch(self.record_fingerprint):
            raise ValueError(
                "manifest record_fingerprint must contain 64 lowercase hex digits"
            )
        if self.schema_version != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version must be {ANNOTATION_SCHEMA_VERSION}"
            )
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(
                f"manifest_version must be {MANIFEST_VERSION}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "release": self.release,
            "split": self.split,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "source_artifacts": [item.to_dict() for item in self.source_artifacts],
            "counts": dict(sorted(self.counts.items())),
            "anomalies": dict(sorted(self.anomalies.items())),
            "exclusions": dict(sorted(self.exclusions.items())),
            "record_fingerprint": self.record_fingerprint,
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial")
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        try:
            temporary.write_text(f"{payload}\n", encoding="utf-8")
            temporary.replace(destination)
        except Exception:
            if temporary.exists() and temporary.is_file():
                temporary.unlink()
            raise
        return destination


@dataclass(slots=True)
class ManifestBuilder:
    dataset: str
    release: str
    split: str
    adapter: str
    adapter_version: str
    source_artifacts: tuple[SourceArtifactIdentity, ...]
    exclusions: dict[str, int] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict, init=False)
    _anomalies: dict[str, int] = field(default_factory=dict, init=False)
    _record_ids: set[str] = field(default_factory=set, init=False)
    _digest: Any = field(default_factory=hashlib.sha256, init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name in (
            "dataset",
            "release",
            "split",
            "adapter",
            "adapter_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"manifest builder {field_name} cannot be empty")
        if not self.source_artifacts:
            raise ValueError(
                "manifest builder must identify at least one source artifact"
            )
        artifact_names = [artifact.name for artifact in self.source_artifacts]
        if len(artifact_names) != len(set(artifact_names)):
            raise ValueError("manifest source artifact names must be unique")
        _validate_counts(self.exclusions, label="exclusions")
        self._counts.update(
            {
                "records": 0,
                "candidates": 0,
                "verbal_candidates": 0,
                "nominal_candidates": 0,
                "positive_eventive_nominals": 0,
                "questions": 0,
                "judgments": 0,
                "valid_judgments": 0,
                "invalid_judgments": 0,
                "answer_alternatives": 0,
                "answer_spans": 0,
            }
        )

    def add(self, record: AnnotationRecord) -> None:
        if record.provenance.dataset != self.dataset:
            raise DatasetFormatError("record dataset does not match manifest dataset")
        if record.provenance.release != self.release:
            raise DatasetFormatError("record release does not match manifest release")
        if record.provenance.split != self.split:
            raise DatasetFormatError("record split does not match manifest split")
        record_id = record.provenance.record_id
        if record_id in self._record_ids:
            raise DatasetFormatError(f"duplicate adapted record ID: {record_id}")
        self._record_ids.add(record_id)

        canonical = json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._digest.update(canonical)
        self._digest.update(b"\n")

        self._counts["records"] += 1
        raw_anomalies = record.metadata.get("upstream_anomaly_counts", {})
        if isinstance(raw_anomalies, Mapping):
            for key, value in raw_anomalies.items():
                if isinstance(key, str) and isinstance(value, int) and not isinstance(
                    value, bool
                ):
                    self._add_anomaly(key, value)
        duplicate_rows = record.metadata.get("duplicate_qa_row_count", 0)
        if isinstance(duplicate_rows, int) and not isinstance(duplicate_rows, bool):
            self._add_anomaly("exact_duplicate_qa_rows", duplicate_rows)
        artifact_counts = record.metadata.get(
            "nonempty_release_artifact_value_counts", {}
        )
        if isinstance(artifact_counts, Mapping):
            for key, value in artifact_counts.items():
                if isinstance(key, str) and isinstance(value, int) and not isinstance(
                    value, bool
                ):
                    self._add_anomaly(f"discarded_{key}_values", value)
        for candidate in record.candidates:
            self._counts["candidates"] += 1
            key = f"{candidate.predicate_type}_candidates"
            self._counts[key] += 1
            if candidate.predicate_type == "nominal" and any(
                judgment.is_eventive
                for judgment in candidate.eventivity_judgments
            ):
                self._counts["positive_eventive_nominals"] += 1
            if candidate.metadata.get("upstream_eventivity_question_conflict") is True:
                self._add_anomaly("eventivity_question_conflicts", 1)
            if candidate.metadata.get("noun_matches_target_case_sensitive") is False:
                self._add_anomaly("case_normalized_noun_fields", 1)
            for question in candidate.questions:
                self._counts["questions"] += 1
                for judgment in question.judgments:
                    self._counts["judgments"] += 1
                    judgment_key = (
                        "valid_judgments" if judgment.is_valid else "invalid_judgments"
                    )
                    self._counts[judgment_key] += 1
                    for answer in judgment.answers:
                        self._counts["answer_alternatives"] += 1
                        self._counts["answer_spans"] += len(answer.spans)
                        if answer.metadata.get("upstream_answer_text_missing") is True:
                            self._add_anomaly("reconstructed_missing_answer_text", 1)

    def _add_anomaly(self, key: str, value: int) -> None:
        if value < 0:
            raise DatasetFormatError("manifest anomaly counts cannot be negative")
        if value == 0:
            return
        self._anomalies[key] = self._anomalies.get(key, 0) + value

    def add_exclusion(self, key: str, value: int = 1) -> None:
        """Record an intentional preparation exclusion by named policy."""

        if not isinstance(key, str) or not key.strip():
            raise DatasetFormatError("manifest exclusion key cannot be empty")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DatasetFormatError(
                "manifest exclusion counts must be non-negative integers"
            )
        if value == 0:
            return
        self.exclusions[key] = self.exclusions.get(key, 0) + value

    def finish(self) -> PreparationManifest:
        return PreparationManifest(
            dataset=self.dataset,
            release=self.release,
            split=self.split,
            adapter=self.adapter,
            adapter_version=self.adapter_version,
            source_artifacts=self.source_artifacts,
            counts=dict(self._counts),
            anomalies=dict(self._anomalies),
            exclusions=dict(self.exclusions),
            record_fingerprint=self._digest.hexdigest(),
        )


def _validate_counts(counts: Mapping[str, int], *, label: str) -> None:
    if not isinstance(counts, Mapping):
        raise ValueError(f"manifest {label} must be a mapping")
    for key, value in counts.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"manifest {label} keys cannot be empty")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"manifest {label} values must be non-negative integers"
            )
