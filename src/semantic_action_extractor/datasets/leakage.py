"""Cross-partition identity checks for joint verbal and nominal experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping
import uuid

from ..annotation_schema import AnnotationRecord
from .common import DatasetFormatError


_VALID_ROLES = {"train", "development", "test"}


@dataclass(frozen=True, slots=True)
class DatasetPartition:
    name: str
    role: str
    records: Iterable[AnnotationRecord]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("partition name cannot be empty")
        if self.role not in _VALID_ROLES:
            choices = ", ".join(sorted(_VALID_ROLES))
            raise ValueError(f"partition role must be one of: {choices}")


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    identity_type: str
    identity: str
    left_partition: str
    left_role: str
    right_partition: str
    right_role: str

    def to_dict(self) -> dict[str, str]:
        return {
            "identity_type": self.identity_type,
            "identity": self.identity,
            "left_partition": self.left_partition,
            "left_role": self.left_role,
            "right_partition": self.right_partition,
            "right_role": self.right_role,
        }


@dataclass(frozen=True, slots=True)
class LeakageReport:
    findings: tuple[LeakageFinding, ...]

    @property
    def is_clean(self) -> bool:
        return not self.findings

    def raise_if_found(self) -> None:
        if self.findings:
            first = self.findings[0]
            raise DatasetFormatError(
                "cross-partition leakage detected: "
                f"{first.identity_type} {first.identity} appears in "
                f"{first.left_partition} and {first.right_partition}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "is_clean": self.is_clean,
            "finding_count": len(self.findings),
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True, slots=True)
class RecordIdentity:
    """The release identities needed for split and contamination checks."""

    partition: str
    role: str
    dataset: str
    split: str
    source_id: str
    record_id: str
    document_id: str
    text_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "partition",
            "dataset",
            "split",
            "source_id",
            "record_id",
            "document_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"record identity {field_name} cannot be empty")
        if self.role not in _VALID_ROLES:
            raise ValueError("record identity role is unsupported")
        if (
            not isinstance(self.text_sha256, str)
            or len(self.text_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.text_sha256
            )
        ):
            raise ValueError("record identity text_sha256 must be lowercase SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "partition": self.partition,
            "role": self.role,
            "dataset": self.dataset,
            "split": self.split,
            "source_id": self.source_id,
            "record_id": self.record_id,
            "document_id": self.document_id,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True, slots=True)
class ContaminationTrigger:
    """One training record that links a training document to evaluation data."""

    reason: str
    training_record: RecordIdentity
    evaluation_records: tuple[RecordIdentity, ...]

    def __post_init__(self) -> None:
        if self.reason not in {"document_id_match", "exact_text_match"}:
            raise ValueError("unsupported contamination trigger reason")
        if self.training_record.role != "train":
            raise ValueError("contamination trigger must identify a training record")
        if not self.evaluation_records or any(
            record.role not in {"development", "test"}
            for record in self.evaluation_records
        ):
            raise ValueError(
                "contamination trigger must identify development or test records"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "training_record": self.training_record.to_dict(),
            "evaluation_records": [
                record.to_dict() for record in self.evaluation_records
            ],
        }


@dataclass(frozen=True, slots=True)
class TrainingQuarantineReport:
    """A deterministic whole-document exclusion policy for training data."""

    contaminated_document_ids: tuple[str, ...]
    triggers: tuple[ContaminationTrigger, ...]
    training_record_counts: Mapping[str, int]
    quarantined_record_counts: Mapping[str, int]
    source_artifacts: Mapping[str, str] = field(default_factory=dict)
    policy: str = "cross-role-document-quarantine-v1"

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.contaminated_document_ids))) != (
            self.contaminated_document_ids
        ):
            raise ValueError(
                "contaminated document IDs must be unique and sorted"
            )
        for label, counts in (
            ("training_record_counts", self.training_record_counts),
            ("quarantined_record_counts", self.quarantined_record_counts),
        ):
            if any(
                not isinstance(key, str)
                or not key.strip()
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in counts.items()
            ):
                raise ValueError(f"{label} must contain non-negative integer counts")
        if any(
            trigger.training_record.document_id
            not in self.contaminated_document_ids
            for trigger in self.triggers
        ):
            raise ValueError(
                "every contamination trigger must belong to a quarantined document"
            )
        if any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in self.source_artifacts.items()
        ):
            raise ValueError(
                "source_artifacts must map non-empty names to lowercase SHA-256 values"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "contaminated_document_count": len(self.contaminated_document_ids),
            "contaminated_document_ids": list(self.contaminated_document_ids),
            "trigger_count": len(self.triggers),
            "training_record_counts": dict(sorted(self.training_record_counts.items())),
            "quarantined_record_counts": dict(
                sorted(self.quarantined_record_counts.items())
            ),
            "source_artifacts": dict(sorted(self.source_artifacts.items())),
            "triggers": [trigger.to_dict() for trigger in self.triggers],
        }

    def write(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to replace {destination}")
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(
                f"quarantine report destination is a directory: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.partial"
        )
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        try:
            partial.write_text(f"{payload}\n", encoding="utf-8")
            partial.replace(destination)
        except Exception:
            if partial.exists() and partial.is_file():
                partial.unlink()
            raise
        return destination


def load_training_quarantine_report(
    path: str | Path,
) -> TrainingQuarantineReport:
    """Load and strictly validate a document-quarantine report."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DatasetFormatError(f"invalid quarantine report {source}: {error}") from error
    if not isinstance(raw, Mapping):
        raise DatasetFormatError("quarantine report must be a JSON object")
    expected = {
        "policy",
        "contaminated_document_count",
        "contaminated_document_ids",
        "trigger_count",
        "training_record_counts",
        "quarantined_record_counts",
        "source_artifacts",
        "triggers",
    }
    if set(raw) != expected:
        raise DatasetFormatError(
            "quarantine report fields differ from the schema; "
            f"missing={sorted(expected - set(raw))}, "
            f"extra={sorted(set(raw) - expected)}"
        )
    document_ids = _string_tuple(
        raw["contaminated_document_ids"],
        label="contaminated_document_ids",
    )
    triggers_raw = raw["triggers"]
    if not isinstance(triggers_raw, list):
        raise DatasetFormatError("quarantine report triggers must be a list")
    triggers = tuple(
        _contamination_trigger(item, index=index)
        for index, item in enumerate(triggers_raw)
    )
    try:
        report = TrainingQuarantineReport(
            policy=_required_string(raw["policy"], label="policy"),
            contaminated_document_ids=document_ids,
            triggers=triggers,
            training_record_counts=_nonnegative_counts(
                raw["training_record_counts"], label="training_record_counts"
            ),
            quarantined_record_counts=_nonnegative_counts(
                raw["quarantined_record_counts"], label="quarantined_record_counts"
            ),
            source_artifacts=_string_mapping(
                raw["source_artifacts"], label="source_artifacts"
            ),
        )
    except ValueError as error:
        raise DatasetFormatError(f"invalid quarantine report {source}: {error}") from error
    declared_document_count = _nonnegative_integer(
        raw["contaminated_document_count"],
        label="contaminated_document_count",
    )
    declared_trigger_count = _nonnegative_integer(
        raw["trigger_count"], label="trigger_count"
    )
    if declared_document_count != len(report.contaminated_document_ids):
        raise DatasetFormatError("quarantine report document count is inconsistent")
    if declared_trigger_count != len(report.triggers):
        raise DatasetFormatError("quarantine report trigger count is inconsistent")
    return report


def build_training_quarantine(
    training_partitions: Iterable[DatasetPartition],
    evaluation_partitions: Iterable[DatasetPartition],
) -> TrainingQuarantineReport:
    """Find copied evaluation text and quarantine its full training document.

    Evaluation partitions are indexed first and are never altered. A training
    document is quarantined when one of its records shares either a document ID
    or an exact adapter-canonicalized text hash with development or test data.
    The input streams are consumed once; callers can reopen training streams to
    apply the returned policy.
    """

    training = tuple(training_partitions)
    evaluation = tuple(evaluation_partitions)
    all_partitions = training + evaluation
    if len({partition.name for partition in all_partitions}) != len(all_partitions):
        raise ValueError("partition names must be unique")
    if not evaluation:
        raise ValueError("at least one evaluation partition is required")
    if any(partition.role != "train" for partition in training):
        raise ValueError("training partitions must use the train role")
    if any(partition.role not in {"development", "test"} for partition in evaluation):
        raise ValueError("evaluation partitions must use development or test roles")

    evaluation_identities: list[RecordIdentity] = []
    for partition in evaluation:
        evaluation_identities.extend(_record_identities(partition))
    _raise_on_evaluation_overlap(evaluation_identities)

    by_document: dict[str, list[RecordIdentity]] = {}
    by_text: dict[str, list[RecordIdentity]] = {}
    for identity in evaluation_identities:
        by_document.setdefault(identity.document_id, []).append(identity)
        by_text.setdefault(identity.text_sha256, []).append(identity)

    triggers: list[ContaminationTrigger] = []
    contaminated_documents: set[str] = set()
    document_counts: dict[str, dict[str, int]] = {}
    for partition in training:
        local_document_counts = document_counts.setdefault(partition.name, {})
        for identity in _record_identities(partition):
            local_document_counts[identity.document_id] = (
                local_document_counts.get(identity.document_id, 0) + 1
            )
            reasons = (
                ("document_id_match", by_document.get(identity.document_id, [])),
                ("exact_text_match", by_text.get(identity.text_sha256, [])),
            )
            for reason, matches in reasons:
                if not matches:
                    continue
                contaminated_documents.add(identity.document_id)
                triggers.append(
                    ContaminationTrigger(
                        reason=reason,
                        training_record=identity,
                        evaluation_records=tuple(sorted(matches, key=_identity_key)),
                    )
                )

    triggers.sort(
        key=lambda item: (
            item.training_record.document_id,
            item.training_record.partition,
            item.training_record.record_id,
            item.reason,
        )
    )
    return TrainingQuarantineReport(
        contaminated_document_ids=tuple(sorted(contaminated_documents)),
        triggers=tuple(triggers),
        training_record_counts={
            partition: sum(counts.values())
            for partition, counts in sorted(document_counts.items())
        },
        quarantined_record_counts={
            partition: sum(
                count
                for document_id, count in counts.items()
                if document_id in contaminated_documents
            )
            for partition, counts in sorted(document_counts.items())
        },
    )


def iter_after_document_quarantine(
    records: Iterable[AnnotationRecord],
    report: TrainingQuarantineReport,
    *,
    on_exclude: Callable[[AnnotationRecord], None] | None = None,
) -> Iterable[AnnotationRecord]:
    """Yield training records outside quarantined documents."""

    blocked = frozenset(report.contaminated_document_ids)
    for record in records:
        if record.provenance.split != "train":
            raise DatasetFormatError(
                "document quarantine can be applied only to training records"
            )
        document_id = record.provenance.document_id
        if document_id is None:
            raise DatasetFormatError(
                f"record {record.provenance.record_id} has no document ID for "
                "document quarantine"
            )
        if document_id in blocked:
            if on_exclude is not None:
                on_exclude(record)
            continue
        yield record


def check_cross_partition_leakage(
    partitions: Iterable[DatasetPartition],
    *,
    require_document_ids: bool = True,
    include_predicate_families: bool = False,
) -> LeakageReport:
    """Find shared identities only when they cross train/dev/test roles.

    Sharing a QA-SRL sentence between two datasets assigned to the same role is
    permitted.  Reusing it across different roles is reported.
    """

    partition_list = tuple(partitions)
    if len({partition.name for partition in partition_list}) != len(partition_list):
        raise ValueError("partition names must be unique")

    indexes: dict[str, dict[str, dict[str, str]]] = {}
    for partition in partition_list:
        indexes[partition.name] = _partition_index(
            partition,
            require_document_ids=require_document_ids,
            include_predicate_families=include_predicate_families,
        )

    findings: list[LeakageFinding] = []
    for left_index, left in enumerate(partition_list):
        for right in partition_list[left_index + 1 :]:
            if left.role == right.role:
                continue
            left_markers = indexes[left.name]
            right_markers = indexes[right.name]
            for identity_type in sorted(left_markers.keys() & right_markers.keys()):
                shared = sorted(
                    left_markers[identity_type].keys()
                    & right_markers[identity_type].keys()
                )
                for identity in shared:
                    findings.append(
                        LeakageFinding(
                            identity_type=identity_type,
                            identity=identity,
                            left_partition=left.name,
                            left_role=left.role,
                            right_partition=right.name,
                            right_role=right.role,
                        )
                    )
    return LeakageReport(findings=tuple(findings))


def _partition_index(
    partition: DatasetPartition,
    *,
    require_document_ids: bool,
    include_predicate_families: bool,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {
        "document_id": {},
        "source_id": {},
        "text_sha256": {},
        "derived_lineage": {},
    }
    if include_predicate_families:
        result["predicate_family"] = {}

    record_ids: set[str] = set()
    for record in partition.records:
        record_id = record.provenance.record_id
        if record_id in record_ids:
            raise DatasetFormatError(
                f"duplicate record ID in {partition.name}: {record_id}"
            )
        record_ids.add(record_id)

        document_id = record.provenance.document_id
        if require_document_ids and document_id is None:
            raise DatasetFormatError(
                f"record {record_id} has no document ID for leakage checks"
            )
        if document_id is not None:
            result["document_id"].setdefault(document_id, record_id)
        result["source_id"].setdefault(record.provenance.source_id, record_id)
        text_digest = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
        result["text_sha256"].setdefault(text_digest, record_id)

        for lineage_id in _lineage_ids(record.metadata):
            result["derived_lineage"].setdefault(lineage_id, record_id)
        if include_predicate_families:
            for candidate in record.candidates:
                family = (
                    candidate.related_verbal_form
                    or candidate.lemma
                ).casefold()
                result["predicate_family"].setdefault(family, record_id)
    return result


def _record_identities(partition: DatasetPartition) -> Iterable[RecordIdentity]:
    record_ids: set[str] = set()
    for record in partition.records:
        record_id = record.provenance.record_id
        if record_id in record_ids:
            raise DatasetFormatError(
                f"duplicate record ID in {partition.name}: {record_id}"
            )
        record_ids.add(record_id)
        document_id = record.provenance.document_id
        if document_id is None:
            raise DatasetFormatError(
                f"record {record_id} has no document ID for leakage checks"
            )
        yield RecordIdentity(
            partition=partition.name,
            role=partition.role,
            dataset=record.provenance.dataset,
            split=record.provenance.split,
            source_id=record.provenance.source_id,
            record_id=record_id,
            document_id=document_id,
            text_sha256=hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
        )


def _raise_on_evaluation_overlap(
    identities: Iterable[RecordIdentity],
) -> None:
    """Fail if document, source, or exact text crosses dev and test."""

    seen: dict[tuple[str, str], RecordIdentity] = {}
    for identity in identities:
        markers = (
            ("document_id", identity.document_id),
            ("source_id", identity.source_id),
            ("text_sha256", identity.text_sha256),
        )
        for identity_type, value in markers:
            key = (identity_type, value)
            previous = seen.get(key)
            if previous is not None and previous.role != identity.role:
                raise DatasetFormatError(
                    "development/test leakage detected: "
                    f"{identity_type} {value} appears in {previous.partition} "
                    f"and {identity.partition}"
                )
            seen.setdefault(key, identity)


def _identity_key(identity: RecordIdentity) -> tuple[str, ...]:
    return (
        identity.role,
        identity.partition,
        identity.dataset,
        identity.record_id,
    )


def _contamination_trigger(raw: object, *, index: int) -> ContaminationTrigger:
    label = f"triggers[{index}]"
    if not isinstance(raw, Mapping) or set(raw) != {
        "reason",
        "training_record",
        "evaluation_records",
    }:
        raise DatasetFormatError(f"{label} fields differ from the schema")
    evaluation_raw = raw["evaluation_records"]
    if not isinstance(evaluation_raw, list):
        raise DatasetFormatError(f"{label}.evaluation_records must be a list")
    evaluation = tuple(
        _record_identity(item, label=f"{label}.evaluation_records[{position}]")
        for position, item in enumerate(evaluation_raw)
    )
    if not evaluation:
        raise DatasetFormatError(f"{label} must identify evaluation records")
    try:
        return ContaminationTrigger(
            reason=_required_string(raw["reason"], label=f"{label}.reason"),
            training_record=_record_identity(
                raw["training_record"], label=f"{label}.training_record"
            ),
            evaluation_records=evaluation,
        )
    except ValueError as error:
        raise DatasetFormatError(f"invalid {label}: {error}") from error


def _record_identity(raw: object, *, label: str) -> RecordIdentity:
    expected = {
        "partition",
        "role",
        "dataset",
        "split",
        "source_id",
        "record_id",
        "document_id",
        "text_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise DatasetFormatError(f"{label} fields differ from the schema")
    values = {
        key: _required_string(raw[key], label=f"{label}.{key}")
        for key in expected
    }
    try:
        return RecordIdentity(**values)
    except ValueError as error:
        raise DatasetFormatError(f"invalid {label}: {error}") from error


def _required_string(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise DatasetFormatError(f"{label} must be a non-empty string")
    return raw


def _string_tuple(raw: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise DatasetFormatError(f"{label} must be a list")
    return tuple(
        _required_string(value, label=f"{label}[{index}]")
        for index, value in enumerate(raw)
    )


def _nonnegative_integer(raw: object, *, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise DatasetFormatError(f"{label} must be a non-negative integer")
    return raw


def _nonnegative_counts(raw: object, *, label: str) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise DatasetFormatError(f"{label} must be an object")
    return {
        _required_string(key, label=f"{label} key"): _nonnegative_integer(
            value, label=f"{label}.{key}"
        )
        for key, value in raw.items()
    }


def _string_mapping(raw: object, *, label: str) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise DatasetFormatError(f"{label} must be an object")
    return {
        _required_string(key, label=f"{label} key"): _required_string(
            value, label=f"{label}.{key}"
        )
        for key, value in raw.items()
    }


def _lineage_ids(metadata: Mapping[str, object]) -> tuple[str, ...]:
    raw = metadata.get("derived_from")
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)) and all(
        isinstance(value, str) and value for value in raw
    ):
        return tuple(raw)
    raise DatasetFormatError("record derived_from metadata must contain string IDs")
