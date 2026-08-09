"""Streaming writer for adapted records and their deterministic manifest."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import uuid

from ..annotation_schema import AnnotationRecord
from .common import DatasetFormatError
from .io import sha256_file
from .manifest import ManifestBuilder, PreparationManifest


def write_adapted_jsonl(
    records: Iterable[AnnotationRecord],
    output_path: str | Path,
    manifest_path: str | Path,
    builder: ManifestBuilder,
    *,
    overwrite: bool = False,
) -> PreparationManifest:
    """Write canonical JSONL and its manifest without retaining corpus data."""

    output = Path(output_path)
    manifest_output = Path(manifest_path)
    if output.resolve() == manifest_output.resolve():
        raise ValueError("adapted records and manifest must use different paths")
    for destination in (output, manifest_output):
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to replace {destination}")
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(f"output destination is a directory: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    transaction_id = uuid.uuid4().hex
    partial = output.with_name(f".{output.name}.{transaction_id}.partial")
    manifest_partial = manifest_output.with_name(
        f".{manifest_output.name}.{transaction_id}.partial"
    )
    record_count = 0
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                builder.add(record)
                serialized = json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                handle.write(serialized)
                handle.write("\n")
                record_count += 1
        if not record_count:
            raise DatasetFormatError("adapter produced no records")
        manifest = builder.finish()
        if sha256_file(partial) != manifest.record_fingerprint:
            raise DatasetFormatError(
                "adapted JSONL checksum does not match the record fingerprint"
            )
        manifest.write(manifest_partial)
        _publish_pair(
            partial,
            output,
            manifest_partial,
            manifest_output,
            transaction_id=transaction_id,
        )
    except Exception:
        for staged in (partial, manifest_partial):
            if staged.exists() and staged.is_file():
                staged.unlink()
        raise
    return manifest


def _publish_pair(
    staged_data: Path,
    data_output: Path,
    staged_manifest: Path,
    manifest_output: Path,
    *,
    transaction_id: str,
) -> None:
    """Publish data and manifest together, restoring prior files on failure."""

    data_backup = data_output.with_name(
        f".{data_output.name}.{transaction_id}.backup"
    )
    manifest_backup = manifest_output.with_name(
        f".{manifest_output.name}.{transaction_id}.backup"
    )
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, backup in (
            (data_output, data_backup),
            (manifest_output, manifest_backup),
        ):
            if destination.exists():
                destination.replace(backup)
                backups.append((destination, backup))
        staged_data.replace(data_output)
        published.append(data_output)
        staged_manifest.replace(manifest_output)
        published.append(manifest_output)
    except Exception:
        for destination in reversed(published):
            if destination.exists() and destination.is_file():
                destination.unlink()
        for destination, backup in reversed(backups):
            if backup.exists():
                backup.replace(destination)
        raise
    else:
        for _, backup in backups:
            if backup.exists():
                backup.unlink()
