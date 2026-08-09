"""Checksum-gated download and safe archive extraction utilities."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
from urllib.request import urlopen
import zipfile

from .common import DatasetFormatError
from .registry import DatasetArtifact


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise DatasetFormatError(
            f"checksum mismatch for {Path(path).name}: expected "
            f"{expected_sha256}, found {actual}"
        )
    return actual


def download_verified(
    artifact: DatasetArtifact,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Stream an archive to disk and publish it only after verification."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_dir():
            raise IsADirectoryError(f"download destination is a directory: {target}")
        if sha256_file(target) == artifact.sha256:
            return target
        if not overwrite:
            raise FileExistsError(f"refusing to replace {target}")

    partial = target.with_name(f".{target.name}.partial")
    if partial.exists():
        partial.unlink()
    try:
        with urlopen(artifact.url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        verify_sha256(partial, artifact.sha256)
        partial.replace(target)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return target


def extract_verified(
    artifact: DatasetArtifact,
    archive_path: str | Path,
    destination: str | Path,
) -> Path:
    """Verify an archive and extract regular files without path traversal."""

    source = Path(archive_path)
    verify_sha256(source, artifact.sha256)
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"extraction directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    if artifact.archive_format == "tar":
        _extract_tar(source, target)
    elif artifact.archive_format == "zip":
        _extract_zip(source, target)
    else:
        raise DatasetFormatError(
            f"unsupported archive format: {artifact.archive_format}"
        )
    return target


def _member_target(root: Path, member_name: str) -> Path:
    normalized = PurePosixPath(member_name)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise DatasetFormatError(f"unsafe archive member: {member_name}")
    candidate = root.joinpath(*normalized.parts)
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    if os.path.commonpath((root_resolved, candidate_resolved)) != str(root_resolved):
        raise DatasetFormatError(f"unsafe archive member: {member_name}")
    return candidate


def _extract_tar(source: Path, target: Path) -> None:
    seen: set[Path] = set()
    with tarfile.open(source, "r:*") as archive:
        for member in archive.getmembers():
            output = _member_target(target, member.name)
            if output in seen:
                raise DatasetFormatError(f"duplicate archive member: {member.name}")
            seen.add(output)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise DatasetFormatError(
                    f"archive member is not a regular file: {member.name}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise DatasetFormatError(f"could not read archive member: {member.name}")
            with extracted, output.open("wb") as destination:
                shutil.copyfileobj(extracted, destination)


def _extract_zip(source: Path, target: Path) -> None:
    seen: set[Path] = set()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            output = _member_target(target, member.filename)
            if output in seen:
                raise DatasetFormatError(f"duplicate archive member: {member.filename}")
            seen.add(output)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise DatasetFormatError(
                    f"archive member is a symbolic link: {member.filename}"
                )
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as extracted, output.open("wb") as destination:
                shutil.copyfileobj(extracted, destination)
