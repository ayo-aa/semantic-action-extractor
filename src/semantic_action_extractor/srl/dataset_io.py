"""Deterministic persistence for source-neutral word-level SRL datasets.

The on-disk boundary contains only final :class:`WordLevelSRLExample`
records.  A dataset is represented by three canonical JSONL files and a
manifest::

    train.jsonl
    development.jsonl
    test.jsonl
    manifest.json

Every JSONL row carries the schema version so that a split file can never be
interpreted without knowing its record contract.  The manifest pins each
file's SHA-256 digest and a dataset fingerprint.  The fingerprint is the
SHA-256 of the schema version followed by the ordered ``filename + digest``
pairs in train/development/test order.

Files are replaced atomically and the manifest is replaced last.  A process
interrupted between replacements can therefore leave a snapshot that fails
manifest verification, but cannot leave a partially written file that is
accepted as valid.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .example import DatasetSplit, WordLevelSRLExample


DATASET_SCHEMA_VERSION = "semantic-action-extractor.word-level-srl/v1"
MANIFEST_FILENAME = "manifest.json"

_SPLIT_FILES: tuple[tuple[DatasetSplit, str], ...] = (
    ("train", "train.jsonl"),
    ("development", "development.jsonl"),
    ("test", "test.jsonl"),
)
_VALID_SPLITS = frozenset(split for split, _ in _SPLIT_FILES)
_EXPECTED_FILENAMES = frozenset(filename for _, filename in _SPLIT_FILES)
_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "example_id",
        "document_id",
        "sentence_id",
        "split",
        "words",
        "predicate_index",
        "tags",
        "predicate_roleset",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "record_counts",
        "file_sha256",
        "dataset_fingerprint",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class DatasetFormatError(ValueError):
    """Raised when a prepared dataset violates its persisted contract."""


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Verified metadata for one prepared dataset snapshot."""

    schema_version: str
    record_counts: Mapping[DatasetSplit, int]
    file_sha256: Mapping[str, str]
    dataset_fingerprint: str


@dataclass(frozen=True, slots=True)
class PreparedSRLDataset:
    """Final examples and the manifest against which they were verified."""

    examples: tuple[WordLevelSRLExample, ...]
    manifest: DatasetManifest


def compute_dataset_fingerprint(file_sha256: Mapping[str, str]) -> str:
    """Return the deterministic fingerprint for canonical split digests.

    The input must contain exactly the three canonical split filenames.  The
    caller's mapping order is ignored; digests are always consumed in
    train/development/test order.  Including filenames and the record schema
    version domain-separates the fingerprint from an unlabelled digest list.
    """

    _require_exact_keys(
        file_sha256,
        _EXPECTED_FILENAMES,
        context="file_sha256",
    )
    for filename, digest in file_sha256.items():
        _validate_sha256(digest, context=f"file_sha256[{filename!r}]")

    fingerprint_input = bytearray()
    fingerprint_input.extend(DATASET_SCHEMA_VERSION.encode("utf-8"))
    fingerprint_input.extend(b"\n")
    for _, filename in _SPLIT_FILES:
        fingerprint_input.extend(filename.encode("utf-8"))
        fingerprint_input.extend(b"\0")
        fingerprint_input.extend(file_sha256[filename].encode("ascii"))
        fingerprint_input.extend(b"\n")
    return hashlib.sha256(fingerprint_input).hexdigest()


def write_prepared_dataset(
    directory: str | os.PathLike[str],
    examples: Iterable[WordLevelSRLExample],
) -> DatasetManifest:
    """Validate and atomically persist final SRL examples.

    Input order does not affect the bytes on disk: rows are grouped by split
    and ordered by ``example_id``.  Empty splits are represented by empty
    files and have a zero manifest count.
    """

    validated = tuple(_validated_copy(example) for example in examples)
    _validate_collection(validated)

    by_split: dict[DatasetSplit, list[WordLevelSRLExample]] = {
        split: [] for split, _ in _SPLIT_FILES
    }
    for example in validated:
        by_split[example.split].append(example)

    split_bytes: dict[str, bytes] = {}
    record_counts: dict[DatasetSplit, int] = {}
    for split, filename in _SPLIT_FILES:
        ordered = sorted(by_split[split], key=lambda item: item.example_id)
        split_bytes[filename] = b"".join(_record_bytes(item) for item in ordered)
        record_counts[split] = len(ordered)

    file_sha256 = {
        filename: hashlib.sha256(split_bytes[filename]).hexdigest()
        for _, filename in _SPLIT_FILES
    }
    manifest = _new_manifest(record_counts, file_sha256)

    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for _, filename in _SPLIT_FILES:
        _atomic_write_bytes(output_directory / filename, split_bytes[filename])
    _atomic_write_bytes(
        output_directory / MANIFEST_FILENAME,
        _manifest_bytes(manifest),
    )
    return manifest


def read_prepared_dataset(
    directory: str | os.PathLike[str],
) -> PreparedSRLDataset:
    """Load a dataset only after schema, digest, count, and leakage checks."""

    input_directory = Path(directory)
    manifest_path = input_directory / MANIFEST_FILENAME
    manifest = _parse_manifest(manifest_path.read_bytes(), context=str(manifest_path))

    actual_digests: dict[str, str] = {}
    examples: list[WordLevelSRLExample] = []
    for split, filename in _SPLIT_FILES:
        path = input_directory / filename
        contents = path.read_bytes()
        actual_digest = hashlib.sha256(contents).hexdigest()
        actual_digests[filename] = actual_digest
        expected_digest = manifest.file_sha256[filename]
        if actual_digest != expected_digest:
            raise DatasetFormatError(
                f"SHA-256 mismatch for {filename}: expected "
                f"{expected_digest}, found {actual_digest}"
            )

        split_examples = _parse_split(contents, split=split, context=str(path))
        expected_count = manifest.record_counts[split]
        if len(split_examples) != expected_count:
            raise DatasetFormatError(
                f"record count mismatch for {filename}: expected "
                f"{expected_count}, found {len(split_examples)}"
            )
        examples.extend(split_examples)

    actual_fingerprint = compute_dataset_fingerprint(actual_digests)
    if actual_fingerprint != manifest.dataset_fingerprint:
        raise DatasetFormatError(
            "dataset fingerprint does not match the verified split files"
        )

    result = tuple(examples)
    _validate_collection(result)
    return PreparedSRLDataset(examples=result, manifest=manifest)


def _validated_copy(example: WordLevelSRLExample) -> WordLevelSRLExample:
    if type(example) is not WordLevelSRLExample:
        raise TypeError(
            "prepared datasets accept final WordLevelSRLExample records only"
        )
    try:
        return WordLevelSRLExample(
            example_id=example.example_id,
            document_id=example.document_id,
            sentence_id=example.sentence_id,
            split=example.split,
            words=example.words,
            predicate_index=example.predicate_index,
            tags=example.tags,
            predicate_roleset=example.predicate_roleset,
        )
    except (IndexError, TypeError, ValueError) as exc:
        example_id = getattr(example, "example_id", "<unknown>")
        raise DatasetFormatError(
            f"invalid final SRL record {example_id!r}: {exc}"
        ) from exc


def _validate_collection(examples: tuple[WordLevelSRLExample, ...]) -> None:
    seen_example_ids: set[str] = set()
    seen_semantic_identities: set[tuple[str, str, int]] = set()
    document_splits: dict[str, DatasetSplit] = {}
    sentence_details: dict[
        str, tuple[str, DatasetSplit, tuple[str, ...]]
    ] = {}
    sentence_text_splits: dict[tuple[str, ...], DatasetSplit] = {}

    for example in examples:
        if example.example_id in seen_example_ids:
            raise DatasetFormatError(
                f"duplicate example ID: {example.example_id!r}"
            )
        seen_example_ids.add(example.example_id)

        semantic_identity = (
            example.document_id,
            example.sentence_id,
            example.predicate_index,
        )
        if semantic_identity in seen_semantic_identities:
            raise DatasetFormatError(
                "duplicate semantic identity: "
                f"document={example.document_id!r}, "
                f"sentence={example.sentence_id!r}, "
                f"predicate_index={example.predicate_index}"
            )
        seen_semantic_identities.add(semantic_identity)

        previous_document_split = document_splits.setdefault(
            example.document_id, example.split
        )
        if previous_document_split != example.split:
            raise DatasetFormatError(
                f"document leakage: {example.document_id!r} occurs in "
                f"{previous_document_split!r} and {example.split!r}"
            )

        previous_sentence = sentence_details.get(example.sentence_id)
        if previous_sentence is None:
            sentence_details[example.sentence_id] = (
                example.document_id,
                example.split,
                example.words,
            )
        else:
            previous_document, previous_split, previous_words = previous_sentence
            if previous_split != example.split:
                raise DatasetFormatError(
                    f"sentence leakage: {example.sentence_id!r} occurs in "
                    f"{previous_split!r} and {example.split!r}"
                )
            if previous_document != example.document_id:
                raise DatasetFormatError(
                    f"sentence ID {example.sentence_id!r} maps to multiple documents"
                )
            if previous_words != example.words:
                raise DatasetFormatError(
                    f"sentence ID {example.sentence_id!r} maps to inconsistent words"
                )

        previous_text_split = sentence_text_splits.setdefault(
            example.words, example.split
        )
        if previous_text_split != example.split:
            raise DatasetFormatError(
                "sentence-text leakage: identical words occur in "
                f"{previous_text_split!r} and {example.split!r}"
            )


def _record_payload(example: WordLevelSRLExample) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "example_id": example.example_id,
        "document_id": example.document_id,
        "sentence_id": example.sentence_id,
        "split": example.split,
        "words": list(example.words),
        "predicate_index": example.predicate_index,
        "tags": list(example.tags),
        "predicate_roleset": example.predicate_roleset,
    }


def _record_bytes(example: WordLevelSRLExample) -> bytes:
    return _canonical_json_bytes(_record_payload(example))


def _parse_split(
    contents: bytes,
    *,
    split: DatasetSplit,
    context: str,
) -> tuple[WordLevelSRLExample, ...]:
    if b"\r" in contents:
        raise DatasetFormatError(f"{context} must use LF newlines, not CRLF")
    if contents and not contents.endswith(b"\n"):
        raise DatasetFormatError(f"{context} must end with an LF newline")

    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(f"{context} is not valid UTF-8") from exc

    if not text:
        return ()

    examples: list[WordLevelSRLExample] = []
    for line_number, line in enumerate(text[:-1].split("\n"), start=1):
        line_context = f"{context}:{line_number}"
        if not line:
            raise DatasetFormatError(f"{line_context} is a blank JSONL row")
        payload = _load_json_object(line, context=line_context)
        _require_exact_keys(payload, _RECORD_KEYS, context=line_context)

        if payload["schema_version"] != DATASET_SCHEMA_VERSION:
            raise DatasetFormatError(
                f"{line_context} has unsupported schema_version "
                f"{payload['schema_version']!r}"
            )
        if payload["split"] != split:
            raise DatasetFormatError(
                f"{line_context} declares split {payload['split']!r}; "
                f"expected {split!r}"
            )
        if type(payload["words"]) is not list:
            raise DatasetFormatError(f"{line_context} words must be a JSON array")
        if type(payload["tags"]) is not list:
            raise DatasetFormatError(f"{line_context} tags must be a JSON array")

        try:
            example = WordLevelSRLExample(
                example_id=payload["example_id"],
                document_id=payload["document_id"],
                sentence_id=payload["sentence_id"],
                split=payload["split"],
                words=tuple(payload["words"]),
                predicate_index=payload["predicate_index"],
                tags=tuple(payload["tags"]),
                predicate_roleset=payload["predicate_roleset"],
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise DatasetFormatError(f"{line_context} is invalid: {exc}") from exc

        canonical_line = _record_bytes(example).decode("utf-8")[:-1]
        if line != canonical_line:
            raise DatasetFormatError(f"{line_context} is not canonical JSON")
        examples.append(example)

    example_ids = [example.example_id for example in examples]
    if example_ids != sorted(example_ids):
        raise DatasetFormatError(
            f"{context} rows must be ordered by example_id"
        )
    return tuple(examples)


def _new_manifest(
    record_counts: Mapping[DatasetSplit, int],
    file_sha256: Mapping[str, str],
) -> DatasetManifest:
    ordered_counts = {split: record_counts[split] for split, _ in _SPLIT_FILES}
    ordered_digests = {
        filename: file_sha256[filename] for _, filename in _SPLIT_FILES
    }
    return DatasetManifest(
        schema_version=DATASET_SCHEMA_VERSION,
        record_counts=MappingProxyType(ordered_counts),
        file_sha256=MappingProxyType(ordered_digests),
        dataset_fingerprint=compute_dataset_fingerprint(ordered_digests),
    )


def _manifest_payload(manifest: DatasetManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "record_counts": {
            split: manifest.record_counts[split] for split, _ in _SPLIT_FILES
        },
        "file_sha256": {
            filename: manifest.file_sha256[filename]
            for _, filename in _SPLIT_FILES
        },
        "dataset_fingerprint": manifest.dataset_fingerprint,
    }


def _manifest_bytes(manifest: DatasetManifest) -> bytes:
    return _canonical_json_bytes(_manifest_payload(manifest))


def _parse_manifest(contents: bytes, *, context: str) -> DatasetManifest:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(f"{context} is not valid UTF-8") from exc
    payload = _load_json_object(text, context=context)
    _require_exact_keys(payload, _MANIFEST_KEYS, context=context)

    if payload["schema_version"] != DATASET_SCHEMA_VERSION:
        raise DatasetFormatError(
            f"{context} has unsupported schema_version "
            f"{payload['schema_version']!r}"
        )

    record_counts = payload["record_counts"]
    if type(record_counts) is not dict:
        raise DatasetFormatError(f"{context} record_counts must be an object")
    _require_exact_keys(record_counts, _VALID_SPLITS, context="record_counts")
    for split, count in record_counts.items():
        if type(count) is not int or count < 0:
            raise DatasetFormatError(
                f"record_counts[{split!r}] must be a non-negative integer"
            )

    file_sha256 = payload["file_sha256"]
    if type(file_sha256) is not dict:
        raise DatasetFormatError(f"{context} file_sha256 must be an object")
    _require_exact_keys(file_sha256, _EXPECTED_FILENAMES, context="file_sha256")
    for filename, digest in file_sha256.items():
        _validate_sha256(digest, context=f"file_sha256[{filename!r}]")

    fingerprint = payload["dataset_fingerprint"]
    _validate_sha256(fingerprint, context="dataset_fingerprint")
    declared_fingerprint = compute_dataset_fingerprint(file_sha256)
    if fingerprint != declared_fingerprint:
        raise DatasetFormatError(
            f"{context} dataset_fingerprint is inconsistent with file_sha256"
        )

    manifest = _new_manifest(record_counts, file_sha256)
    if manifest.dataset_fingerprint != fingerprint:
        raise DatasetFormatError(f"{context} has an inconsistent fingerprint")
    if contents != _manifest_bytes(manifest):
        raise DatasetFormatError(f"{context} is not canonical JSON")
    return manifest


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _load_json_object(text: str, *, context: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DatasetFormatError(
                    f"{context} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise DatasetFormatError(f"{context} contains invalid JSON value {value}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except DatasetFormatError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DatasetFormatError(f"{context} contains malformed JSON") from exc
    if type(payload) is not dict:
        raise DatasetFormatError(f"{context} must contain a JSON object")
    return payload


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise DatasetFormatError(f"{context} has {'; '.join(details)}")


def _validate_sha256(value: Any, *, context: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise DatasetFormatError(f"{context} must be a lowercase SHA-256 hex digest")


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
