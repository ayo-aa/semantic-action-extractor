"""Privacy-safe BabySRL CHAT conversion and deterministic preparation.

The adapter reads the pinned BabySRL ZIP directly and never extracts archive
members.  Corpus text appears only in returned model examples or in prepared
JSONL files chosen by the caller.  Audit dictionaries contain aggregate
counts, digests, and policy identifiers only.

BabySRL's ``%srl:`` tier is a physical-token table.  Each row contains the
tier marker, a word, a predicate marker (``-`` for ordinary rows), and one
bracket cell per proposition in the utterance.  Proposition columns, rather
than predicate-marker rows, are therefore the conversion denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile, ZipInfo

from .dataset_io import DatasetManifest, write_prepared_dataset
from .example import DatasetSplit, PreparedWordLevelSRLExample, WordLevelSRLExample


BABYSRL_ARCHIVE_SHA256 = (
    "a2d8d38b0818910d05154cb62adcb05ef36b00f0aeae2690b7e1677fd5154604"
)
BABYSRL_ARCHIVE_SIZE_BYTES = 3_151_030
BABYSRL_AUDIT_SCHEMA_VERSION = "semantic-action-extractor.babysrl-audit/v1"
BABYSRL_ASSIGNMENT_MANIFEST_SCHEMA_VERSION = (
    "semantic-action-extractor.babysrl-document-splits/v1"
)
BABYSRL_SPLIT_POLICY = "child-stratified-document-sha256-80-10-10/v1"
BABYSRL_DUPLICATE_POLICY = (
    "cross-split-words-exclude-all_then_dev-test-exact-semantic-dedupe/v1"
)
DEFAULT_MINIMUM_CONVERSION_COVERAGE = 0.99

_CHILDREN = ("Adam", "Eve", "Sarah")
_MEMBER_PATTERN = re.compile(
    r"BabySRL/(?:(?P<adam_child>Adam)/(?P<adam_document>adam[0-9]{2})"
    r"|(?P<eve_child>Eve)/(?P<eve_document>eve[0-9]{2})"
    r"|(?P<sarah_child>Sarah)/(?P<sarah_document>sarah[0-9]{3}))"
    r"\.srl\.cha"
)
_ALLOWED_NON_CHAT_MEMBERS = frozenset(
    {
        "BabySRL/",
        "BabySRL/.DS_Store",
        "__MACOSX/BabySRL/._.DS_Store",
        "BabySRL/Eve/",
        "BabySRL/Adam/",
        "BabySRL/Sarah/",
        "__MACOSX/BabySRL/Adam/._adam09.srl.cha",
    }
)
_DOCUMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_OPEN_CELL_PATTERN = re.compile(r"(?:\([^()*]+)*")
_OPEN_LABEL_PATTERN = re.compile(r"\(([^()*]+)")
_ARGUMENT_PATTERN = re.compile(r"A([0-5])")
_MODIFIER_PATTERN = re.compile(r"AM-([A-Z]+)")
_SPLIT_ORDER: tuple[DatasetSplit, ...] = (
    "train",
    "development",
    "test",
)


class BabySRLArchiveError(ValueError):
    """Raised when the archive fails its immutable integrity boundary."""


class BabySRLPreparationError(ValueError):
    """Raised when converted data cannot satisfy the preparation policy."""


class _ColumnFailure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class BabySRLDocument:
    """One decoded CHAT document retained only inside the conversion boundary."""

    child: str
    document_name: str
    source: str

    @property
    def document_id(self) -> str:
        return f"babysrl:{self.child.lower()}:{self.document_name.lower()}"


@dataclass(frozen=True, slots=True)
class BabySRLConversion:
    """Converted, unsplit examples plus aggregate-only accounting."""

    examples: tuple[PreparedWordLevelSRLExample, ...]
    report: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BabySRLDatasetBuild:
    """Final examples and an aggregate-only audit report."""

    examples: tuple[WordLevelSRLExample, ...]
    report: Mapping[str, object]
    assignment_manifest: BabySRLDocumentAssignmentManifest


@dataclass(frozen=True, slots=True)
class BabySRLDocumentAssignmentManifest:
    """Canonical, corpus-text-free freeze of every document assignment."""

    canonical_bytes: bytes
    sha256: str
    document_count: int


@dataclass(frozen=True, slots=True)
class BabySRLPreparationReceipt:
    """Aggregate result of writing a prepared dataset."""

    report: Mapping[str, object]
    manifest: DatasetManifest
    assignment_manifest: BabySRLDocumentAssignmentManifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_is_unsafe(member: ZipInfo) -> bool:
    name = member.filename
    if not name or "\\" in name or "\x00" in name:
        return True
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        return True
    if pure.as_posix() != name.rstrip("/"):
        return True
    mode = member.external_attr >> 16
    return stat.S_IFMT(mode) == stat.S_IFLNK


def _validate_archive(
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> tuple[str, int]:
    if not archive_path.is_file():
        raise BabySRLArchiveError("BabySRL archive is not a regular file")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if type(expected_size_bytes) is not int or expected_size_bytes < 1:
        raise ValueError("expected_size_bytes must be a positive integer")

    size_bytes = archive_path.stat().st_size
    if size_bytes != expected_size_bytes:
        raise BabySRLArchiveError("BabySRL archive size does not match the pin")
    archive_sha256 = _sha256_file(archive_path)
    if archive_sha256 != expected_sha256:
        raise BabySRLArchiveError("BabySRL archive SHA-256 does not match the pin")
    return archive_sha256, size_bytes


def _read_archive_documents(
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> tuple[tuple[BabySRLDocument, ...], dict[str, object]]:
    archive_sha256, size_bytes = _validate_archive(
        archive_path,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    try:
        with ZipFile(archive_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise BabySRLArchiveError("BabySRL archive has duplicate member names")
            if any(_member_is_unsafe(member) for member in members):
                raise BabySRLArchiveError("BabySRL archive contains an unsafe member")
            if any(member.flag_bits & 0x1 for member in members):
                raise BabySRLArchiveError("BabySRL archive contains encrypted members")
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise BabySRLArchiveError("BabySRL archive failed its CRC check")

            documents: list[BabySRLDocument] = []
            for member in members:
                match = _MEMBER_PATTERN.fullmatch(member.filename)
                if match is None:
                    if member.filename not in _ALLOWED_NON_CHAT_MEMBERS:
                        raise BabySRLArchiveError(
                            "BabySRL archive contains an unexpected member"
                        )
                    continue
                if member.is_dir():
                    raise BabySRLArchiveError("BabySRL CHAT member is a directory")
                try:
                    source = archive.read(member).decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise BabySRLArchiveError(
                        "BabySRL CHAT member is not valid UTF-8"
                    ) from exc
                documents.append(
                    BabySRLDocument(
                        child=(
                            match.group("adam_child")
                            or match.group("eve_child")
                            or match.group("sarah_child")
                        ),
                        document_name=(
                            match.group("adam_document")
                            or match.group("eve_document")
                            or match.group("sarah_document")
                        ),
                        source=source,
                    )
                )
    except BadZipFile as exc:
        raise BabySRLArchiveError("BabySRL archive is not a valid ZIP") from exc

    if not documents:
        raise BabySRLArchiveError("BabySRL archive contains no supported CHAT files")
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise BabySRLArchiveError("BabySRL archive has duplicate document identities")
    documents.sort(key=lambda item: item.document_id)
    archive_report: dict[str, object] = {
        "sha256": archive_sha256,
        "size_bytes": size_bytes,
        "member_count": len(members),
        "chat_document_count": len(documents),
        "uncompressed_size_bytes": sum(member.file_size for member in members),
        "crc_status": "pass",
        "path_safety_status": "pass",
    }
    return tuple(documents), archive_report


def _normalized_role(raw_label: str) -> str:
    if raw_label in {"V", "C-V"}:
        return raw_label
    argument = _ARGUMENT_PATTERN.fullmatch(raw_label)
    modifier = _MODIFIER_PATTERN.fullmatch(raw_label)
    if argument is not None:
        return f"ARG{argument.group(1)}"
    if modifier is not None:
        return f"ARGM-{modifier.group(1)}"
    raise _ColumnFailure("unsupported_role_label")


def _decode_column(cells: Sequence[str]) -> tuple[tuple[str, int, int], ...]:
    stack: list[tuple[str, int]] = []
    spans: list[tuple[str, int, int]] = []

    for word_index, cell in enumerate(cells):
        if cell.count("*") != 1:
            raise _ColumnFailure("invalid_bracket_sequence")
        left, right = cell.split("*", 1)
        if _OPEN_CELL_PATTERN.fullmatch(left) is None or any(
            character != ")" for character in right
        ):
            raise _ColumnFailure("invalid_bracket_sequence")

        for raw_label in _OPEN_LABEL_PATTERN.findall(left):
            stack.append((_normalized_role(raw_label), word_index))
            if len(stack) > 1:
                raise _ColumnFailure("invalid_bracket_sequence")

        for _ in right:
            if not stack:
                raise _ColumnFailure("invalid_bracket_sequence")
            label, start = stack.pop()
            spans.append((label, start, word_index + 1))

    if stack:
        raise _ColumnFailure("invalid_bracket_sequence")
    return tuple(spans)


def _convert_column(
    *,
    words: tuple[str, ...],
    predicate_markers: tuple[str, ...],
    cells: tuple[str, ...],
    example_id: str,
    document_id: str,
    sentence_id: str,
) -> PreparedWordLevelSRLExample:
    spans = _decode_column(cells)
    relation_spans = [span for span in spans if span[0] in {"V", "C-V"}]
    if not relation_spans:
        raise _ColumnFailure("missing_relation_span")
    relation_indexes = {
        word_index
        for _, start, end in relation_spans
        for word_index in range(start, end)
    }
    predicate_candidates = sorted(
        word_index
        for word_index in relation_indexes
        if predicate_markers[word_index] != "-"
    )
    if len(predicate_candidates) != 1:
        raise _ColumnFailure("ambiguous_predicate_head")
    predicate_index = predicate_candidates[0]
    predicate_marker = predicate_markers[predicate_index]

    tags = ["O"] * len(words)
    for label, start, end in spans:
        if label in {"V", "C-V"}:
            continue
        if any(tag != "O" for tag in tags[start:end]):
            raise _ColumnFailure("invalid_bracket_sequence")
        tags[start] = f"B-{label}"
        for word_index in range(start + 1, end):
            tags[word_index] = f"I-{label}"

    for _, start, end in relation_spans:
        piece_indexes = [
            word_index
            for word_index in range(start, end)
            if word_index != predicate_index
        ]
        run_start = True
        previous_index: int | None = None
        for word_index in piece_indexes:
            if tags[word_index] != "O":
                raise _ColumnFailure("invalid_bracket_sequence")
            if previous_index is None or word_index != previous_index + 1:
                run_start = True
            tags[word_index] = "B-C-V" if run_start else "I-C-V"
            run_start = False
            previous_index = word_index
    if tags[predicate_index] != "O":
        raise _ColumnFailure("invalid_bracket_sequence")
    tags[predicate_index] = "B-V"

    try:
        return PreparedWordLevelSRLExample(
            example_id=example_id,
            document_id=document_id,
            sentence_id=sentence_id,
            words=words,
            predicate_index=predicate_index,
            tags=tuple(tags),
            predicate_roleset=f"{predicate_marker}.XX",
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise AssertionError("adapter produced an invalid model record") from exc


def convert_babysrl_chat(
    source: str,
    *,
    child: str,
    document_name: str,
) -> BabySRLConversion:
    """Convert one CHAT document and return only aggregate audit metadata.

    This lower-level function is intentionally suitable for wholly invented
    test fixtures.  Its ``examples`` field contains model data; its ``report``
    field never contains words, predicates, labels, document names, or source
    excerpts.
    """

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if child not in _CHILDREN:
        raise ValueError("child must be Adam, Eve, or Sarah")
    if _DOCUMENT_NAME_PATTERN.fullmatch(document_name) is None:
        raise ValueError("document_name contains unsupported characters")

    document_id = f"babysrl:{child.lower()}:{document_name.lower()}"
    examples: list[PreparedWordLevelSRLExample] = []
    rejection_counts: Counter[str] = Counter()
    utterance_blocks = 0
    declared_proposition_columns = 0
    first_row_proposition_columns = 0
    predicate_marker_rows = 0
    malformed_blocks_without_columns = 0
    current_rows: list[tuple[str, ...]] = []
    current_speaker: str | None = None

    def flush_block() -> None:
        nonlocal utterance_blocks
        nonlocal declared_proposition_columns
        nonlocal first_row_proposition_columns
        nonlocal predicate_marker_rows
        nonlocal malformed_blocks_without_columns
        if not current_rows:
            return
        utterance_blocks += 1
        sentence_id = f"{document_id}:u{utterance_blocks:05d}"
        widths = tuple(max(0, len(row) - 3) for row in current_rows)
        column_count = max(widths, default=0)
        declared_proposition_columns += column_count
        first_row_proposition_columns += widths[0]
        predicate_marker_rows += sum(
            len(row) >= 3 and row[2] != "-" for row in current_rows
        )
        if column_count == 0:
            malformed_blocks_without_columns += 1
            return
        if any(len(row) < 3 for row in current_rows) or len(set(widths)) != 1:
            rejection_counts["row_width_mismatch"] += column_count
            return

        words = tuple(row[1] for row in current_rows)
        predicate_markers = tuple(row[2] for row in current_rows)
        for column_index in range(column_count):
            example_id = f"{sentence_id}:p{column_index + 1:02d}"
            cells = tuple(row[3 + column_index] for row in current_rows)
            try:
                example = _convert_column(
                    words=words,
                    predicate_markers=predicate_markers,
                    cells=cells,
                    example_id=example_id,
                    document_id=document_id,
                    sentence_id=sentence_id,
                )
            except _ColumnFailure as exc:
                rejection_counts[exc.reason] += 1
            else:
                examples.append(example)

    for line in source.splitlines():
        if line.startswith("%srl:"):
            if current_speaker not in {"*MOT", "*FAT"}:
                raise BabySRLPreparationError(
                    "BabySRL %srl block is not attached to a MOT/FAT main tier"
                )
            current_rows.append(tuple(line.split()))
        else:
            flush_block()
            current_rows.clear()
            if line.startswith("*"):
                current_speaker = line.partition(":")[0]
            elif line.startswith("@"):
                current_speaker = None
    flush_block()

    rejected = sum(rejection_counts.values())
    if len(examples) + rejected != declared_proposition_columns:
        raise AssertionError("BabySRL conversion counts do not reconcile")
    report: Mapping[str, object] = {
        "utterance_blocks": utterance_blocks,
        "declared_proposition_columns": declared_proposition_columns,
        "first_row_proposition_columns": first_row_proposition_columns,
        "predicate_marker_rows": predicate_marker_rows,
        "accepted_proposition_columns": len(examples),
        "rejected_proposition_columns": rejected,
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "malformed_blocks_without_columns": malformed_blocks_without_columns,
    }
    return BabySRLConversion(examples=tuple(examples), report=report)


def _stable_document_key(child: str, document_id: str) -> tuple[str, str]:
    document_name = document_id.rsplit(":", 1)[-1]
    basename = f"{document_name}.srl.cha"
    payload = f"babysrl-v1\0{child}/{basename}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), document_id


def assign_babysrl_document_splits(
    documents_by_child: Mapping[str, Iterable[str]],
) -> Mapping[str, DatasetSplit]:
    """Assign whole documents using the predeclared child-stratified policy."""

    unknown_children = set(documents_by_child) - set(_CHILDREN)
    if unknown_children:
        raise ValueError("documents_by_child contains an unsupported child")
    assignments: dict[str, DatasetSplit] = {}
    for child in _CHILDREN:
        document_ids = tuple(documents_by_child.get(child, ()))
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("documents_by_child contains duplicate document IDs")
        ordered = sorted(
            document_ids,
            key=lambda item: _stable_document_key(child, item),
        )
        train_count = len(ordered) * 80 // 100
        development_count = len(ordered) * 10 // 100
        boundaries = (train_count, train_count + development_count)
        for index, document_id in enumerate(ordered):
            split: DatasetSplit
            if index < boundaries[0]:
                split = "train"
            elif index < boundaries[1]:
                split = "development"
            else:
                split = "test"
            if document_id in assignments:
                raise ValueError("a document occurs under multiple children")
            assignments[document_id] = split
    return MappingProxyType(assignments)


def build_babysrl_document_assignment_manifest(
    documents_by_child: Mapping[str, Iterable[str]],
    assignments: Mapping[str, DatasetSplit] | None = None,
) -> BabySRLDocumentAssignmentManifest:
    """Return canonical bytes and a digest for the exact document split.

    The manifest deliberately contains source basenames and stable document
    identifiers, but no utterance text, tokens, labels, or corpus excerpts.
    Input mapping order cannot change its bytes.
    """

    materialized = {
        child: tuple(document_ids)
        for child, document_ids in documents_by_child.items()
    }
    resolved_assignments = (
        assign_babysrl_document_splits(materialized)
        if assignments is None
        else assignments
    )
    expected_ids = {
        document_id
        for document_ids in materialized.values()
        for document_id in document_ids
    }
    if set(resolved_assignments) != expected_ids:
        raise ValueError(
            "assignments must contain exactly the supplied document IDs"
        )

    child_order = {child: index for index, child in enumerate(_CHILDREN)}
    records: list[dict[str, object]] = []
    for child, document_ids in materialized.items():
        if child not in child_order:
            raise ValueError("documents_by_child contains an unsupported child")
        for document_id in document_ids:
            document_name = document_id.rsplit(":", 1)[-1]
            if document_id != f"babysrl:{child.lower()}:{document_name}":
                raise ValueError("document ID is inconsistent with its child")
            split = resolved_assignments[document_id]
            if split not in _SPLIT_ORDER:
                raise ValueError("assignment contains an unsupported split")
            records.append(
                {
                    "child": child,
                    "source_basename": f"{document_name}.srl.cha",
                    "document_id": document_id,
                    "split": split,
                }
            )
    records.sort(
        key=lambda record: (
            child_order[str(record["child"])],
            str(record["source_basename"]),
        )
    )
    payload = {
        "schema_version": BABYSRL_ASSIGNMENT_MANIFEST_SCHEMA_VERSION,
        "split_policy": BABYSRL_SPLIT_POLICY,
        "documents": records,
    }
    canonical_bytes = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return BabySRLDocumentAssignmentManifest(
        canonical_bytes=canonical_bytes,
        sha256=hashlib.sha256(canonical_bytes).hexdigest(),
        document_count=len(records),
    )


def _word_sequence_digest(words: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"babysrl-word-sequence/v1\0")
    for word in words:
        encoded = word.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _apply_split_and_duplicate_policy(
    examples: Sequence[PreparedWordLevelSRLExample],
    assignments: Mapping[str, DatasetSplit],
) -> tuple[tuple[WordLevelSRLExample, ...], dict[str, object]]:
    assigned: list[WordLevelSRLExample] = []
    before_counts: Counter[DatasetSplit] = Counter()
    for example in examples:
        try:
            split = assignments[example.document_id]
        except KeyError as exc:
            raise BabySRLPreparationError(
                "converted example has no document split assignment"
            ) from exc
        final = example.assign_split(split)
        assigned.append(final)
        before_counts[split] += 1

    seen_semantic_identities: set[tuple[str, str, int]] = set()
    for example in assigned:
        semantic_identity = (
            example.document_id,
            example.sentence_id,
            example.predicate_index,
        )
        if semantic_identity in seen_semantic_identities:
            raise BabySRLPreparationError(
                "accepted columns contain a duplicate sentence/predicate identity"
            )
        seen_semantic_identities.add(semantic_identity)

    digest_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
    digest_words: dict[str, tuple[str, ...]] = {}
    for example in assigned:
        digest = _word_sequence_digest(example.words)
        previous_words = digest_words.setdefault(digest, example.words)
        if previous_words != example.words:
            raise BabySRLPreparationError("word-sequence digest collision")
        digest_splits[digest].add(example.split)
    cross_split_digests = {
        digest for digest, splits in digest_splits.items() if len(splits) > 1
    }

    leakage_excluded: Counter[DatasetSplit] = Counter()
    leakage_safe: list[WordLevelSRLExample] = []
    for example in assigned:
        if _word_sequence_digest(example.words) in cross_split_digests:
            leakage_excluded[example.split] += 1
        else:
            leakage_safe.append(example)

    deduplicated: list[WordLevelSRLExample] = []
    duplicates_removed: Counter[DatasetSplit] = Counter()
    seen_exact: dict[
        DatasetSplit,
        set[tuple[tuple[str, ...], int, tuple[str, ...]]],
    ] = defaultdict(set)
    for example in sorted(leakage_safe, key=lambda item: item.example_id):
        exact_key = (
            example.words,
            example.predicate_index,
            example.tags,
        )
        if example.split != "train" and exact_key in seen_exact[example.split]:
            duplicates_removed[example.split] += 1
            continue
        seen_exact[example.split].add(exact_key)
        deduplicated.append(example)

    final_counts = Counter(example.split for example in deduplicated)
    if len(assigned) != (
        len(deduplicated)
        + sum(leakage_excluded.values())
        + sum(duplicates_removed.values())
    ):
        raise AssertionError("BabySRL split-policy counts do not reconcile")
    report: dict[str, object] = {
        "policy": BABYSRL_SPLIT_POLICY,
        "duplicate_policy": BABYSRL_DUPLICATE_POLICY,
        "examples_before_duplicate_policy": {
            split: before_counts[split] for split in _SPLIT_ORDER
        },
        "cross_split_word_sequence_groups": len(cross_split_digests),
        "cross_split_leakage_exclusions": {
            split: leakage_excluded[split] for split in _SPLIT_ORDER
        },
        "dev_test_exact_semantic_duplicates_removed": {
            split: duplicates_removed[split] for split in _SPLIT_ORDER
        },
        "final_examples": {
            split: final_counts[split] for split in _SPLIT_ORDER
        },
    }
    return tuple(deduplicated), report


def build_babysrl_dataset(
    archive_path: str | os.PathLike[str],
    *,
    expected_sha256: str = BABYSRL_ARCHIVE_SHA256,
    expected_size_bytes: int = BABYSRL_ARCHIVE_SIZE_BYTES,
    minimum_conversion_coverage: float = DEFAULT_MINIMUM_CONVERSION_COVERAGE,
) -> BabySRLDatasetBuild:
    """Build final examples in memory and return an aggregate-only report."""

    if isinstance(minimum_conversion_coverage, bool) or not isinstance(
        minimum_conversion_coverage, (int, float)
    ):
        raise TypeError("minimum_conversion_coverage must be numeric")
    if not 0.0 <= float(minimum_conversion_coverage) <= 1.0:
        raise ValueError("minimum_conversion_coverage must be between zero and one")

    documents, archive_report = _read_archive_documents(
        Path(archive_path),
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    all_examples: list[PreparedWordLevelSRLExample] = []
    conversion_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    documents_by_child: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        conversion = convert_babysrl_chat(
            document.source,
            child=document.child,
            document_name=document.document_name,
        )
        all_examples.extend(conversion.examples)
        documents_by_child[document.child].append(document.document_id)
        for key in (
            "utterance_blocks",
            "declared_proposition_columns",
            "first_row_proposition_columns",
            "predicate_marker_rows",
            "accepted_proposition_columns",
            "rejected_proposition_columns",
            "malformed_blocks_without_columns",
        ):
            conversion_counts[key] += int(conversion.report[key])
        rejection_counts.update(conversion.report["rejection_reasons"])

    declared = conversion_counts["declared_proposition_columns"]
    accepted = conversion_counts["accepted_proposition_columns"]
    rejected = conversion_counts["rejected_proposition_columns"]
    if accepted + rejected != declared or accepted != len(all_examples):
        raise AssertionError("BabySRL archive conversion counts do not reconcile")
    coverage = accepted / declared if declared else 0.0

    assignments = assign_babysrl_document_splits(documents_by_child)
    assignment_manifest = build_babysrl_document_assignment_manifest(
        documents_by_child, assignments
    )
    final_examples, split_report = _apply_split_and_duplicate_policy(
        all_examples, assignments
    )
    document_split_counts: dict[str, dict[str, int]] = {}
    for child in _CHILDREN:
        counts = Counter(
            assignments[document_id] for document_id in documents_by_child[child]
        )
        document_split_counts[child.lower()] = {
            split: counts[split] for split in _SPLIT_ORDER
        }

    conversion_status = (
        "pass" if coverage >= float(minimum_conversion_coverage) else "fail"
    )
    minimum_accepted = math.ceil(
        declared * float(minimum_conversion_coverage)
    )
    nonempty_splits = all(
        int(split_report["final_examples"][split]) > 0 for split in _SPLIT_ORDER
    )
    split_status = "pass" if nonempty_splits else "fail"
    overall_status = (
        "pass" if conversion_status == "pass" and split_status == "pass" else "fail"
    )
    conversion_report: dict[str, object] = {
        **dict(conversion_counts),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "conversion_coverage": coverage,
        "denominator_policy": "maximum role-column count per %srl utterance block",
        "denominator_diagnostics": {
            "max_role_columns_minus_first_row_role_columns": (
                declared - conversion_counts["first_row_proposition_columns"]
            ),
            "predicate_marker_rows_minus_max_role_columns": (
                conversion_counts["predicate_marker_rows"] - declared
            ),
        },
        "minimum_accepted_proposition_columns": minimum_accepted,
        "coverage_margin_proposition_columns": accepted - minimum_accepted,
    }
    report: Mapping[str, object] = {
        "schema_version": BABYSRL_AUDIT_SCHEMA_VERSION,
        "archive": archive_report,
        "conversion": conversion_report,
        "split": {
            **split_report,
            "document_counts_by_child": document_split_counts,
            "document_assignment_manifest": {
                "schema_version": BABYSRL_ASSIGNMENT_MANIFEST_SCHEMA_VERSION,
                "document_count": assignment_manifest.document_count,
                "sha256": assignment_manifest.sha256,
            },
        },
        "gate": {
            "minimum_conversion_coverage": float(minimum_conversion_coverage),
            "archive_integrity": "pass",
            "conversion": conversion_status,
            "nonempty_final_splits": split_status,
            "status": overall_status,
        },
    }
    return BabySRLDatasetBuild(
        examples=tuple(sorted(final_examples, key=lambda item: item.example_id)),
        report=report,
        assignment_manifest=assignment_manifest,
    )


def audit_babysrl_archive(
    archive_path: str | os.PathLike[str],
    *,
    expected_sha256: str = BABYSRL_ARCHIVE_SHA256,
    expected_size_bytes: int = BABYSRL_ARCHIVE_SIZE_BYTES,
    minimum_conversion_coverage: float = DEFAULT_MINIMUM_CONVERSION_COVERAGE,
) -> Mapping[str, object]:
    """Return only aggregate, JSON-serializable audit information."""

    return build_babysrl_dataset(
        archive_path,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        minimum_conversion_coverage=minimum_conversion_coverage,
    ).report


def prepare_babysrl_archive(
    archive_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    talkbank_access_and_rules_confirmed: bool = False,
    expected_sha256: str = BABYSRL_ARCHIVE_SHA256,
    expected_size_bytes: int = BABYSRL_ARCHIVE_SIZE_BYTES,
    minimum_conversion_coverage: float = DEFAULT_MINIMUM_CONVERSION_COVERAGE,
) -> BabySRLPreparationReceipt:
    """Validate, convert, gate, and atomically write the prepared split files."""

    if talkbank_access_and_rules_confirmed is not True:
        raise BabySRLPreparationError(
            "preparation requires confirmation of TalkBank registration and "
            "acceptance of the current access and ground rules"
        )
    build = build_babysrl_dataset(
        archive_path,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        minimum_conversion_coverage=minimum_conversion_coverage,
    )
    if build.report["gate"]["status"] != "pass":
        raise BabySRLPreparationError(
            "BabySRL preparation gate failed; no dataset was written"
        )
    manifest = write_prepared_dataset(output_directory, build.examples)
    return BabySRLPreparationReceipt(
        report=build.report,
        manifest=manifest,
        assignment_manifest=build.assignment_manifest,
    )


def _canonical_report_bytes(report: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            report,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run aggregate audit and optionally prepare ignored local split files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--confirm-talkbank-access-and-rules",
        action="store_true",
        help=(
            "confirm that you registered/signed into TalkBank and accepted "
            "the current access and ground rules"
        ),
    )
    arguments = parser.parse_args(argv)

    if arguments.output_directory is None:
        report = audit_babysrl_archive(arguments.archive)
    else:
        if not arguments.confirm_talkbank_access_and_rules:
            parser.error(
                "--output-directory requires "
                "--confirm-talkbank-access-and-rules"
            )
        receipt = prepare_babysrl_archive(
            arguments.archive,
            arguments.output_directory,
            talkbank_access_and_rules_confirmed=(
                arguments.confirm_talkbank_access_and_rules
            ),
        )
        report = receipt.report
    contents = _canonical_report_bytes(report)
    if arguments.report is None:
        print(contents.decode("utf-8"), end="")
    else:
        _atomic_write(arguments.report, contents)
    return 0 if report["gate"]["status"] == "pass" else 2


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
