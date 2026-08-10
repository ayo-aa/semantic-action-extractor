"""Pinned, fail-closed English Web Treebank span-SRL preparation.

PropBank's public EWT ``.gold_skel`` files contain word-level proposition
brackets but redact the words.  Universal Dependencies English EWT contains
the corresponding public token sequences.  This adapter joins those two
official releases by normalized document identity and sentence position.

The default source revisions are immutable.  A relevant tracked-file change,
an unexpected revision, a malformed source record, or a split disagreement is
rejected before any prepared data is written.  Individual proposition or
token-width failures are excluded atomically and counted in an aggregate-only
audit.  Exact word sequences observed in more than one official split are
excluded from every split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Mapping, Sequence

from .bio import continuation_tag
from .dataset_io import DatasetManifest, write_prepared_dataset
from .example import DatasetSplit, PreparedWordLevelSRLExample, WordLevelSRLExample
from .label_vocabulary import build_training_label_vocabulary
from .propbank import PropBankAdapterError, VERBAL_PENN_TAGS, normalize_role_label


PROPBANK_RELEASE_COMMIT = "4abade0b53ce4a181e1d98b3518101c1a44d395a"
UD_ENGLISH_EWT_COMMIT = "6e064999a75b9c941c515ce1be98352e6f9831e0"
UD_ENGLISH_EWT_TAG = "r2.2"
EWT_AUDIT_SCHEMA_VERSION = "semantic-action-extractor.ewt-audit/v1"
EWT_PROVENANCE_SCHEMA_VERSION = "semantic-action-extractor.ewt-provenance/v1"
EWT_PREPARATION_RECEIPT_SCHEMA_VERSION = (
    "semantic-action-extractor.ewt-preparation-receipt/v1"
)
EWT_PROVENANCE_FILENAME = "ewt_provenance.json"
DEFAULT_MINIMUM_VERBAL_PREDICATE_COVERAGE = 0.99
EWT_JOIN_POLICY = "normalized-document_sentence-position_equal-token-width/v1"
EWT_DUPLICATE_POLICY = (
    "cross-split-words-exclude-all_then-conflicting-input-exclude-all_"
    "then-same-source-identical-target-dedupe_"
    "then-dev-test-cross-source-exact-semantic-dedupe/v2"
)

_SPLIT_ORDER: tuple[DatasetSplit, ...] = (
    "train",
    "development",
    "test",
)
_UD_FILENAMES: tuple[tuple[DatasetSplit, str], ...] = (
    ("train", "en_ewt-ud-train.conllu"),
    ("development", "en_ewt-ud-dev.conllu"),
    ("test", "en_ewt-ud-test.conllu"),
)
_PB_SPLIT_FILENAMES: tuple[tuple[DatasetSplit, str], ...] = (
    ("train", "ewt.train.txt"),
    ("development", "ewt.dev.txt"),
    ("test", "ewt.test.txt"),
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_WORD_ID_PATTERN = re.compile(r"[1-9][0-9]*")
_MULTIWORD_ID_PATTERN = re.compile(
    r"(?P<start>[1-9][0-9]*)-(?P<end>[1-9][0-9]*)"
)
_EMPTY_NODE_ID_PATTERN = re.compile(r"[1-9][0-9]*\.[1-9][0-9]*")
_OPEN_CELL_PATTERN = re.compile(r"(?:\([^()*]+)*")
_OPEN_LABEL_PATTERN = re.compile(r"\(([^()*]+)")


class EWTSourceError(ValueError):
    """Raised when a pinned source checkout violates its structural contract."""


class EWTPreparationError(ValueError):
    """Raised when an EWT build does not pass the write gate."""


class _ColumnFailure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _UDSentence:
    words: tuple[str, ...]
    lemmas: tuple[str, ...]
    xpos: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SkeletonRow:
    xpos: str
    lemma: str
    roleset: str
    role_cells: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SkeletonSentence:
    rows: tuple[_SkeletonRow, ...]

    @property
    def predicate_rows(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, row in enumerate(self.rows)
            if row.lemma != "-" or row.roleset != "-"
        )

    @property
    def role_width(self) -> int:
        return len(self.rows[0].role_cells)


@dataclass(frozen=True, slots=True)
class _ColumnAnalysis:
    column_index: int
    metadata_index: int
    predicate_index: int
    primary_v_width: int
    spans: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class EWTDatasetBuild:
    """Final leakage-safe examples and corpus-text-free aggregate audit."""

    examples: tuple[WordLevelSRLExample, ...]
    report: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EWTPreparationReceipt:
    """Result of writing canonical prepared splits and source provenance."""

    report: Mapping[str, object]
    manifest: DatasetManifest
    provenance_sha256: str


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
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


def _require_directory(path: Path, *, label: str) -> Path:
    if not path.is_dir():
        raise EWTSourceError(f"{label} root is not a directory")
    if path.is_symlink():
        raise EWTSourceError(f"{label} root cannot be a symbolic link")
    return path.resolve()


def _read_source_text(path: Path, *, root: Path, label: str) -> str:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EWTSourceError(f"missing {label}") from exc
    if not resolved.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise EWTSourceError(f"unsafe {label}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EWTSourceError(f"{label} is not valid UTF-8") from exc


def _run_git(root: Path, arguments: Sequence[str], *, label: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", os.fspath(root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EWTSourceError(f"could not inspect {label} Git checkout") from exc
    if completed.returncode != 0:
        raise EWTSourceError(f"could not inspect {label} Git checkout")
    return completed.stdout.strip()


def _verify_git_source(
    root: Path,
    *,
    expected_commit: str,
    relevant_paths: Sequence[str],
    label: str,
) -> str:
    if not isinstance(expected_commit, str) or _COMMIT_PATTERN.fullmatch(
        expected_commit
    ) is None:
        raise ValueError(f"expected {label} commit must be a lowercase Git SHA-1")
    actual_commit = _run_git(root, ("rev-parse", "--verify", "HEAD"), label=label)
    if actual_commit != expected_commit:
        raise EWTSourceError(
            f"{label} revision does not match the immutable source pin"
        )
    status = _run_git(
        root,
        ("status", "--porcelain", "--untracked-files=all", "--", *relevant_paths),
        label=label,
    )
    if status:
        raise EWTSourceError(
            f"{label} relevant source files are modified or untracked"
        )
    return actual_commit


def _normalize_propbank_document(raw_path: str) -> str:
    prefix = "google/ewt/"
    if not raw_path.startswith(prefix):
        raise EWTSourceError("PropBank EWT document path has an unexpected prefix")
    relative = raw_path[len(prefix) :]
    parts = relative.split("/")
    if len(parts) != 3 or parts[1] != "00" or not all(parts):
        raise EWTSourceError("PropBank EWT document path has an unexpected shape")
    genre, _, filename = parts
    if filename.endswith(".conllu"):
        filename = filename[: -len(".conllu")]
    if not filename.endswith(".xml"):
        raise EWTSourceError("PropBank EWT document path must end in .xml")
    filename = filename[: -len(".xml")]
    if not filename:
        raise EWTSourceError("PropBank EWT document filename is empty")
    return f"{genre}-{filename}"


def _parse_nonnegative_integer(raw: str, *, label: str) -> int:
    if not raw.isascii() or not raw.isdecimal():
        raise EWTSourceError(f"{label} must be a non-negative integer")
    return int(raw)


def _read_ud_documents(
    root: Path,
) -> tuple[
    dict[str, tuple[_UDSentence, ...]],
    dict[str, DatasetSplit],
    dict[str, object],
]:
    documents: dict[str, tuple[_UDSentence, ...]] = {}
    document_splits: dict[str, DatasetSplit] = {}
    sentence_counts: Counter[DatasetSplit] = Counter()
    token_counts: Counter[DatasetSplit] = Counter()
    document_counts: Counter[DatasetSplit] = Counter()

    for split, filename in _UD_FILENAMES:
        path = root / filename
        source = _read_source_text(path, root=root, label=filename)
        current_document: str | None = None
        current_sentences: list[_UDSentence] = []
        words: list[str] = []
        lemmas: list[str] = []
        xpos: list[str] = []
        expected_token_id = 1

        def flush_sentence() -> None:
            nonlocal words, lemmas, xpos, expected_token_id
            if not words:
                return
            current_sentences.append(
                _UDSentence(tuple(words), tuple(lemmas), tuple(xpos))
            )
            sentence_counts[split] += 1
            token_counts[split] += len(words)
            words = []
            lemmas = []
            xpos = []
            expected_token_id = 1

        def flush_document() -> None:
            nonlocal current_document, current_sentences
            flush_sentence()
            if current_document is None:
                return
            if not current_sentences:
                raise EWTSourceError("UD EWT document has no word-level sentences")
            if current_document in documents:
                raise EWTSourceError("UD EWT contains a duplicate document ID")
            documents[current_document] = tuple(current_sentences)
            document_splits[current_document] = split
            document_counts[split] += 1
            current_document = None
            current_sentences = []

        for line_number, raw_line in enumerate(source.splitlines(), start=1):
            if raw_line.startswith("# newdoc id = "):
                flush_document()
                current_document = raw_line.removeprefix("# newdoc id = ").strip()
                if not current_document:
                    raise EWTSourceError(f"{filename}:{line_number} has an empty ID")
            elif not raw_line:
                flush_sentence()
            elif raw_line.startswith("#"):
                continue
            else:
                if current_document is None:
                    raise EWTSourceError(
                        f"{filename}:{line_number} has a token outside a document"
                    )
                fields = raw_line.split("\t")
                if len(fields) != 10:
                    raise EWTSourceError(
                        f"{filename}:{line_number} is not ten-column CoNLL-U"
                    )
                token_id = fields[0]
                multiword_match = _MULTIWORD_ID_PATTERN.fullmatch(token_id)
                if multiword_match is not None:
                    if int(multiword_match.group("start")) >= int(
                        multiword_match.group("end")
                    ):
                        raise EWTSourceError(
                            f"{filename}:{line_number} has an invalid token range"
                        )
                    continue
                if _EMPTY_NODE_ID_PATTERN.fullmatch(token_id) is not None:
                    continue
                if _WORD_ID_PATTERN.fullmatch(token_id) is None:
                    raise EWTSourceError(
                        f"{filename}:{line_number} has an invalid token ID"
                    )
                parsed_token_id = _parse_nonnegative_integer(
                    token_id, label="UD token ID"
                )
                if parsed_token_id != expected_token_id:
                    raise EWTSourceError(
                        f"{filename}:{line_number} has non-contiguous word IDs"
                    )
                if not fields[1] or not fields[2] or not fields[4]:
                    raise EWTSourceError(
                        f"{filename}:{line_number} has an empty word, lemma, or XPOS"
                    )
                words.append(fields[1])
                lemmas.append(fields[2])
                xpos.append(fields[4])
                expected_token_id += 1
        flush_document()

    report = {
        "documents": sum(document_counts.values()),
        "sentences": sum(sentence_counts.values()),
        "tokens": sum(token_counts.values()),
        "documents_by_split": {
            split: document_counts[split] for split in _SPLIT_ORDER
        },
        "sentences_by_split": {
            split: sentence_counts[split] for split in _SPLIT_ORDER
        },
        "tokens_by_split": {
            split: token_counts[split] for split in _SPLIT_ORDER
        },
    }
    return documents, document_splits, report


def _read_official_splits(root: Path) -> dict[str, DatasetSplit]:
    split_directory = root / "docs" / "evaluation"
    assignments: dict[str, DatasetSplit] = {}
    for split, filename in _PB_SPLIT_FILENAMES:
        source = _read_source_text(
            split_directory / filename,
            root=root,
            label=f"docs/evaluation/{filename}",
        )
        count = 0
        for line_number, raw_line in enumerate(source.splitlines(), start=1):
            path = raw_line.strip()
            if not path:
                continue
            try:
                document_id = _normalize_propbank_document("google/ewt/" + path)
            except EWTSourceError as exc:
                raise EWTSourceError(
                    f"{filename}:{line_number} has an invalid document path"
                ) from exc
            if document_id in assignments:
                raise EWTSourceError(
                    "PropBank EWT split lists contain a duplicate document"
                )
            assignments[document_id] = split
            count += 1
        if count == 0:
            raise EWTSourceError(f"PropBank EWT {split} split list is empty")
    return assignments


def _read_skeleton_documents(
    root: Path,
) -> tuple[dict[str, tuple[_SkeletonSentence, ...]], dict[str, object]]:
    skeleton_root = root / "data" / "google" / "ewt"
    if not skeleton_root.is_dir() or skeleton_root.is_symlink():
        raise EWTSourceError("missing or unsafe PropBank EWT skeleton directory")
    paths = sorted(skeleton_root.rglob("*.gold_skel"))
    if not paths:
        raise EWTSourceError("PropBank EWT checkout contains no .gold_skel files")
    actual_relative_paths = {
        path.relative_to(root).as_posix() for path in paths
    }
    tracked_relative_paths = {
        line
        for line in _run_git(
            root,
            ("ls-files", "--", "data/google/ewt"),
            label="PropBank release",
        ).splitlines()
        if line.endswith(".gold_skel")
    }
    if actual_relative_paths != tracked_relative_paths:
        raise EWTSourceError(
            "PropBank EWT skeleton inventory differs from tracked source files"
        )

    documents: dict[str, tuple[_SkeletonSentence, ...]] = {}
    total_sentences = 0
    total_tokens = 0
    total_columns = 0

    for path in paths:
        source = _read_source_text(
            path,
            root=root,
            label="PropBank EWT .gold_skel file",
        )
        relative = path.relative_to(skeleton_root).as_posix()
        raw_file_document = "google/ewt/" + relative[: -len(".gold_skel")]
        file_document_id = _normalize_propbank_document(raw_file_document)
        indexed_sentences: dict[int, dict[int, _SkeletonRow]] = defaultdict(dict)

        for line_number, raw_line in enumerate(source.splitlines(), start=1):
            if not raw_line.strip():
                continue
            fields = raw_line.split()
            if len(fields) < 8:
                raise EWTSourceError(
                    f".gold_skel row {line_number} has fewer than eight columns"
                )
            document_id = _normalize_propbank_document(fields[0])
            if document_id != file_document_id:
                raise EWTSourceError(".gold_skel row does not match its source file")
            sentence_index = _parse_nonnegative_integer(
                fields[1], label="skeleton sentence index"
            )
            token_index = _parse_nonnegative_integer(
                fields[2], label="skeleton token index"
            )
            if fields[3] != "[WORD]":
                raise EWTSourceError(".gold_skel source-word placeholder is invalid")
            if not fields[4] or not fields[5]:
                raise EWTSourceError(".gold_skel row has empty PTB metadata")
            sentence = indexed_sentences[sentence_index]
            if token_index in sentence:
                raise EWTSourceError(".gold_skel contains a duplicate token row")
            sentence[token_index] = _SkeletonRow(
                xpos=fields[4],
                lemma=fields[6],
                roleset=fields[7],
                role_cells=tuple(fields[8:]),
            )

        if not indexed_sentences:
            raise EWTSourceError(".gold_skel document contains no sentences")
        expected_sentence_indexes = set(range(max(indexed_sentences) + 1))
        if set(indexed_sentences) != expected_sentence_indexes:
            raise EWTSourceError(".gold_skel sentence indexes are not contiguous")

        ordered_sentences: list[_SkeletonSentence] = []
        for sentence_index in range(len(indexed_sentences)):
            indexed_rows = indexed_sentences[sentence_index]
            expected_token_indexes = set(range(max(indexed_rows) + 1))
            if set(indexed_rows) != expected_token_indexes:
                raise EWTSourceError(".gold_skel token indexes are not contiguous")
            rows = tuple(indexed_rows[index] for index in range(len(indexed_rows)))
            widths = {len(row.role_cells) for row in rows}
            if len(widths) != 1:
                raise EWTSourceError(
                    ".gold_skel sentence has inconsistent role-column widths"
                )
            sentence = _SkeletonSentence(rows)
            predicate_rows = sentence.predicate_rows
            for row in rows:
                if (row.lemma == "-") != (row.roleset == "-"):
                    raise EWTSourceError(
                        ".gold_skel predicate lemma and roleset are inconsistent"
                    )
            if len(predicate_rows) != sentence.role_width:
                raise EWTSourceError(
                    ".gold_skel predicate metadata does not match role width"
                )
            if any("." not in rows[index].roleset for index in predicate_rows):
                raise EWTSourceError(".gold_skel contains an invalid roleset")
            total_columns += sentence.role_width
            ordered_sentences.append(sentence)
            total_sentences += 1
            total_tokens += len(rows)

        if file_document_id in documents:
            raise EWTSourceError("PropBank EWT contains a duplicate skeleton document")
        documents[file_document_id] = tuple(ordered_sentences)

    report = {
        "documents": len(documents),
        "sentences": total_sentences,
        "tokens": total_tokens,
        "all_predicate_columns": total_columns,
    }
    return documents, report


def _normalized_cell_label(raw_label: str) -> str:
    if raw_label in {"V", "C-V"}:
        return raw_label
    try:
        return normalize_role_label(raw_label).model_label
    except (PropBankAdapterError, TypeError) as exc:
        raise _ColumnFailure("unsupported_role_label") from exc


def _decode_role_column(
    cells: Sequence[str],
) -> tuple[tuple[str, int, int], ...]:
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
        raw_open_labels = _OPEN_LABEL_PATTERN.findall(left)
        if len(raw_open_labels) > 1 or (raw_open_labels and stack):
            raise _ColumnFailure("overlapping_role_spans")
        for raw_label in raw_open_labels:
            stack.append((_normalized_cell_label(raw_label), word_index))
        for _ in right:
            if not stack:
                raise _ColumnFailure("invalid_bracket_sequence")
            label, start = stack.pop()
            spans.append((label, start, word_index + 1))
    if stack:
        raise _ColumnFailure("invalid_bracket_sequence")
    return tuple(spans)


def _analyze_column(
    sentence: _SkeletonSentence,
    *,
    column_index: int,
    metadata_index: int,
) -> _ColumnAnalysis:
    cells = tuple(row.role_cells[column_index] for row in sentence.rows)
    spans = _decode_role_column(cells)
    primary_v_spans = tuple(span for span in spans if span[0] == "V")
    if not primary_v_spans:
        raise _ColumnFailure("missing_relation_span")
    if len(primary_v_spans) != 1:
        raise _ColumnFailure("multiple_primary_relation_spans")
    _, start, end = primary_v_spans[0]
    relation_indexes = {
        index
        for label, span_start, span_end in spans
        if label in {"V", "C-V"}
        for index in range(span_start, span_end)
    }
    if metadata_index not in relation_indexes:
        raise _ColumnFailure("predicate_metadata_outside_relation_span")
    return _ColumnAnalysis(
        column_index=column_index,
        metadata_index=metadata_index,
        predicate_index=start,
        primary_v_width=end - start,
        spans=spans,
    )


def _convert_column(
    *,
    document_id: str,
    sentence_index: int,
    words: tuple[str, ...],
    sentence: _SkeletonSentence,
    analysis: _ColumnAnalysis,
) -> PreparedWordLevelSRLExample:
    column_index = analysis.column_index
    predicate_index = analysis.predicate_index
    spans = analysis.spans
    relation_spans = tuple(span for span in spans if span[0] in {"V", "C-V"})

    tags = ["O"] * len(words)
    for label, start, end in spans:
        if label in {"V", "C-V"}:
            continue
        if any(tag != "O" for tag in tags[start:end]):
            raise _ColumnFailure("overlapping_role_spans")
        tags[start] = f"B-{label}"
        for index in range(start + 1, end):
            tags[index] = f"I-{label}"

    for label, start, end in relation_spans:
        if label == "V" and start != predicate_index:
            raise _ColumnFailure("predicate_outside_relation_span")
        if any(tag != "O" for tag in tags[start:end]):
            raise _ColumnFailure("overlapping_role_spans")
        tags[start] = f"B-{label}"
        for index in range(start + 1, end):
            tags[index] = f"I-{label}"

    source_row = sentence.rows[analysis.metadata_index]
    try:
        return PreparedWordLevelSRLExample(
            example_id=(
                f"ewt:{document_id}:{sentence_index:05d}:"
                f"{predicate_index:04d}:{column_index:03d}"
            ),
            document_id=f"ewt:{document_id}",
            sentence_id=f"ewt:{document_id}:{sentence_index:05d}",
            words=words,
            predicate_index=predicate_index,
            tags=tuple(tags),
            predicate_roleset=source_row.roleset,
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise AssertionError("EWT adapter produced an invalid model record") from exc


def _validate_coverage(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_verbal_predicate_coverage must be numeric")
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(
            "minimum_verbal_predicate_coverage must be between zero and one"
        )
    return resolved


def build_ewt_dataset(
    propbank_root: str | os.PathLike[str],
    ud_ewt_root: str | os.PathLike[str],
    *,
    expected_propbank_commit: str = PROPBANK_RELEASE_COMMIT,
    expected_ud_commit: str = UD_ENGLISH_EWT_COMMIT,
    minimum_verbal_predicate_coverage: float = (
        DEFAULT_MINIMUM_VERBAL_PREDICATE_COVERAGE
    ),
) -> EWTDatasetBuild:
    """Join pinned official releases and build final leakage-safe examples."""

    minimum_coverage = _validate_coverage(minimum_verbal_predicate_coverage)
    propbank_directory = _require_directory(
        Path(propbank_root), label="PropBank release"
    )
    ud_directory = _require_directory(Path(ud_ewt_root), label="UD English EWT")
    propbank_commit = _verify_git_source(
        propbank_directory,
        expected_commit=expected_propbank_commit,
        relevant_paths=("data/google/ewt", "docs/evaluation"),
        label="PropBank release",
    )
    ud_commit = _verify_git_source(
        ud_directory,
        expected_commit=expected_ud_commit,
        relevant_paths=tuple(filename for _, filename in _UD_FILENAMES),
        label="UD English EWT",
    )

    ud_documents, ud_splits, ud_report = _read_ud_documents(ud_directory)
    official_splits = _read_official_splits(propbank_directory)
    skeleton_documents, skeleton_report = _read_skeleton_documents(
        propbank_directory
    )
    column_analyses: dict[
        tuple[str, int], tuple[_ColumnAnalysis, ...]
    ] = {}
    classification_rejections: Counter[str] = Counter()
    verbal_predicate_columns = 0
    nonverbal_predicate_columns = 0
    multiword_primary_v_columns = 0
    primary_v_tokens = 0
    metadata_primary_anchor_differences = 0
    for document_id, sentences in skeleton_documents.items():
        for sentence_index, sentence in enumerate(sentences):
            verbal_analyses: list[_ColumnAnalysis] = []
            for column_index, metadata_index in enumerate(sentence.predicate_rows):
                try:
                    analysis = _analyze_column(
                        sentence,
                        column_index=column_index,
                        metadata_index=metadata_index,
                    )
                except _ColumnFailure as exc:
                    classification_rejections[exc.reason] += 1
                    continue
                primary_v_tokens += analysis.primary_v_width
                if analysis.primary_v_width > 1:
                    multiword_primary_v_columns += 1
                if analysis.predicate_index != analysis.metadata_index:
                    metadata_primary_anchor_differences += 1
                primary_xpos = sentence.rows[analysis.predicate_index].xpos
                if primary_xpos in VERBAL_PENN_TAGS:
                    verbal_predicate_columns += 1
                    verbal_analyses.append(analysis)
                else:
                    nonverbal_predicate_columns += 1
            column_analyses[(document_id, sentence_index)] = tuple(
                verbal_analyses
            )
    skeleton_report = {
        **skeleton_report,
        "verbal_predicate_columns": verbal_predicate_columns,
        "nonverbal_predicate_columns": nonverbal_predicate_columns,
        "unclassifiable_predicate_columns": sum(
            classification_rejections.values()
        ),
        "classification_rejection_reasons": dict(
            sorted(classification_rejections.items())
        ),
        "primary_v_span_tokens": primary_v_tokens,
        "multiword_primary_v_columns": multiword_primary_v_columns,
        "metadata_primary_anchor_differences": (
            metadata_primary_anchor_differences
        ),
    }
    if (
        verbal_predicate_columns
        + nonverbal_predicate_columns
        + sum(classification_rejections.values())
        != int(skeleton_report["all_predicate_columns"])
    ):
        raise AssertionError("EWT predicate classification counts do not reconcile")
    skeleton_ids = set(skeleton_documents)
    ud_ids = set(ud_documents)
    split_ids = set(official_splits)
    if skeleton_ids - split_ids:
        raise EWTSourceError("a PropBank EWT skeleton document has no official split")
    split_disagreements = {
        document_id
        for document_id in skeleton_ids & ud_ids
        if official_splits[document_id] != ud_splits[document_id]
    }
    if split_disagreements:
        raise EWTSourceError("PropBank and UD EWT official splits disagree")

    rejection_counts: Counter[str] = Counter()
    aligned_sentences: list[
        tuple[tuple[str, ...], DatasetSplit, tuple[PreparedWordLevelSRLExample, ...]]
    ] = []
    aligned_sentences_by_split: Counter[DatasetSplit] = Counter()
    examples_before_leakage: Counter[DatasetSplit] = Counter()
    sentence_count_matches = 0
    sentence_count_mismatches = 0
    token_count_matches = 0
    token_count_mismatches = 0
    xpos_exact_matches = 0
    xpos_mismatches = 0
    xpos_token_matches = 0
    xpos_token_mismatches = 0
    lexical_comparisons = 0
    metadata_lemma_matches = 0
    roleset_lemma_matches = 0
    metadata_and_roleset_lemma_matches = 0
    accepted_before_leakage = 0

    for document_id in sorted(skeleton_ids):
        split = official_splits[document_id]
        skeleton_sentences = skeleton_documents[document_id]
        ud_sentences = ud_documents.get(document_id)
        if ud_sentences is None:
            rejection_counts["document_missing_ud"] += sum(
                len(column_analyses[(document_id, sentence_index)])
                for sentence_index in range(len(skeleton_sentences))
            )
            continue
        if len(skeleton_sentences) != len(ud_sentences):
            sentence_count_mismatches += 1
            rejection_counts["document_sentence_count_mismatch"] += sum(
                len(column_analyses[(document_id, sentence_index)])
                for sentence_index in range(len(skeleton_sentences))
            )
            continue
        sentence_count_matches += 1

        for sentence_index, (skeleton_sentence, ud_sentence) in enumerate(
            zip(skeleton_sentences, ud_sentences, strict=True)
        ):
            verbal_columns = column_analyses[(document_id, sentence_index)]
            if len(skeleton_sentence.rows) != len(ud_sentence.words):
                token_count_mismatches += 1
                rejection_counts["token_count_mismatch"] += len(verbal_columns)
                continue
            token_count_matches += 1
            aligned_sentences_by_split[split] += 1
            skeleton_xpos = tuple(row.xpos for row in skeleton_sentence.rows)
            if skeleton_xpos == ud_sentence.xpos:
                xpos_exact_matches += 1
            else:
                xpos_mismatches += 1
            for skeleton_pos, ud_pos in zip(
                skeleton_xpos, ud_sentence.xpos, strict=True
            ):
                if skeleton_pos == ud_pos:
                    xpos_token_matches += 1
                else:
                    xpos_token_mismatches += 1

            sentence_examples: list[PreparedWordLevelSRLExample] = []
            for analysis in verbal_columns:
                predicate_index = analysis.predicate_index
                metadata_row = skeleton_sentence.rows[analysis.metadata_index]
                ud_lemma = ud_sentence.lemmas[predicate_index].casefold()
                metadata_match = metadata_row.lemma.casefold() == ud_lemma
                roleset_stem = metadata_row.roleset.rsplit(".", 1)[0].casefold()
                roleset_match = roleset_stem == ud_lemma
                lexical_comparisons += 1
                metadata_lemma_matches += int(metadata_match)
                roleset_lemma_matches += int(roleset_match)
                metadata_and_roleset_lemma_matches += int(
                    metadata_match and roleset_match
                )
                try:
                    example = _convert_column(
                        document_id=document_id,
                        sentence_index=sentence_index,
                        words=ud_sentence.words,
                        sentence=skeleton_sentence,
                        analysis=analysis,
                    )
                except _ColumnFailure as exc:
                    rejection_counts[exc.reason] += 1
                    continue
                sentence_examples.append(example)
                accepted_before_leakage += 1
                examples_before_leakage[split] += 1
            aligned_sentences.append(
                (ud_sentence.words, split, tuple(sentence_examples))
            )

    all_verbal_columns = verbal_predicate_columns
    rejected_verbal_columns = sum(rejection_counts.values())
    if accepted_before_leakage + rejected_verbal_columns != all_verbal_columns:
        raise AssertionError("EWT verbal-predicate conversion counts do not reconcile")
    conversion_coverage = (
        accepted_before_leakage / all_verbal_columns if all_verbal_columns else 0.0
    )

    word_sequence_splits: dict[tuple[str, ...], set[DatasetSplit]] = defaultdict(set)
    for words, split, _ in aligned_sentences:
        word_sequence_splits[words].add(split)
    cross_split_sequences = {
        words
        for words, splits in word_sequence_splits.items()
        if len(splits) > 1
    }
    leakage_sentence_exclusions: Counter[DatasetSplit] = Counter()
    leakage_example_exclusions: Counter[DatasetSplit] = Counter()
    leakage_safe_examples: list[WordLevelSRLExample] = []
    for words, split, sentence_examples in aligned_sentences:
        if words in cross_split_sequences:
            leakage_sentence_exclusions[split] += 1
            leakage_example_exclusions[split] += len(sentence_examples)
            continue
        leakage_safe_examples.extend(
            example.assign_split(split) for example in sentence_examples
        )

    examples_by_input: dict[
        tuple[DatasetSplit, tuple[str, ...], int], list[WordLevelSRLExample]
    ] = defaultdict(list)
    for example in leakage_safe_examples:
        input_key = (example.split, example.words, example.predicate_index)
        examples_by_input[input_key].append(example)
    conflicting_inputs: set[tuple[DatasetSplit, tuple[str, ...], int]] = set()
    for input_key, grouped_examples in examples_by_input.items():
        if len({example.tags for example in grouped_examples}) > 1:
            conflicting_inputs.add(input_key)

    conflicting_input_groups: Counter[DatasetSplit] = Counter()
    conflicting_input_exclusions: Counter[DatasetSplit] = Counter()
    conflict_safe_examples: list[WordLevelSRLExample] = []
    for input_key, grouped_examples in examples_by_input.items():
        split = input_key[0]
        if input_key in conflicting_inputs:
            conflicting_input_groups[split] += 1
            conflicting_input_exclusions[split] += len(grouped_examples)
            continue
        conflict_safe_examples.extend(grouped_examples)

    same_source_groups: dict[
        tuple[str, str, int], list[WordLevelSRLExample]
    ] = defaultdict(list)
    for example in conflict_safe_examples:
        source_identity = (
            example.document_id,
            example.sentence_id,
            example.predicate_index,
        )
        same_source_groups[source_identity].append(example)

    same_source_groups_deduplicated: Counter[DatasetSplit] = Counter()
    same_source_duplicates_removed: Counter[DatasetSplit] = Counter()
    source_unique_examples: list[WordLevelSRLExample] = []
    for grouped_examples in same_source_groups.values():
        ordered = sorted(grouped_examples, key=lambda item: item.example_id)
        split = ordered[0].split
        if any(example.split != split for example in ordered):
            raise AssertionError("one EWT source identity spans multiple splits")
        if len({example.words for example in ordered}) != 1:
            raise AssertionError("one EWT source identity has inconsistent words")
        if len({example.tags for example in ordered}) != 1:
            raise AssertionError(
                "conflicting same-source targets survived the conflict policy"
            )
        if len(ordered) > 1:
            same_source_groups_deduplicated[split] += 1
            same_source_duplicates_removed[split] += len(ordered) - 1
        source_unique_examples.append(ordered[0])

    exact_duplicates_removed: Counter[DatasetSplit] = Counter()
    seen_eval_semantics: dict[
        DatasetSplit, set[tuple[tuple[str, ...], int, tuple[str, ...]]]
    ] = defaultdict(set)
    final_examples: list[WordLevelSRLExample] = []
    for example in sorted(source_unique_examples, key=lambda item: item.example_id):
        exact_semantics = (example.words, example.predicate_index, example.tags)
        if (
            example.split != "train"
            and exact_semantics in seen_eval_semantics[example.split]
        ):
            exact_duplicates_removed[example.split] += 1
            continue
        seen_eval_semantics[example.split].add(exact_semantics)
        final_examples.append(example)

    final_counts = Counter(example.split for example in final_examples)
    if accepted_before_leakage != (
        len(final_examples)
        + sum(leakage_example_exclusions.values())
        + sum(conflicting_input_exclusions.values())
        + sum(same_source_duplicates_removed.values())
        + sum(exact_duplicates_removed.values())
    ):
        raise AssertionError("EWT duplicate-policy counts do not reconcile")
    training_examples = tuple(
        example for example in final_examples if example.split == "train"
    )
    vocabulary_size = 0
    missing_evaluation_labels: dict[DatasetSplit, int] = {
        "development": 0,
        "test": 0,
        "train": 0,
    }
    label_compatibility_status = "fail"
    if training_examples:
        vocabulary = build_training_label_vocabulary(training_examples)
        vocabulary_size = len(vocabulary)
        for split in ("development", "test"):
            required_labels = {
                required
                for example in final_examples
                if example.split == split
                for tag in example.tags
                for required in (tag, continuation_tag(tag))
            }
            missing_evaluation_labels[split] = len(
                required_labels.difference(vocabulary.label_to_id)
            )
        label_compatibility_status = (
            "pass"
            if all(
                missing_evaluation_labels[split] == 0
                for split in ("development", "test")
            )
            else "fail"
        )

    conversion_status = (
        "pass" if conversion_coverage >= minimum_coverage else "fail"
    )
    nonempty_status = (
        "pass" if all(final_counts[split] > 0 for split in _SPLIT_ORDER) else "fail"
    )
    overall_status = (
        "pass"
        if (
            not classification_rejections
            and conversion_status == "pass"
            and label_compatibility_status == "pass"
            and nonempty_status == "pass"
        )
        else "fail"
    )
    minimum_accepted = math.ceil(all_verbal_columns * minimum_coverage)

    report: Mapping[str, object] = {
        "schema_version": EWT_AUDIT_SCHEMA_VERSION,
        "source": {
            "propbank_release_commit": propbank_commit,
            "ud_english_ewt_commit": ud_commit,
            "ud_english_ewt_tag": UD_ENGLISH_EWT_TAG,
            "relevant_worktrees": "clean",
            "join_policy": EWT_JOIN_POLICY,
            "duplicate_policy": EWT_DUPLICATE_POLICY,
        },
        "ud": ud_report,
        "propbank_skeleton": skeleton_report,
        "join": {
            "shared_documents": len(skeleton_ids & ud_ids),
            "propbank_only_documents": len(skeleton_ids - ud_ids),
            "ud_only_documents": len(ud_ids - skeleton_ids),
            "split_list_only_documents": len(split_ids - skeleton_ids),
            "split_disagreements": 0,
            "document_sentence_count_matches": sentence_count_matches,
            "document_sentence_count_mismatches": sentence_count_mismatches,
            "token_count_matches": token_count_matches,
            "token_count_mismatches": token_count_mismatches,
            "xpos_exact_matches": xpos_exact_matches,
            "xpos_mismatches_at_equal_width": xpos_mismatches,
            "xpos_token_matches": xpos_token_matches,
            "xpos_token_mismatches": xpos_token_mismatches,
            "xpos_token_agreement": (
                xpos_token_matches
                / (xpos_token_matches + xpos_token_mismatches)
                if xpos_token_matches + xpos_token_mismatches
                else 0.0
            ),
            "aligned_sentences_by_split": {
                split: aligned_sentences_by_split[split] for split in _SPLIT_ORDER
            },
        },
        "conversion": {
            "eligible_verbal_predicate_columns": all_verbal_columns,
            "accepted_before_leakage_policy": accepted_before_leakage,
            "rejected_verbal_predicate_columns": rejected_verbal_columns,
            "rejection_reasons": dict(sorted(rejection_counts.items())),
            "conversion_coverage": conversion_coverage,
            "minimum_accepted_verbal_predicate_columns": minimum_accepted,
            "coverage_margin_verbal_predicate_columns": (
                accepted_before_leakage - minimum_accepted
            ),
        },
        "lexical_diagnostic": {
            "policy": (
                "Unicode-casefold exact comparison at primary-V start; "
                "roleset lemma is text before its final period"
            ),
            "hard_gate": False,
            "compared_aligned_verbal_predicates": lexical_comparisons,
            "metadata_lemma_exact_matches": metadata_lemma_matches,
            "metadata_lemma_exact_mismatches": (
                lexical_comparisons - metadata_lemma_matches
            ),
            "metadata_lemma_exact_match_rate": (
                metadata_lemma_matches / lexical_comparisons
                if lexical_comparisons
                else 0.0
            ),
            "roleset_lemma_exact_matches": roleset_lemma_matches,
            "roleset_lemma_exact_mismatches": (
                lexical_comparisons - roleset_lemma_matches
            ),
            "roleset_lemma_exact_match_rate": (
                roleset_lemma_matches / lexical_comparisons
                if lexical_comparisons
                else 0.0
            ),
            "both_metadata_and_roleset_exact_matches": (
                metadata_and_roleset_lemma_matches
            ),
        },
        "label_vocabulary": {
            "policy": "training-only-with-continuation-closure/v1",
            "size": vocabulary_size,
            "missing_labels": {
                split: missing_evaluation_labels[split]
                for split in ("development", "test")
            },
            "development_test_compatibility": label_compatibility_status,
        },
        "split": {
            "policy": "official-propbank-ewt-document-splits/v1",
            "duplicate_policy": EWT_DUPLICATE_POLICY,
            "examples_before_leakage_policy": {
                split: examples_before_leakage[split] for split in _SPLIT_ORDER
            },
            "cross_split_word_sequence_groups": len(cross_split_sequences),
            "cross_split_sentence_exclusions": {
                split: leakage_sentence_exclusions[split]
                for split in _SPLIT_ORDER
            },
            "cross_split_example_exclusions": {
                split: leakage_example_exclusions[split] for split in _SPLIT_ORDER
            },
            "conflicting_identical_input_groups": {
                split: conflicting_input_groups[split] for split in _SPLIT_ORDER
            },
            "conflicting_identical_input_exclusions": {
                split: conflicting_input_exclusions[split]
                for split in _SPLIT_ORDER
            },
            "same_source_identical_target_groups_deduplicated": {
                split: same_source_groups_deduplicated[split]
                for split in _SPLIT_ORDER
            },
            "same_source_identical_target_duplicates_removed": {
                split: same_source_duplicates_removed[split]
                for split in _SPLIT_ORDER
            },
            "dev_test_cross_source_exact_semantic_duplicates_removed": {
                split: exact_duplicates_removed[split] for split in _SPLIT_ORDER
            },
            "final_examples": {
                split: final_counts[split] for split in _SPLIT_ORDER
            },
        },
        "gate": {
            "source_integrity": "pass",
            "predicate_column_classification": (
                "pass" if not classification_rejections else "fail"
            ),
            "official_split_agreement": "pass",
            "minimum_verbal_predicate_coverage": minimum_coverage,
            "conversion": conversion_status,
            "cross_split_word_sequence_leakage": "pass",
            "conflicting_identical_inputs": "pass",
            "same_source_semantic_identity_uniqueness": "pass",
            "development_test_exact_semantic_duplicates": "pass",
            "development_test_label_compatibility": (
                label_compatibility_status
            ),
            "nonempty_final_splits": nonempty_status,
            "status": overall_status,
        },
    }
    return EWTDatasetBuild(
        examples=tuple(sorted(final_examples, key=lambda item: item.example_id)),
        report=report,
    )


def audit_ewt_dataset(
    propbank_root: str | os.PathLike[str],
    ud_ewt_root: str | os.PathLike[str],
    *,
    expected_propbank_commit: str = PROPBANK_RELEASE_COMMIT,
    expected_ud_commit: str = UD_ENGLISH_EWT_COMMIT,
    minimum_verbal_predicate_coverage: float = (
        DEFAULT_MINIMUM_VERBAL_PREDICATE_COVERAGE
    ),
) -> Mapping[str, object]:
    """Return only aggregate, JSON-serializable audit information."""

    return build_ewt_dataset(
        propbank_root,
        ud_ewt_root,
        expected_propbank_commit=expected_propbank_commit,
        expected_ud_commit=expected_ud_commit,
        minimum_verbal_predicate_coverage=minimum_verbal_predicate_coverage,
    ).report


def _provenance_payload(
    report: Mapping[str, object], manifest: DatasetManifest
) -> dict[str, object]:
    return {
        "schema_version": EWT_PROVENANCE_SCHEMA_VERSION,
        "dataset": {
            "schema_version": manifest.schema_version,
            "record_counts": dict(manifest.record_counts),
            "file_sha256": dict(manifest.file_sha256),
            "dataset_fingerprint": manifest.dataset_fingerprint,
        },
        "audit": report,
    }


def _gate_status(report: Mapping[str, object]) -> str:
    gate = report.get("gate")
    if not isinstance(gate, Mapping) or gate.get("status") not in {"pass", "fail"}:
        raise AssertionError("EWT audit report has an invalid gate status")
    status = gate["status"]
    assert isinstance(status, str)
    return status


def _preparation_receipt_payload(
    receipt: EWTPreparationReceipt,
) -> dict[str, object]:
    return {
        "schema_version": EWT_PREPARATION_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "audit": receipt.report,
        "prepared_dataset": {
            "schema_version": receipt.manifest.schema_version,
            "record_counts": dict(receipt.manifest.record_counts),
            "file_sha256": dict(receipt.manifest.file_sha256),
            "dataset_fingerprint": receipt.manifest.dataset_fingerprint,
        },
        "provenance": {
            "filename": EWT_PROVENANCE_FILENAME,
            "sha256": receipt.provenance_sha256,
        },
        "output_filenames": (
            "train.jsonl",
            "development.jsonl",
            "test.jsonl",
            "manifest.json",
            EWT_PROVENANCE_FILENAME,
        ),
    }


def prepare_ewt_dataset(
    propbank_root: str | os.PathLike[str],
    ud_ewt_root: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    expected_propbank_commit: str = PROPBANK_RELEASE_COMMIT,
    expected_ud_commit: str = UD_ENGLISH_EWT_COMMIT,
    minimum_verbal_predicate_coverage: float = (
        DEFAULT_MINIMUM_VERBAL_PREDICATE_COVERAGE
    ),
    output_path_policy: Callable[[Path], None] | None = None,
) -> EWTPreparationReceipt:
    """Gate, prepare, and write canonical splits plus pinned provenance."""

    requested_output_path = Path(output_directory)
    if requested_output_path.exists() or requested_output_path.is_symlink():
        raise EWTPreparationError("EWT output directory must be a new path")
    output_path = requested_output_path.resolve()
    resolved_output_policy = (
        _require_ignored_output_path
        if output_path_policy is None
        else output_path_policy
    )
    if not callable(resolved_output_policy):
        raise TypeError("output_path_policy must be callable")
    resolved_output_policy(output_path)
    build = build_ewt_dataset(
        propbank_root,
        ud_ewt_root,
        expected_propbank_commit=expected_propbank_commit,
        expected_ud_commit=expected_ud_commit,
        minimum_verbal_predicate_coverage=minimum_verbal_predicate_coverage,
    )
    if _gate_status(build.report) != "pass":
        raise EWTPreparationError(
            "EWT preparation gate failed; no prepared dataset was written"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".staging",
        )
    )
    try:
        manifest = write_prepared_dataset(staging_path, build.examples)
        provenance_bytes = _canonical_json_bytes(
            _provenance_payload(build.report, manifest)
        )
        _atomic_write(staging_path / EWT_PROVENANCE_FILENAME, provenance_bytes)
        os.replace(staging_path, output_path)
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path)
    return EWTPreparationReceipt(
        report=build.report,
        manifest=manifest,
        provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
    )


def _require_ignored_output_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise EWTPreparationError("EWT output directory must be a new path")
    repository_probe = path.parent.resolve()
    while (
        not repository_probe.exists()
        and repository_probe != repository_probe.parent
    ):
        repository_probe = repository_probe.parent
    try:
        repository = Path(
            _run_git(
                repository_probe,
                ("rev-parse", "--show-toplevel"),
                label="project",
            )
        ).resolve()
    except EWTSourceError as exc:
        raise EWTPreparationError(
            "preparation must run inside the project Git repository"
        ) from exc
    resolved = path.resolve()
    if not resolved.is_relative_to(repository):
        raise EWTPreparationError(
            "prepared output must be inside the project repository"
        )
    relative = resolved.relative_to(repository).as_posix()
    try:
        completed = subprocess.run(
            ("git", "-C", os.fspath(repository), "check-ignore", "-q", "--", relative),
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EWTPreparationError(
            "could not verify the prepared output ignore policy"
        ) from exc
    if completed.returncode != 0:
        raise EWTPreparationError(
            "prepared output must be covered by the repository ignore policy"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Audit pinned checkouts and optionally write ignored prepared splits."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("propbank_root", type=Path)
    parser.add_argument("ud_ewt_root", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args(argv)

    reserved_paths: set[Path] = set()
    if arguments.output_directory is not None:
        try:
            _require_ignored_output_path(arguments.output_directory)
        except EWTPreparationError as exc:
            parser.error(str(exc))
        reserved_paths = {
            (arguments.output_directory / filename).resolve()
            for filename in (
                "train.jsonl",
                "development.jsonl",
                "test.jsonl",
                "manifest.json",
                EWT_PROVENANCE_FILENAME,
            )
        }
    if arguments.report is not None:
        if arguments.report.exists() or arguments.report.is_symlink():
            parser.error("--report must be a new path")
        if (
            arguments.output_directory is not None
            and arguments.report.resolve() == arguments.output_directory.resolve()
        ):
            parser.error("--report cannot be the prepared output directory")
        if arguments.report.resolve() in reserved_paths:
            parser.error("--report cannot overwrite a prepared dataset file")

    try:
        if arguments.output_directory is None:
            report = audit_ewt_dataset(
                arguments.propbank_root, arguments.ud_ewt_root
            )
            payload: Mapping[str, object] = report
        else:
            receipt = prepare_ewt_dataset(
                arguments.propbank_root,
                arguments.ud_ewt_root,
                arguments.output_directory,
            )
            report = receipt.report
            payload = _preparation_receipt_payload(receipt)
    except (EWTPreparationError, EWTSourceError) as exc:
        parser.error(str(exc))
    contents = _canonical_json_bytes(payload)
    if arguments.report is None:
        print(contents.decode("utf-8"), end="")
    else:
        _atomic_write(arguments.report, contents)
    return 0 if _gate_status(report) == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
