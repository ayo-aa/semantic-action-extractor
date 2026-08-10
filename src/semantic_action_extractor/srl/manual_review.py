"""Privacy-safe manual review workflow for prepared BabySRL examples.

Review packages contain corpus text and therefore may be written only beneath
the repository's explicitly ignored ``data/review/`` directory.  The public
decision boundary contains aggregate counts and cryptographic fingerprints,
never sentences, tokens, tags, rolesets, source IDs, or reviewer notes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

from .babysrl import (
    BABYSRL_ARCHIVE_SHA256,
    BABYSRL_ARCHIVE_SIZE_BYTES,
    build_babysrl_dataset,
)
from .dataset_io import DatasetFormatError, read_prepared_dataset
from .example import WordLevelSRLExample


TALKBANK_ACCESS_URL = "https://talkbank.org/childes/access.html"
TALKBANK_GROUND_RULES_URL = "https://talkbank.org/0share/rules.html"
SELECTION_POLICY = "babysrl-manual-stratified-sha256/v1"
REVIEW_MANIFEST_SCHEMA_VERSION = (
    "semantic-action-extractor.babysrl-manual-review-manifest/v1"
)
REVIEW_ITEM_SCHEMA_VERSION = (
    "semantic-action-extractor.babysrl-manual-review-item/v1"
)
REVIEW_DECISION_SCHEMA_VERSION = (
    "semantic-action-extractor.babysrl-manual-review-decision/v1"
)
REVIEW_AGGREGATE_SCHEMA_VERSION = (
    "semantic-action-extractor.babysrl-manual-review-aggregate/v1"
)

REVIEW_ITEMS_FILENAME = "review_items.jsonl"
REVIEW_DECISIONS_FILENAME = "review_decisions.jsonl"
REVIEW_MANIFEST_FILENAME = "review_manifest.json"
REVIEW_AGGREGATE_FILENAME = "aggregate_decision.json"

_SPLIT_ORDER = {"train": 0, "development": 1, "test": 2}
_CHILD_ORDER = {"adam": 0, "eve": 1, "sarah": 2}
_SHAPE_ORDER = {
    "continued_relation": 0,
    "discontinuous_argument": 1,
    "multiword_span": 2,
    "single_token_spans": 3,
}
_DECISIONS = ("approve", "reject", "uncertain", "pending")
_REASON_CODES = (
    "predicate_head_error",
    "argument_boundary_error",
    "role_label_error",
    "bio_encoding_error",
    "source_or_display_uncertain",
    "other_structured_issue",
)
_CHILD_DOCUMENT_PATTERN = re.compile(r"babysrl:(adam|eve|sarah):[^\s]+")
_EXAMPLE_ID_PATTERN = re.compile(
    r"babysrl:(?P<child>adam|eve|sarah):"
    r"(?P<document>(?:adam|eve)[0-9]{2}|sarah[0-9]{3}):"
    r"u(?P<utterance>[0-9]{5}):p(?P<proposition>[0-9]{2})"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "archive_sha256",
        "archive_size_bytes",
        "dataset_fingerprint",
        "selection_policy",
        "per_stratum",
        "sample_size",
        "population_counts_by_stratum",
        "sample_counts_by_stratum",
        "items_sha256",
        "sample_fingerprint",
        "prepared_raw_identity",
        "access_confirmation",
    }
)
_ITEM_KEYS = frozenset(
    {
        "schema_version",
        "review_id",
        "stratum",
        "predicate_index",
        "predicate_roleset",
        "rows",
    }
)
_DECISION_KEYS = frozenset(
    {"schema_version", "review_id", "decision", "reason_codes"}
)
_ITEM_ROW_KEYS = frozenset(
    {"word", "predicate_marker", "raw_role_cell", "prepared_tag"}
)


class ManualReviewError(ValueError):
    """Base error for a rejected manual-review operation."""


class ReviewAuthorizationError(ManualReviewError):
    """Raised when TalkBank access and rules were not explicitly confirmed."""


class ReviewPathError(ManualReviewError):
    """Raised when a data-bearing path could enter version control."""


class ReviewFormatError(ManualReviewError):
    """Raised when a review package violates its machine-readable schema."""


class ReviewIntegrityError(ManualReviewError):
    """Raised when a review package no longer matches its fingerprint."""


@dataclass(frozen=True, slots=True)
class TalkBankReviewAuthorization:
    """Explicit, non-credentialed acknowledgement required for corpus review."""

    registration_confirmed: bool
    ground_rules_confirmed: bool
    accepted_on: str

    def __post_init__(self) -> None:
        if self.registration_confirmed is not True:
            raise ReviewAuthorizationError(
                "TalkBank registration and sign-in must be explicitly confirmed"
            )
        if self.ground_rules_confirmed is not True:
            raise ReviewAuthorizationError(
                "acceptance of the current TalkBank ground rules must be "
                "explicitly confirmed"
            )
        if not isinstance(self.accepted_on, str):
            raise ReviewAuthorizationError(
                "the TalkBank rules acceptance date must be an ISO date"
            )
        try:
            parsed = date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ReviewAuthorizationError(
                "the TalkBank rules acceptance date must be YYYY-MM-DD"
            ) from exc
        if parsed.isoformat() != self.accepted_on:
            raise ReviewAuthorizationError(
                "the TalkBank rules acceptance date must be canonical YYYY-MM-DD"
            )

    def as_payload(self) -> dict[str, object]:
        """Return the credential-free acknowledgement persisted in the package."""

        return {
            "registration_confirmed": True,
            "ground_rules_confirmed": True,
            "accepted_on": self.accepted_on,
            "access_url": TALKBANK_ACCESS_URL,
            "ground_rules_url": TALKBANK_GROUND_RULES_URL,
        }


@dataclass(frozen=True, slots=True)
class ManualReviewReceipt:
    """Corpus-free receipt returned after deterministic package creation."""

    archive_sha256: str
    dataset_fingerprint: str
    sample_fingerprint: str
    sample_size: int
    sample_counts_by_stratum: Mapping[str, int]

    def as_payload(self) -> dict[str, object]:
        return {
            "status": "review_pending",
            "archive_sha256": self.archive_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "sample_fingerprint": self.sample_fingerprint,
            "sample_size": self.sample_size,
            "sample_counts_by_stratum": dict(self.sample_counts_by_stratum),
        }


@dataclass(frozen=True, slots=True)
class _SelectedReviewItem:
    example: WordLevelSRLExample
    review_id: str
    stratum: str
    rank_digest: str


def discover_repository_root(start: str | os.PathLike[str]) -> Path:
    """Return the exact containing Git worktree without exposing Git output."""

    start_path = Path(start).resolve()
    completed = subprocess.run(
        ["git", "-C", str(start_path), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise ReviewPathError("repository root is not inside a Git worktree")
    try:
        root = Path(completed.stdout.decode("utf-8").strip()).resolve()
    except UnicodeDecodeError as exc:
        raise ReviewPathError("Git returned a non-UTF-8 repository path") from exc
    if not root.is_dir():
        raise ReviewPathError("Git worktree root is not a directory")
    return root


def create_manual_review_package(
    archive_path: str | os.PathLike[str],
    prepared_directory: str | os.PathLike[str],
    review_directory: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    authorization: TalkBankReviewAuthorization | None,
    per_stratum: int = 2,
) -> ManualReviewReceipt:
    """Select a deterministic stratified sample and write a private package.

    Authorization and path checks happen before the prepared dataset is read.
    The destination must be a new directory strictly below the repository's
    ignored ``data/review`` root.
    """

    confirmed = _require_authorization(authorization)
    if type(per_stratum) is not int or not 1 <= per_stratum <= 20:
        raise ValueError("per_stratum must be an integer between 1 and 20")

    root = _validate_repository_root(repository_root)
    raw_archive_path = _validate_raw_archive_path(archive_path, root)
    prepared_path = _validate_prepared_path(prepared_directory, root)
    review_path = _validate_review_path(
        review_directory,
        root,
        must_exist=False,
    )
    if review_path.exists():
        raise ReviewPathError("review output directory must not already exist")

    dataset = read_prepared_dataset(prepared_path)
    raw_build = build_babysrl_dataset(
        raw_archive_path,
        expected_sha256=BABYSRL_ARCHIVE_SHA256,
        expected_size_bytes=BABYSRL_ARCHIVE_SIZE_BYTES,
    )
    _validate_prepared_raw_identity(dataset.examples, raw_build.examples)
    selected, population_counts, sample_counts = _select_stratified_sample(
        dataset.examples,
        dataset_fingerprint=dataset.manifest.dataset_fingerprint,
        per_stratum=per_stratum,
    )
    if not selected:
        raise ReviewFormatError("prepared BabySRL dataset produced an empty sample")

    raw_rows_by_review_id = _selected_raw_rows(
        raw_archive_path,
        selected,
    )
    if _sha256_file(raw_archive_path) != BABYSRL_ARCHIVE_SHA256:
        raise ReviewIntegrityError("BabySRL archive changed during review creation")
    item_bytes = b"".join(
        _item_bytes(item, raw_rows_by_review_id[item.review_id])
        for item in selected
    )
    items_sha256 = hashlib.sha256(item_bytes).hexdigest()
    sample_fingerprint = _sample_fingerprint(
        dataset_fingerprint=dataset.manifest.dataset_fingerprint,
        archive_sha256=BABYSRL_ARCHIVE_SHA256,
        per_stratum=per_stratum,
        items_sha256=items_sha256,
    )
    decision_bytes = b"".join(_decision_template_bytes(item) for item in selected)
    manifest = {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "archive_sha256": BABYSRL_ARCHIVE_SHA256,
        "archive_size_bytes": BABYSRL_ARCHIVE_SIZE_BYTES,
        "dataset_fingerprint": dataset.manifest.dataset_fingerprint,
        "selection_policy": SELECTION_POLICY,
        "per_stratum": per_stratum,
        "sample_size": len(selected),
        "population_counts_by_stratum": dict(population_counts),
        "sample_counts_by_stratum": dict(sample_counts),
        "items_sha256": items_sha256,
        "sample_fingerprint": sample_fingerprint,
        "prepared_raw_identity": "exact_match",
        "access_confirmation": confirmed.as_payload(),
    }

    review_root = (root / "data" / "review").resolve()
    review_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=".manual-review-", dir=review_root)
    )
    temporary_path.chmod(0o700)
    try:
        _write_new_private_file(
            temporary_path / REVIEW_ITEMS_FILENAME,
            item_bytes,
        )
        _write_new_private_file(
            temporary_path / REVIEW_DECISIONS_FILENAME,
            decision_bytes,
        )
        _write_new_private_file(
            temporary_path / REVIEW_MANIFEST_FILENAME,
            _canonical_json_bytes(manifest),
        )
        review_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary_path.rename(review_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            shutil.rmtree(temporary_path)

    return ManualReviewReceipt(
        archive_sha256=BABYSRL_ARCHIVE_SHA256,
        dataset_fingerprint=dataset.manifest.dataset_fingerprint,
        sample_fingerprint=sample_fingerprint,
        sample_size=len(selected),
        sample_counts_by_stratum=MappingProxyType(dict(sample_counts)),
    )


def finalize_manual_review(
    review_directory: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    authorization: TalkBankReviewAuthorization | None,
) -> Mapping[str, object]:
    """Validate review decisions and write a corpus-free aggregate record.

    The status passes only when every sampled item is approved. Any rejection
    fails the review; an uncertain or pending item keeps the result on hold.
    """

    confirmed = _require_authorization(authorization)
    root = _validate_repository_root(repository_root)
    review_path = _validate_review_path(
        review_directory,
        root,
        must_exist=True,
    )

    manifest = _read_json_object(
        review_path / REVIEW_MANIFEST_FILENAME,
        expected_keys=_MANIFEST_KEYS,
        context="review manifest",
    )
    _validate_manifest(manifest, confirmed)

    item_path = review_path / REVIEW_ITEMS_FILENAME
    item_bytes = _read_private_file(item_path, context="review items")
    actual_item_digest = hashlib.sha256(item_bytes).hexdigest()
    if actual_item_digest != manifest["items_sha256"]:
        raise ReviewIntegrityError("review items do not match the manifest digest")
    expected_sample_fingerprint = _sample_fingerprint(
        dataset_fingerprint=manifest["dataset_fingerprint"],
        archive_sha256=manifest["archive_sha256"],
        per_stratum=manifest["per_stratum"],
        items_sha256=actual_item_digest,
    )
    if expected_sample_fingerprint != manifest["sample_fingerprint"]:
        raise ReviewIntegrityError("review sample fingerprint does not reconcile")

    review_ids, review_strata = _parse_review_items(
        item_bytes,
        expected_sample_size=manifest["sample_size"],
        expected_stratum_counts=manifest["sample_counts_by_stratum"],
    )
    decisions = _parse_decisions(
        _read_private_file(
            review_path / REVIEW_DECISIONS_FILENAME,
            context="review decisions",
        ),
        expected_review_ids=review_ids,
    )

    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    stratum_decisions: dict[str, Counter[str]] = defaultdict(Counter)
    for review_id, decision, reason_codes in decisions:
        normalized_decision = decision if decision is not None else "pending"
        decision_counts[normalized_decision] += 1
        stratum_decisions[review_strata[review_id]][normalized_decision] += 1
        reason_counts.update(reason_codes)

    if decision_counts["reject"]:
        status = "fail"
    elif decision_counts["uncertain"] or decision_counts["pending"]:
        status = "hold"
    else:
        status = "pass"

    aggregate: dict[str, object] = {
        "schema_version": REVIEW_AGGREGATE_SCHEMA_VERSION,
        "archive_sha256": manifest["archive_sha256"],
        "archive_size_bytes": manifest["archive_size_bytes"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "sample_fingerprint": manifest["sample_fingerprint"],
        "selection_policy": SELECTION_POLICY,
        "prepared_raw_identity": "exact_match",
        "sample_size": manifest["sample_size"],
        "access_confirmation": confirmed.as_payload(),
        "decision_counts": {
            value: decision_counts[value] for value in _DECISIONS
        },
        "reason_counts": {
            value: reason_counts[value] for value in _REASON_CODES
        },
        "decision_counts_by_stratum": {
            stratum: {
                value: stratum_decisions[stratum][value]
                for value in _DECISIONS
            }
            for stratum in sorted(stratum_decisions, key=_stratum_sort_key)
        },
        "approval_criterion": (
            "pass iff every sampled item is approved; fail if any item is "
            "rejected; otherwise hold"
        ),
        "status": status,
    }
    _atomic_private_write(
        review_path / REVIEW_AGGREGATE_FILENAME,
        _canonical_json_bytes(aggregate),
    )
    return MappingProxyType(aggregate)


def _require_authorization(
    authorization: TalkBankReviewAuthorization | None,
) -> TalkBankReviewAuthorization:
    if type(authorization) is not TalkBankReviewAuthorization:
        raise ReviewAuthorizationError(
            "manual review requires explicit TalkBank registration, sign-in, "
            "and current ground-rules confirmation"
        )
    return authorization


def _validate_repository_root(root: str | os.PathLike[str]) -> Path:
    requested = Path(root).resolve()
    discovered = discover_repository_root(requested)
    if discovered != requested:
        raise ReviewPathError("repository_root must be the exact Git worktree root")
    return requested


def _validate_prepared_path(
    path: str | os.PathLike[str],
    repository_root: Path,
) -> Path:
    prepared = Path(path).resolve()
    allowed_root = (repository_root / "data" / "processed").resolve()
    if prepared == allowed_root or not prepared.is_relative_to(allowed_root):
        raise ReviewPathError(
            "prepared data must be in an ignored child of data/processed"
        )
    if not prepared.is_dir():
        raise ReviewPathError("prepared dataset directory does not exist")
    _require_ignored_and_untracked(prepared, repository_root)
    return prepared


def _validate_raw_archive_path(
    path: str | os.PathLike[str],
    repository_root: Path,
) -> Path:
    archive = Path(path).resolve()
    allowed_root = (repository_root / "data" / "raw").resolve()
    if not archive.is_relative_to(allowed_root) or archive == allowed_root:
        raise ReviewPathError("BabySRL archive must be inside ignored data/raw")
    if archive.name != "BabySRL.zip" or not archive.is_file():
        raise ReviewPathError("expected the pinned data/raw/BabySRL.zip archive")
    _require_ignored_and_untracked(archive, repository_root)
    return archive


def _validate_review_path(
    path: str | os.PathLike[str],
    repository_root: Path,
    *,
    must_exist: bool,
) -> Path:
    review = Path(path).resolve()
    allowed_root = (repository_root / "data" / "review").resolve()
    if review == allowed_root or not review.is_relative_to(allowed_root):
        raise ReviewPathError(
            "review output must be a child of the ignored data/review directory"
        )
    if must_exist and not review.is_dir():
        raise ReviewPathError("review package directory does not exist")
    _require_ignored_and_untracked(review, repository_root)
    return review


def _require_ignored_and_untracked(path: Path, repository_root: Path) -> None:
    relative = path.relative_to(repository_root).as_posix()
    tracked = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "--", relative],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if tracked.returncode != 0:
        raise ReviewPathError("could not verify whether the path is tracked")
    if tracked.stdout.strip():
        raise ReviewPathError("review data path contains a Git-tracked target")

    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ignored.returncode != 0:
        raise ReviewPathError("review data path is not covered by Git ignore rules")


def _validate_prepared_raw_identity(
    prepared_examples: tuple[WordLevelSRLExample, ...],
    raw_examples: tuple[WordLevelSRLExample, ...],
) -> None:
    prepared_by_id = {example.example_id: example for example in prepared_examples}
    raw_by_id = {example.example_id: example for example in raw_examples}
    if len(prepared_by_id) != len(prepared_examples):
        raise ReviewIntegrityError("prepared examples contain duplicate identities")
    if len(raw_by_id) != len(raw_examples):
        raise ReviewIntegrityError("raw conversion contains duplicate identities")
    if set(prepared_by_id) != set(raw_by_id):
        raise ReviewIntegrityError(
            "prepared dataset does not exactly match the pinned raw conversion"
        )
    if any(prepared_by_id[key] != raw_by_id[key] for key in prepared_by_id):
        raise ReviewIntegrityError(
            "prepared dataset does not exactly match the pinned raw conversion"
        )


def _selected_raw_rows(
    archive_path: Path,
    selected: tuple[_SelectedReviewItem, ...],
) -> Mapping[str, tuple[tuple[str, str, str], ...]]:
    result: dict[str, tuple[tuple[str, str, str], ...]] = {}
    try:
        with ZipFile(archive_path) as archive:
            for item in selected:
                match = _EXAMPLE_ID_PATTERN.fullmatch(item.example.example_id)
                if match is None:
                    raise ReviewIntegrityError(
                        "prepared example ID cannot be joined to raw BabySRL"
                    )
                child = match.group("child")
                child_directory = child.capitalize()
                document = match.group("document")
                member = (
                    f"BabySRL/{child_directory}/{document}.srl.cha"
                )
                try:
                    source = archive.read(member).decode("utf-8-sig")
                except (KeyError, UnicodeDecodeError) as exc:
                    raise ReviewIntegrityError(
                        "selected example cannot be read from raw BabySRL"
                    ) from exc
                blocks = _raw_srl_blocks(source)
                utterance_index = int(match.group("utterance")) - 1
                proposition_index = int(match.group("proposition")) - 1
                if not 0 <= utterance_index < len(blocks):
                    raise ReviewIntegrityError(
                        "selected example has no matching raw utterance block"
                    )
                rows = blocks[utterance_index]
                if not rows or any(
                    len(row) <= 3 + proposition_index for row in rows
                ):
                    raise ReviewIntegrityError(
                        "selected example has no matching raw proposition column"
                    )
                selected_rows = tuple(
                    (row[1], row[2], row[3 + proposition_index])
                    for row in rows
                )
                if tuple(row[0] for row in selected_rows) != item.example.words:
                    raise ReviewIntegrityError(
                        "prepared words do not align with the raw proposition"
                    )
                predicate_marker = selected_rows[item.example.predicate_index][1]
                if (
                    predicate_marker == "-"
                    or item.example.predicate_roleset != f"{predicate_marker}.XX"
                ):
                    raise ReviewIntegrityError(
                        "prepared predicate does not align with the raw proposition"
                    )
                result[item.review_id] = selected_rows
    except BadZipFile as exc:
        raise ReviewIntegrityError("BabySRL archive is no longer a valid ZIP") from exc
    if len(result) != len(selected):
        raise ReviewIntegrityError("raw review evidence does not reconcile")
    return MappingProxyType(result)


def _raw_srl_blocks(source: str) -> tuple[tuple[tuple[str, ...], ...], ...]:
    blocks: list[tuple[tuple[str, ...], ...]] = []
    current_rows: list[tuple[str, ...]] = []

    def flush() -> None:
        if current_rows:
            blocks.append(tuple(current_rows))

    for line in source.splitlines():
        if line.startswith("%srl:"):
            current_rows.append(tuple(line.split()))
        else:
            flush()
            current_rows.clear()
    flush()
    return tuple(blocks)


def _select_stratified_sample(
    examples: Iterable[WordLevelSRLExample],
    *,
    dataset_fingerprint: str,
    per_stratum: int,
) -> tuple[
    tuple[_SelectedReviewItem, ...],
    Mapping[str, int],
    Mapping[str, int],
]:
    _require_sha256(dataset_fingerprint, context="dataset fingerprint")
    grouped: dict[str, list[_SelectedReviewItem]] = defaultdict(list)
    for example in examples:
        child = _babysrl_child(example.document_id)
        shape = _annotation_shape(example.tags)
        stratum = f"{example.split}/{child}/{shape}"
        rank_digest = hashlib.sha256(
            (
                f"{SELECTION_POLICY}\0{dataset_fingerprint}\0{stratum}\0"
                f"{example.example_id}"
            ).encode("utf-8")
        ).hexdigest()
        review_id = hashlib.sha256(
            (
                f"babysrl-manual-review-id/v1\0{dataset_fingerprint}\0"
                f"{example.example_id}"
            ).encode("utf-8")
        ).hexdigest()
        grouped[stratum].append(
            _SelectedReviewItem(
                example=example,
                review_id=review_id,
                stratum=stratum,
                rank_digest=rank_digest,
            )
        )

    population_counts = {
        stratum: len(items)
        for stratum, items in sorted(
            grouped.items(), key=lambda item: _stratum_sort_key(item[0])
        )
    }
    selected: list[_SelectedReviewItem] = []
    sample_counts: dict[str, int] = {}
    for stratum in population_counts:
        ordered = sorted(
            grouped[stratum],
            key=lambda item: (item.rank_digest, item.example.example_id),
        )
        chosen = ordered[:per_stratum]
        selected.extend(chosen)
        sample_counts[stratum] = len(chosen)

    selected.sort(
        key=lambda item: (
            _stratum_sort_key(item.stratum),
            item.rank_digest,
            item.example.example_id,
        )
    )
    review_ids = [item.review_id for item in selected]
    if len(review_ids) != len(set(review_ids)):
        raise ReviewIntegrityError("manual-review ID collision")
    return (
        tuple(selected),
        MappingProxyType(population_counts),
        MappingProxyType(sample_counts),
    )


def _babysrl_child(document_id: str) -> str:
    match = _CHILD_DOCUMENT_PATTERN.fullmatch(document_id)
    if match is None:
        raise ReviewFormatError(
            "prepared dataset contains a non-BabySRL document identity"
        )
    return match.group(1)


def _annotation_shape(tags: tuple[str, ...]) -> str:
    if any(tag in {"B-C-V", "I-C-V"} for tag in tags):
        return "continued_relation"
    argument_starts = Counter(
        tag[2:]
        for tag in tags
        if tag.startswith("B-") and tag not in {"B-V", "B-C-V"}
    )
    if any(count > 1 for count in argument_starts.values()):
        return "discontinuous_argument"
    if any(tag.startswith("I-") for tag in tags):
        return "multiword_span"
    return "single_token_spans"


def _stratum_sort_key(stratum: str) -> tuple[int, int, int, str]:
    pieces = stratum.split("/")
    if len(pieces) != 3:
        raise ReviewFormatError("invalid review stratum")
    split, child, shape = pieces
    if split not in _SPLIT_ORDER or child not in _CHILD_ORDER:
        raise ReviewFormatError("invalid review stratum")
    if shape not in _SHAPE_ORDER:
        raise ReviewFormatError("invalid review stratum")
    return (
        _SPLIT_ORDER[split],
        _CHILD_ORDER[child],
        _SHAPE_ORDER[shape],
        stratum,
    )


def _item_bytes(
    item: _SelectedReviewItem,
    raw_rows: tuple[tuple[str, str, str], ...],
) -> bytes:
    example = item.example
    if len(raw_rows) != len(example.tags):
        raise ReviewIntegrityError("raw and prepared review rows do not align")
    return _canonical_json_bytes(
        {
            "schema_version": REVIEW_ITEM_SCHEMA_VERSION,
            "review_id": item.review_id,
            "stratum": item.stratum,
            "predicate_index": example.predicate_index,
            "predicate_roleset": example.predicate_roleset,
            "rows": [
                {
                    "word": word,
                    "predicate_marker": predicate_marker,
                    "raw_role_cell": raw_role_cell,
                    "prepared_tag": prepared_tag,
                }
                for (word, predicate_marker, raw_role_cell), prepared_tag in zip(
                    raw_rows,
                    example.tags,
                    strict=True,
                )
            ],
        }
    )


def _decision_template_bytes(item: _SelectedReviewItem) -> bytes:
    return _canonical_json_bytes(
        {
            "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
            "review_id": item.review_id,
            "decision": None,
            "reason_codes": [],
        }
    )


def _sample_fingerprint(
    *,
    dataset_fingerprint: str,
    archive_sha256: str,
    per_stratum: int,
    items_sha256: str,
) -> str:
    _require_sha256(dataset_fingerprint, context="dataset fingerprint")
    _require_sha256(archive_sha256, context="archive digest")
    _require_sha256(items_sha256, context="review items digest")
    payload = (
        f"{SELECTION_POLICY}\0{archive_sha256}\0{dataset_fingerprint}\0"
        f"{per_stratum}\0{items_sha256}"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest(
    manifest: Mapping[str, Any],
    authorization: TalkBankReviewAuthorization,
) -> None:
    if manifest["schema_version"] != REVIEW_MANIFEST_SCHEMA_VERSION:
        raise ReviewFormatError("unsupported review manifest schema")
    if manifest["selection_policy"] != SELECTION_POLICY:
        raise ReviewFormatError("unsupported review selection policy")
    if manifest["archive_sha256"] != BABYSRL_ARCHIVE_SHA256:
        raise ReviewIntegrityError("review manifest has the wrong BabySRL archive")
    if manifest["archive_size_bytes"] != BABYSRL_ARCHIVE_SIZE_BYTES:
        raise ReviewIntegrityError("review manifest has the wrong archive size")
    if manifest["prepared_raw_identity"] != "exact_match":
        raise ReviewIntegrityError("prepared/raw identity was not verified")
    _require_sha256(manifest["archive_sha256"], context="archive digest")
    _require_sha256(manifest["dataset_fingerprint"], context="dataset fingerprint")
    _require_sha256(manifest["items_sha256"], context="review items digest")
    _require_sha256(manifest["sample_fingerprint"], context="sample fingerprint")
    if (
        type(manifest["per_stratum"]) is not int
        or not 1 <= manifest["per_stratum"] <= 20
    ):
        raise ReviewFormatError("invalid per_stratum in review manifest")
    if type(manifest["sample_size"]) is not int or manifest["sample_size"] < 1:
        raise ReviewFormatError("invalid sample_size in review manifest")
    population_counts = _validated_count_mapping(
        manifest["population_counts_by_stratum"],
        context="population counts",
        allow_zero=False,
    )
    sample_counts = _validated_count_mapping(
        manifest["sample_counts_by_stratum"],
        context="sample counts",
        allow_zero=False,
    )
    if set(population_counts) != set(sample_counts):
        raise ReviewFormatError("population and sample strata do not match")
    if any(sample_counts[key] > population_counts[key] for key in sample_counts):
        raise ReviewFormatError("sample count exceeds its stratum population")
    if sum(sample_counts.values()) != manifest["sample_size"]:
        raise ReviewFormatError("sample counts do not reconcile to sample_size")
    if manifest["access_confirmation"] != authorization.as_payload():
        raise ReviewAuthorizationError(
            "authorization does not match the review package acknowledgement"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_count_mapping(
    value: Any,
    *,
    context: str,
    allow_zero: bool,
) -> dict[str, int]:
    if type(value) is not dict:
        raise ReviewFormatError(f"{context} must be an object")
    result: dict[str, int] = {}
    for stratum, count in value.items():
        if not isinstance(stratum, str):
            raise ReviewFormatError(f"{context} contains a non-string stratum")
        _stratum_sort_key(stratum)
        minimum = 0 if allow_zero else 1
        if type(count) is not int or count < minimum:
            raise ReviewFormatError(f"{context} contains an invalid count")
        result[stratum] = count
    return result


def _parse_review_items(
    contents: bytes,
    *,
    expected_sample_size: int,
    expected_stratum_counts: Any,
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    rows = _parse_jsonl(contents, context="review items")
    if len(rows) != expected_sample_size:
        raise ReviewFormatError("review item count does not match sample_size")
    expected_counts = _validated_count_mapping(
        expected_stratum_counts,
        context="sample counts",
        allow_zero=False,
    )
    review_ids: list[str] = []
    review_strata: dict[str, str] = {}
    actual_counts: Counter[str] = Counter()
    for row in rows:
        _require_exact_keys(row, _ITEM_KEYS, context="review item")
        if row["schema_version"] != REVIEW_ITEM_SCHEMA_VERSION:
            raise ReviewFormatError("unsupported review item schema")
        review_id = row["review_id"]
        _require_sha256(review_id, context="review ID")
        if review_id in review_strata:
            raise ReviewFormatError("duplicate review ID")
        stratum = row["stratum"]
        if not isinstance(stratum, str):
            raise ReviewFormatError("review item stratum must be a string")
        _stratum_sort_key(stratum)
        predicate_index = row["predicate_index"]
        raw_rows = row["rows"]
        if type(raw_rows) is not list or not raw_rows:
            raise ReviewFormatError("review item rows must be a nonempty list")
        if type(predicate_index) is not int or not 0 <= predicate_index < len(raw_rows):
            raise ReviewFormatError("review item predicate index is invalid")
        if not isinstance(row["predicate_roleset"], str):
            raise ReviewFormatError("review item roleset must be a string")
        for raw_row in raw_rows:
            if type(raw_row) is not dict:
                raise ReviewFormatError("review item contains an invalid row")
            _require_exact_keys(
                raw_row,
                _ITEM_ROW_KEYS,
                context="review item row",
            )
            if any(not isinstance(raw_row[key], str) for key in _ITEM_ROW_KEYS):
                raise ReviewFormatError("review item row fields must be strings")
        review_ids.append(review_id)
        review_strata[review_id] = stratum
        actual_counts[stratum] += 1
    if dict(actual_counts) != expected_counts:
        raise ReviewFormatError("review item strata do not match the manifest")
    return tuple(review_ids), MappingProxyType(review_strata)


def _parse_decisions(
    contents: bytes,
    *,
    expected_review_ids: tuple[str, ...],
) -> tuple[tuple[str, str | None, tuple[str, ...]], ...]:
    rows = _parse_jsonl(contents, context="review decisions")
    if len(rows) != len(expected_review_ids):
        raise ReviewFormatError("decision count does not match review sample")
    parsed: list[tuple[str, str | None, tuple[str, ...]]] = []
    for index, row in enumerate(rows):
        _require_exact_keys(row, _DECISION_KEYS, context="review decision")
        if row["schema_version"] != REVIEW_DECISION_SCHEMA_VERSION:
            raise ReviewFormatError("unsupported review decision schema")
        review_id = row["review_id"]
        if review_id != expected_review_ids[index]:
            raise ReviewFormatError("review decisions are missing or out of order")
        decision = row["decision"]
        if decision not in {None, "approve", "reject", "uncertain"}:
            raise ReviewFormatError("review decision has an unsupported value")
        raw_reasons = row["reason_codes"]
        if type(raw_reasons) is not list or any(
            not isinstance(value, str) for value in raw_reasons
        ):
            raise ReviewFormatError("reason_codes must be a string list")
        reasons = tuple(raw_reasons)
        if len(reasons) != len(set(reasons)):
            raise ReviewFormatError("reason_codes cannot contain duplicates")
        if any(reason not in _REASON_CODES for reason in reasons):
            raise ReviewFormatError("reason_codes contains an unsupported code")
        if decision in {None, "approve"} and reasons:
            raise ReviewFormatError(
                "pending and approved decisions cannot contain reason codes"
            )
        if decision in {"reject", "uncertain"} and not reasons:
            raise ReviewFormatError(
                "rejected and uncertain decisions require a reason code"
            )
        parsed.append((review_id, decision, reasons))
    return tuple(parsed)


def _read_json_object(
    path: Path,
    *,
    expected_keys: frozenset[str],
    context: str,
) -> dict[str, Any]:
    contents = _read_private_file(path, context=context)
    if b"\r" in contents or not contents.endswith(b"\n"):
        raise ReviewFormatError(f"{context} must use a final LF newline")
    if contents.count(b"\n") != 1:
        raise ReviewFormatError(f"{context} must contain one JSON object")
    row = _decode_json_object(contents[:-1], context=context)
    _require_exact_keys(row, expected_keys, context=context)
    return row


def _parse_jsonl(contents: bytes, *, context: str) -> tuple[dict[str, Any], ...]:
    if b"\r" in contents:
        raise ReviewFormatError(f"{context} must use LF newlines")
    if contents and not contents.endswith(b"\n"):
        raise ReviewFormatError(f"{context} must end with an LF newline")
    if not contents:
        return ()
    return tuple(
        _decode_json_object(line, context=f"{context} row {line_number}")
        for line_number, line in enumerate(contents[:-1].split(b"\n"), start=1)
    )


def _decode_json_object(contents: bytes, *, context: str) -> dict[str, Any]:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewFormatError(f"{context} is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ReviewFormatError) as exc:
        raise ReviewFormatError(f"{context} is not valid strict JSON") from exc
    if type(value) is not dict:
        raise ReviewFormatError(f"{context} must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewFormatError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ReviewFormatError("non-finite JSON number is not allowed")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        raise ReviewFormatError(f"{context} has unexpected or missing fields")


def _require_sha256(value: Any, *, context: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ReviewFormatError(f"{context} must be a lowercase SHA-256 digest")


def _read_private_file(path: Path, *, context: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ReviewFormatError(f"could not read {context}") from exc


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_new_private_file(path: Path, contents: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_private_write(path: Path, contents: bytes) -> None:
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
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _authorization_from_arguments(
    arguments: argparse.Namespace,
) -> TalkBankReviewAuthorization:
    return TalkBankReviewAuthorization(
        registration_confirmed=arguments.confirm_talkbank_registration,
        ground_rules_confirmed=arguments.confirm_talkbank_ground_rules,
        accepted_on=arguments.rules_accepted_on,
    )


def _add_authorization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-talkbank-registration",
        action="store_true",
        help="confirm completed TalkBank registration and sign-in",
    )
    parser.add_argument(
        "--confirm-talkbank-ground-rules",
        action="store_true",
        help="confirm acceptance of the current TalkBank ground rules",
    )
    parser.add_argument(
        "--rules-accepted-on",
        help="ground-rules acceptance date in YYYY-MM-DD form",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Create or finalize a private review package without printing corpus text."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        help="exact repository root; defaults to the current Git worktree",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("archive_path", type=Path)
    create_parser.add_argument("prepared_directory", type=Path)
    create_parser.add_argument("review_directory", type=Path)
    create_parser.add_argument("--per-stratum", type=int, default=2)
    _add_authorization_arguments(create_parser)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("review_directory", type=Path)
    _add_authorization_arguments(finalize_parser)

    arguments = parser.parse_args(argv)
    try:
        authorization = _authorization_from_arguments(arguments)
        repository_root = (
            arguments.repository_root.resolve()
            if arguments.repository_root is not None
            else discover_repository_root(Path.cwd())
        )
        if arguments.command == "create":
            receipt = create_manual_review_package(
                arguments.archive_path,
                arguments.prepared_directory,
                arguments.review_directory,
                repository_root=repository_root,
                authorization=authorization,
                per_stratum=arguments.per_stratum,
            )
            safe_payload: Mapping[str, object] = receipt.as_payload()
        else:
            safe_payload = finalize_manual_review(
                arguments.review_directory,
                repository_root=repository_root,
                authorization=authorization,
            )
    except (ManualReviewError, DatasetFormatError, OSError, ValueError) as exc:
        parser.error(str(exc))

    print(_canonical_json_bytes(safe_payload).decode("utf-8"), end="")
    return 0 if safe_payload.get("status") not in {"fail", "hold"} else 2


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
