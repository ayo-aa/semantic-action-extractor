"""Read-only feasibility audit for the pinned MASC PropBank archive.

The audit reads the ZIP in place and emits aggregate metadata only.  It does
not extract corpus files, write prepared examples, or reproduce source text.
The source-family policy is deliberately narrower than ANC's corpus-wide
license statement so the research gate fails closed when lineage is unclear.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

from .propbank import (
    PropBankAdapterError,
    convert_propbank_record,
    parse_penn_trees,
    parse_propbank_record,
    resolve_pointer,
)


AUDIT_SCHEMA_VERSION = 1
PINNED_ARCHIVE_FILENAME = "Propbank-original-format.zip"
PINNED_ARCHIVE_SHA256 = (
    "b7e89cfbb7a0b7caf3ba5076ac95ee80810834d3522678eefd488f166ddcc4df"
)
PINNED_ARCHIVE_BYTES = 11_402_565
CONVERSION_THRESHOLD = 0.99

_TEXT_PREFIX = "Propbank/MASC1_textfiles/"
_PTB_PREFIX = "Propbank/Penn_Treebank-orig/data/"
_PROP_PREFIX = "Propbank/Propbank-orig/data/"
_README_PATH = "Propbank/README.txt"

_TEXT_ALIASES = {
    "wsj_1640.mrg-NEW": "wsj_1640",
}
_PTB_ALIASES = {
    "sw2014-UTF16-ms98-a-trans": "sw2014-ms98-a-trans",
    "sw2025-ms98-a-trans.ascii-1-NEW": "sw2025-ms98-a-trans",
    "sw2071-UTF16-ms98-a-trans": "sw2071-ms98-a-trans",
    "sw2078-UTF16-ms98-a-trans": "sw2078-ms98-a-trans",
    "wsj_1640.mrg-NEW": "wsj_1640",
}
_PROP_ALIASES = {
    "ENRON-pearson-email-25jul02_LU_ANNOTATE": (
        "ENRON-pearson-email-25jul02"
    ),
    "enron-thread-159550_LU_ANNOTATE": "enron-thread-159550",
    "wsj_1640.prop-NEW": "wsj_1640",
}
_LABELED_POINTER_RE = re.compile(
    r"^[0-9]+:[0-9]+(?:[,;*][0-9]+:[0-9]+)*-(?P<label>.+)$"
)
_TYPED_LEMMA_RE = re.compile(r"^.+-(?P<kind>[anv])$")
_PRETERMINAL_RE = re.compile(r"\([^()\s]+\s+[^()\s]+\)")
_FRAME_LEGAL_NOTICE_RE = re.compile(r"^(?:copying|license|notice)-[anv]$")
_ARCHIVE_LEGAL_NOTICE_STEMS = frozenset({"copying", "license", "notice"})


_ICIC_DOCUMENTS = frozenset(
    {
        "110CYL067",
        "110CYL068",
        "110CYL069",
        "110CYL070",
        "110CYL071",
        "110CYL072",
        "110CYL200",
        "112C-L012",
        "112C-L013",
        "112C-L014",
        "112C-L015",
        "112C-L016",
        "113CWL017",
        "113CWL018",
        "114CUL057",
        "114CUL058",
        "114CUL059",
        "114CUL060",
        "115CVL035",
        "115CVL036",
        "115CVL037",
        "116CUL032",
        "116CUL033",
        "116CUL034",
        "117CWL008",
        "117CWL009",
        "118CWL048",
        "118CWL049",
        "118CWL050",
        "119CWL041",
        "602CZL285",
    }
)
_SLATE_DOCUMENTS = frozenset(
    {
        "Article247_327",
        "Article247_328",
        "Article247_3500",
        "Article247_400",
        "Article247_500",
        "Article247_66",
    }
)
_BERLITZ_DOCUMENTS = frozenset({"HistoryGreek", "HistoryJerusalem"})
_OUP_DOCUMENTS = frozenset({"ch5", "chZ"})
_SWITCHBOARD_DOCUMENTS = frozenset(
    {
        "sw2014-ms98-a-trans",
        "sw2015-ms98-a-trans",
        "sw2025-ms98-a-trans",
        "sw2071-ms98-a-trans",
        "sw2078-ms98-a-trans",
    }
)
_VERBATIM_DOCUMENTS = frozenset({"VOL15_3"})
_PLOS_DOCUMENTS = frozenset({"pmed.0010029"})

PINNED_ALLOWED_SOURCE_FAMILIES = {
    "Berlitz": _BERLITZ_DOCUMENTS,
    "ICIC": _ICIC_DOCUMENTS,
    "OUP": _OUP_DOCUMENTS,
    "PLOS": _PLOS_DOCUMENTS,
    "Slate": _SLATE_DOCUMENTS,
    "Switchboard": _SWITCHBOARD_DOCUMENTS,
    "Verbatim": _VERBATIM_DOCUMENTS,
}
# "Allowed" here means admitted to the read-only diagnostic audit.  It does
# not mean cleared for training, redistribution, or checkpoint publication.
PINNED_ALLOWED_DOCUMENTS = frozenset(
    document
    for documents in PINNED_ALLOWED_SOURCE_FAMILIES.values()
    for document in documents
)

_HELD_SOURCE_FAMILIES = {
    "9/11 Report": frozenset({"chapter-10"}),
    "Charlotte": frozenset(
        {"NapierDianne", "PolkMaria", "ReidSandra", "RindnerBonnie"}
    ),
    "Enron": frozenset(
        {"ENRON-pearson-email-25jul02", "enron-thread-159550"}
    ),
    "Government transcript": frozenset({"Day3PMSession"}),
    "LU Corpus": frozenset(
        {
            "20000410_nyt-NEW",
            "20000415_apw_eng-NEW",
            "20000419_apw_eng-NEW",
            "20000424_nyt-NEW",
            "A1.E1-NEW",
            "A1.E2-NEW",
        }
    ),
    "Web fiction": frozenset({"lw1"}),
}


def canonical_document_id(layer: str, archive_path: str) -> str:
    """Return the exact canonical ID for one recognized layer path."""

    if layer not in {"text", "ptb", "prop"}:
        raise ValueError("layer must be 'text', 'ptb', or 'prop'")
    name = PurePosixPath(archive_path).name
    if layer == "text":
        if not name.endswith(".txt"):
            raise ValueError("text path must end in .txt")
        name = name[:-4]
        return _TEXT_ALIASES.get(name, name)
    if layer == "ptb":
        if name.endswith(".mrg.txt"):
            name = name[:-8]
        elif name.endswith(".mrg"):
            name = name[:-4]
        else:
            raise ValueError("PTB path must end in .mrg or .mrg.txt")
        return _PTB_ALIASES.get(name, name)

    if not name.endswith(".prop"):
        raise ValueError("PropBank path must end in .prop")
    name = name[:-5]
    return _PROP_ALIASES.get(name, name)


def source_family(document_id: str) -> str:
    """Return the reviewed source family or a fail-closed fallback."""

    if document_id.casefold().startswith("wsj_"):
        return "Wall Street Journal"
    for family, documents in PINNED_ALLOWED_SOURCE_FAMILIES.items():
        if document_id in documents:
            return family
    for family, documents in _HELD_SOURCE_FAMILIES.items():
        if document_id in documents:
            return family
    if document_id == "sw2017-ms98-a-trans":
        return "Switchboard filename mismatch"
    if document_id == "20000815_AFP_ARB.0084.IBM-HA-NEW-en":
        return "Unmapped PTB extra"
    return "Unmapped"


def source_disposition(document_id: str) -> str:
    """Return ``diagnostic``, ``hold``, or ``deny`` for one document ID."""

    if document_id.casefold().startswith("wsj_"):
        return "deny"
    if document_id in PINNED_ALLOWED_DOCUMENTS:
        return "diagnostic"
    return "hold"


def _archive_path_is_unsafe(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or re.match(r"^[A-Za-z]:", name) is not None
    )


def _is_symlink(info: ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == stat.S_IFLNK


def _decode_text(payload: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return payload.decode(encoding), encoding
        except UnicodeError:
            continue
    raise UnicodeError("archive member is not decodable with an approved encoding")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_tree_terminal_counts(source: str) -> tuple[int, ...]:
    """Count sentences and PTB terminals without interpreting tree labels."""

    depth = 0
    start: int | None = None
    terminal_counts: list[int] = []
    for offset, character in enumerate(source):
        if character == "(":
            if depth == 0:
                start = offset
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unmatched_closing_parenthesis")
            if depth == 0:
                assert start is not None
                tree_source = source[start : offset + 1]
                terminal_counts.append(len(_PRETERMINAL_RE.findall(tree_source)))
                start = None
    if depth != 0:
        raise ValueError("unclosed_parenthesis")
    if not terminal_counts:
        raise ValueError("no_balanced_trees")
    if any(count == 0 for count in terminal_counts):
        raise ValueError("tree_without_terminal")
    return tuple(terminal_counts)


def _layer_for_path(name: str) -> str | None:
    if name.startswith(_TEXT_PREFIX) and name.endswith(".txt"):
        if "/.svn/" not in name:
            return "text"
    if name.startswith(_PTB_PREFIX):
        if name.endswith(".mrg") or name.endswith(".mrg.txt"):
            return "ptb"
    if name.startswith(_PROP_PREFIX) and name.endswith(".prop"):
        return "prop"
    return None


def _layer_genre(layer: str, archive_path: str) -> str | None:
    """Return a recognized archive genre without inferring from a basename."""

    prefixes = {
        "text": _TEXT_PREFIX,
        "ptb": _PTB_PREFIX,
        "prop": _PROP_PREFIX,
    }
    prefix = prefixes[layer]
    relative = PurePosixPath(archive_path[len(prefix) :])
    if len(relative.parts) != 2 or relative.parts[0] not in {
        "spoken",
        "written",
    }:
        return None
    return relative.parts[0]


def _select_prop_path(
    archive: ZipFile,
    paths: Sequence[str],
) -> tuple[str | None, int, bool]:
    if len(paths) == 1:
        return paths[0], 0, False
    payloads = [archive.read(path) for path in paths]
    if any(payload != payloads[0] for payload in payloads[1:]):
        return None, 0, True
    ordered = sorted(paths, key=lambda path: ("_LU_ANNOTATE" in path, path))
    duplicate_rows = sum(
        1 for line in payloads[0].decode("utf-8-sig").splitlines() if line.strip()
    ) * (len(paths) - 1)
    return ordered[0], duplicate_rows, False


def _local_record_kind(line: str) -> str:
    fields = line.split()
    if len(fields) >= 6 and fields[5] == "-----":
        match = _TYPED_LEMMA_RE.fullmatch(fields[3])
        if match is not None:
            return match.group("kind")
    return "classic"


def _inventory_record(line: str, inventory: Counter[str]) -> None:
    fields = line.split()
    modern = len(fields) >= 6 and fields[5] == "-----"
    inventory["dialect_modern" if modern else "dialect_classic"] += 1
    if modern:
        kind = _local_record_kind(line)
        inventory[f"predicate_type_{kind}"] += 1
        label_fields = fields[6:]
    else:
        label_fields = fields[5:]

    record_operators: set[str] = set()
    for field in label_fields:
        match = _LABELED_POINTER_RE.fullmatch(field)
        if match is None:
            inventory["unparsed_label_fields"] += 1
            continue
        label = match.group("label")
        if label.startswith("LINK-"):
            inventory[f"link_{label}"] += 1
        elif label.startswith("C-"):
            inventory["continuation_labels"] += 1
        elif label.startswith("R-"):
            inventory["reference_labels"] += 1
        for operator, name in ((",", "comma"), (";", "semicolon"), ("*", "star")):
            pointer = field[: match.start("label") - 1]
            if operator in pointer:
                inventory[f"pointer_fields_{name}"] += 1
                record_operators.add(name)
        if label == "rel" and any(
            operator in field[: match.start("label") - 1]
            for operator in ",;*"
        ):
            inventory["multi_pointer_predicates"] += 1
    combination = "+".join(sorted(record_operators)) or "simple"
    inventory[f"records_{combination}"] += 1


def _source_policy(allowed_documents: frozenset[str]):
    def policy(record: Any) -> str | None:
        if record.document_id.casefold().startswith("wsj_"):
            return "excluded_source_wsj"
        if record.document_id not in allowed_documents:
            return "excluded_source_not_pinned"
        return None

    return policy


def _validate_local_record(
    line: str,
) -> tuple[str, int, int, tuple[str, ...]]:
    """Validate archive-local fixed fields and return join metadata.

    MASC rows omit the document field used by the public PropBank parser.  This
    check validates only the local fixed fields and labeled-pointer syntax; it
    deliberately leaves pointer interpretation to G3.
    """

    fields = line.split()
    try:
        sentence_index = int(fields[0])
        predicate_index = int(fields[1])
    except (IndexError, ValueError) as error:
        raise ValueError("record_fixed_fields_invalid") from error
    if sentence_index < 0 or predicate_index < 0:
        raise ValueError("record_index_negative")

    modern = (
        len(fields) >= 7
        and _TYPED_LEMMA_RE.fullmatch(fields[3]) is not None
        and "." in fields[4]
        and fields[5] == "-----"
    )
    classic = (
        len(fields) >= 6
        and "." in fields[3]
        and len(fields[4]) == 5
    )
    if not modern and not classic:
        raise ValueError("record_fixed_fields_invalid")
    dialect = "modern" if modern else "classic"
    label_fields = tuple(fields[6:] if modern else fields[5:])
    if not label_fields or any(
        _LABELED_POINTER_RE.fullmatch(field) is None
        for field in label_fields
    ):
        raise ValueError("record_label_field_invalid")
    rel_count = sum(
        _LABELED_POINTER_RE.fullmatch(field).group("label") == "rel"
        for field in label_fields
    )
    if rel_count != 1:
        raise ValueError("record_rel_count_invalid")
    return dialect, sentence_index, predicate_index, label_fields


def audit_masc_archive(
    archive_path: str | Path,
    *,
    allowed_document_ids: Collection[str] = PINNED_ALLOWED_DOCUMENTS,
    allow_unpinned_archive: bool = False,
) -> dict[str, Any]:
    """Audit an archive without extracting it or returning corpus text.

    The public path requires the exact pinned artifact before any member is
    decompressed. ``allow_unpinned_archive`` exists only for invented fixtures
    that exercise the audit's failure behavior.
    """

    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if type(allow_unpinned_archive) is not bool:
        raise TypeError("allow_unpinned_archive must be a bool")
    requested_allowed_documents = frozenset(allowed_document_ids)
    denied_override_documents = frozenset(
        document
        for document in requested_allowed_documents
        if document.casefold().startswith("wsj_")
    )
    allowed_documents = requested_allowed_documents - denied_override_documents
    archive_bytes = path.stat().st_size
    archive_sha256 = _sha256(path)
    acquisition_pass = (
        path.name == PINNED_ARCHIVE_FILENAME
        and archive_bytes == PINNED_ARCHIVE_BYTES
        and archive_sha256 == PINNED_ARCHIVE_SHA256
    )
    if not acquisition_pass and not allow_unpinned_archive:
        raise ValueError(
            "archive does not match the pinned filename, byte size, and SHA-256"
        )

    try:
        archive = ZipFile(path)
    except BadZipFile as error:
        raise ValueError("archive is not a valid ZIP file") from error

    with archive:
        infos = archive.infolist()
        files = [info for info in infos if not info.is_dir()]
        clean_files = [
            info for info in files if "/.svn/" not in info.filename
        ]
        unsafe_paths = [
            info.filename
            for info in files
            if _archive_path_is_unsafe(info.filename)
        ]
        symlinks = [info.filename for info in files if _is_symlink(info)]
        encrypted = [
            info.filename for info in files if bool(info.flag_bits & 0x1)
        ]
        bad_member = None if encrypted else archive.testzip()
        duplicate_names = len(infos) - len({info.filename for info in infos})
        casefold_name_collisions = len(infos) - len(
            {info.filename.casefold() for info in infos}
        )
        uncompressed_bytes = sum(info.file_size for info in files)
        compressed_member_bytes = sum(info.compress_size for info in files)
        suffix_counts = Counter(
            PurePosixPath(info.filename).suffix.casefold() or "<none>"
            for info in clean_files
        )
        archive_level_legal_notice_files = sorted(
            info.filename
            for info in clean_files
            if PurePosixPath(info.filename).stem.casefold()
            in _ARCHIVE_LEGAL_NOTICE_STEMS
            and not info.filename.startswith(
                "Propbank/Propbank-orig/framefiles/"
            )
        )
        frame_legal_notice_files = sorted(
            info.filename
            for info in clean_files
            if info.filename.startswith("Propbank/Propbank-orig/framefiles/")
            and _FRAME_LEGAL_NOTICE_RE.fullmatch(
                PurePosixPath(info.filename).stem.casefold()
            )
            is not None
        )
        archive_safe = not any(
            (
                bad_member,
                unsafe_paths,
                symlinks,
                encrypted,
                duplicate_names,
                casefold_name_collisions,
            )
        )

        layer_maps: dict[str, dict[str, list[str]]] = {
            "text": {},
            "ptb": {},
            "prop": {},
        }
        layer_genres: dict[str, dict[str, set[str]]] = {
            "text": {},
            "ptb": {},
            "prop": {},
        }
        invalid_genre_paths: list[str] = []
        for info in clean_files:
            layer = _layer_for_path(info.filename)
            if layer is None:
                continue
            genre = _layer_genre(layer, info.filename)
            if genre is None:
                invalid_genre_paths.append(info.filename)
                continue
            document_id = canonical_document_id(layer, info.filename)
            layer_maps[layer].setdefault(document_id, []).append(info.filename)
            layer_genres[layer].setdefault(document_id, set()).add(genre)

        layer_documents = {
            layer: set(documents) for layer, documents in layer_maps.items()
        }
        document_union = set().union(*layer_documents.values())
        joined_documents = set.intersection(*layer_documents.values())
        missing_by_layer = {
            layer: sorted(document_union - documents)
            for layer, documents in layer_documents.items()
        }
        wsj_union = {
            document
            for document in document_union
            if document.casefold().startswith("wsj_")
        }
        genre_conflicts: dict[str, dict[str, list[str]]] = {}
        canonical_genre_counts: Counter[str] = Counter()
        for document in sorted(document_union):
            by_layer = {
                layer: sorted(layer_genres[layer].get(document, set()))
                for layer in ("text", "ptb", "prop")
            }
            observed = set().union(
                *(set(genres) for genres in by_layer.values())
            )
            layer_has_multiple_genres = any(
                len(genres) > 1 for genres in by_layer.values()
            )
            if len(observed) > 1 or layer_has_multiple_genres:
                genre_conflicts[document] = by_layer
            elif len(observed) == 1:
                canonical_genre_counts[next(iter(observed))] += 1
        layer_summary = {
            layer: {
                "files": sum(len(paths) for paths in layer_maps[layer].values()),
                "canonical_documents": len(layer_maps[layer]),
                "spoken_files": sum(
                    "/spoken/" in path
                    for paths in layer_maps[layer].values()
                    for path in paths
                ),
                "written_files": sum(
                    "/written/" in path
                    for paths in layer_maps[layer].values()
                    for path in paths
                ),
                "wsj_files": sum(
                    len(paths)
                    for document, paths in layer_maps[layer].items()
                    if document.casefold().startswith("wsj_")
                ),
            }
            for layer in ("text", "ptb", "prop")
        }

        duplicate_canonical_paths = {
            layer: {
                document: sorted(PurePosixPath(item).name for item in paths)
                for document, paths in layer_maps[layer].items()
                if len(paths) > 1
            }
            for layer in ("text", "ptb", "prop")
        }
        duplicate_prop_rows = 0
        ambiguous_prop_documents: list[str] = []
        primary_prop_paths: dict[str, str] = {}
        primary_prop_rows: dict[str, tuple[str, ...]] = {}
        physical_prop_rows = 0
        physical_prop_dialects: Counter[str] = Counter()
        for paths in layer_maps["prop"].values():
            for prop_path in paths:
                prop_source, _encoding = _decode_text(archive.read(prop_path))
                for row in prop_source.splitlines():
                    if not row.strip():
                        continue
                    physical_prop_rows += 1
                    dialect = (
                        "classic"
                        if _local_record_kind(row.strip()) == "classic"
                        else "modern"
                    )
                    physical_prop_dialects[dialect] += 1
        for document, paths in layer_maps["prop"].items():
            selected, duplicated_rows, ambiguous = _select_prop_path(
                archive, paths
            )
            duplicate_prop_rows += duplicated_rows
            if ambiguous:
                ambiguous_prop_documents.append(document)
            elif selected is not None:
                primary_prop_paths[document] = selected
                prop_source, _encoding = _decode_text(archive.read(selected))
                primary_prop_rows[document] = tuple(
                    line.strip()
                    for line in prop_source.splitlines()
                    if line.strip()
                )

        ptb_terminal_counts: dict[str, tuple[int, ...]] = {}
        ptb_structure_failures: dict[str, str] = {}
        for document, paths in layer_maps["ptb"].items():
            if len(paths) != 1:
                ptb_structure_failures[document] = "ambiguous_ptb_path"
                continue
            try:
                ptb_source, _encoding = _decode_text(archive.read(paths[0]))
                ptb_terminal_counts[document] = _balanced_tree_terminal_counts(
                    ptb_source
                )
            except (UnicodeError, ValueError) as error:
                ptb_structure_failures[document] = str(error)

        full_join_failures: Counter[str] = Counter()
        full_joined_rows = 0
        full_non_wsj_rows = 0
        full_non_wsj_joined_rows = 0
        canonical_prop_dialects: Counter[str] = Counter()
        invalid_rel_rows = 0
        exact_duplicate_predicates = 0
        conflicting_duplicate_predicates = 0
        conflicting_duplicate_documents: set[str] = set()
        for document, rows in primary_prop_rows.items():
            seen_predicates: dict[tuple[int, int], str] = {}
            is_wsj = document.casefold().startswith("wsj_")
            for row in rows:
                dialect = (
                    "classic" if _local_record_kind(row) == "classic" else "modern"
                )
                canonical_prop_dialects[dialect] += 1
                if not is_wsj:
                    full_non_wsj_rows += 1

                fields = row.split()
                label_start = 5 if dialect == "classic" else 6
                rel_count = 0
                for field in fields[label_start:]:
                    match = _LABELED_POINTER_RE.fullmatch(field)
                    if match is not None and match.group("label") == "rel":
                        rel_count += 1
                if rel_count != 1:
                    invalid_rel_rows += 1

                try:
                    sentence_index = int(fields[0])
                    predicate_index = int(fields[1])
                except (IndexError, ValueError):
                    full_join_failures["record_fixed_fields_invalid"] += 1
                    continue
                predicate_key = (sentence_index, predicate_index)
                prior = seen_predicates.get(predicate_key)
                if prior is not None:
                    if prior == row:
                        exact_duplicate_predicates += 1
                    else:
                        conflicting_duplicate_predicates += 1
                        conflicting_duplicate_documents.add(document)
                else:
                    seen_predicates[predicate_key] = row

                has_text = document in layer_documents["text"]
                terminal_counts = ptb_terminal_counts.get(document)
                if not has_text and terminal_counts is None:
                    full_join_failures[
                        "text_and_ptb_document_missing_or_invalid"
                    ] += 1
                    continue
                if not has_text:
                    full_join_failures["text_document_missing"] += 1
                    continue
                if terminal_counts is None:
                    full_join_failures["ptb_document_missing_or_invalid"] += 1
                    continue
                if sentence_index < 0 or sentence_index >= len(terminal_counts):
                    full_join_failures["sentence_index_out_of_range"] += 1
                    continue
                if (
                    predicate_index < 0
                    or predicate_index >= terminal_counts[sentence_index]
                ):
                    full_join_failures["predicate_terminal_out_of_range"] += 1
                    continue
                full_joined_rows += 1
                if not is_wsj:
                    full_non_wsj_joined_rows += 1

        rights_family_counts: Counter[str] = Counter()
        rights_disposition_counts: Counter[str] = Counter()
        for document in sorted(document_union):
            rights_family_counts[source_family(document)] += 1
            rights_disposition_counts[source_disposition(document)] += 1

        rights_joined = joined_documents & allowed_documents
        g2_tree_failures: Counter[str] = Counter()
        g2_tree_failure_documents: dict[str, str] = {}
        g2_ambiguous_prop_documents: list[str] = []
        g2_row_failures: Counter[str] = Counter()
        g2_exact_duplicate_predicates = 0
        g2_conflicting_duplicate_predicates = 0
        g2_candidate_rows = 0
        g2_retained_rows = 0
        g2_documents: dict[str, tuple[Any, str, tuple[str, ...]]] = {}
        g2_exclusion_reasons: dict[str, set[str]] = {}
        g2_exclusion_rows: dict[str, int] = {}

        def exclude_from_g2(document: str, reason: str) -> None:
            g2_exclusion_reasons.setdefault(document, set()).add(reason)
            g2_exclusion_rows[document] = len(
                primary_prop_rows.get(document, ())
            )

        for document in sorted(denied_override_documents):
            exclude_from_g2(document, "wsj_denied_override")

        for document in sorted(allowed_documents):
            for layer in ("text", "ptb", "prop"):
                if document not in layer_documents[layer]:
                    exclude_from_g2(document, f"{layer}_layer_missing")
            if document in genre_conflicts:
                exclude_from_g2(document, "cross_layer_genre_conflict")

        for document in sorted(rights_joined):
            if document in g2_exclusion_reasons:
                continue
            prop_path = primary_prop_paths.get(document)
            if prop_path is None:
                g2_ambiguous_prop_documents.append(document)
                exclude_from_g2(document, "prop_path_ambiguous")
                continue
            ptb_paths = layer_maps["ptb"][document]
            text_paths = layer_maps["text"][document]
            if len(ptb_paths) != 1 or len(text_paths) != 1:
                g2_ambiguous_prop_documents.append(document)
                exclude_from_g2(document, "layer_path_ambiguous")
                continue
            try:
                ptb_source, ptb_encoding = _decode_text(
                    archive.read(ptb_paths[0])
                )
                trees = parse_penn_trees(ptb_source)
            except (PropBankAdapterError, UnicodeError) as error:
                reason = getattr(error, "reason_code", "ptb_decode_error")
                g2_tree_failures[reason] += 1
                g2_tree_failure_documents[document] = reason
                exclude_from_g2(document, f"ptb_{reason}")
                continue

            prop_source, prop_encoding = _decode_text(archive.read(prop_path))
            rows = tuple(
                line.strip()
                for line in prop_source.splitlines()
                if line.strip()
            )
            g2_candidate_rows += len(rows)
            seen_predicates: dict[tuple[int, int], str] = {}
            document_row_failures = 0
            for row in rows:
                try:
                    (
                        _dialect,
                        sentence_index,
                        predicate_index,
                        _label_fields,
                    ) = _validate_local_record(row)
                except ValueError as error:
                    reason = str(error)
                    g2_row_failures[reason] += 1
                    document_row_failures += 1
                    exclude_from_g2(document, reason)
                    continue
                if sentence_index < 0 or sentence_index >= len(trees):
                    reason = "sentence_index_out_of_range"
                    g2_row_failures[reason] += 1
                    document_row_failures += 1
                    exclude_from_g2(document, reason)
                    continue
                if predicate_index >= len(trees[sentence_index].terminals()):
                    reason = "predicate_terminal_out_of_range"
                    g2_row_failures[reason] += 1
                    document_row_failures += 1
                    exclude_from_g2(document, reason)
                    continue
                key = (sentence_index, predicate_index)
                prior = seen_predicates.get(key)
                if prior is not None:
                    if prior == row:
                        reason = "exact_duplicate_predicate"
                        g2_exact_duplicate_predicates += 1
                    else:
                        reason = "conflicting_duplicate_predicate"
                        g2_conflicting_duplicate_predicates += 1
                    g2_row_failures[reason] += 1
                    document_row_failures += 1
                    exclude_from_g2(document, reason)
                else:
                    seen_predicates[key] = row
            if document_row_failures:
                continue
            g2_documents[document] = (
                trees,
                prop_encoding,
                rows,
            )
            g2_retained_rows += len(rows)
            del ptb_encoding

        inventory: Counter[str] = Counter()
        conversion_failures: Counter[str] = Counter()
        structural: Counter[str] = Counter()
        converted = 0
        eligible_verbal = 0
        nonverbal = 0
        policy = _source_policy(allowed_documents)

        for document, (trees, _prop_encoding, rows) in g2_documents.items():
            for row in rows:
                _inventory_record(row, inventory)
                kind = _local_record_kind(row)
                if kind not in {"v", "classic"}:
                    nonverbal += 1
                    continue
                eligible_verbal += 1
                try:
                    record = parse_propbank_record(f"{document} {row}")
                except PropBankAdapterError as error:
                    conversion_failures[error.reason_code] += 1
                    continue
                if record.sentence_index >= len(trees):
                    conversion_failures["sentence_index_out_of_range"] += 1
                    continue

                tree = trees[record.sentence_index]
                structural["records_examined"] += 1
                record_has_no_surface = False
                record_has_unlinked_no_surface = False
                record_has_unindexed_pro = False
                terminals = tree.terminals()
                for argument in record.arguments:
                    try:
                        resolved = resolve_pointer(argument.pointer, tree)
                    except PropBankAdapterError:
                        continue
                    if any(member.surface_indexes for member in resolved.members):
                        continue
                    structural["no_surface_arguments"] += 1
                    record_has_no_surface = True
                    argument_nodes = set(argument.pointer.nodes)
                    linked = any(
                        argument_nodes.intersection(link.pointer.nodes)
                        for link in record.links
                    )
                    if linked:
                        structural["linked_no_surface_arguments"] += 1
                        continue
                    structural["unlinked_no_surface_arguments"] += 1
                    record_has_unlinked_no_surface = True
                    selected_terminal_indexes = {
                        index
                        for member in resolved.members
                        for piece in member.pieces
                        for index in piece.terminal_indexes
                    }
                    if any(
                        terminals[index].word == "*PRO*"
                        for index in selected_terminal_indexes
                    ):
                        record_has_unindexed_pro = True
                if record_has_no_surface:
                    structural["records_with_no_surface"] += 1
                if record_has_unlinked_no_surface:
                    structural["records_with_unlinked_no_surface"] += 1
                if record_has_unindexed_pro:
                    structural["records_with_unlinked_unindexed_pro"] += 1

                try:
                    convert_propbank_record(record, tree, source_policy=policy)
                except PropBankAdapterError as error:
                    conversion_failures[error.reason_code] += 1
                else:
                    converted += 1

        failure_total = sum(conversion_failures.values())
        if converted + failure_total != eligible_verbal:
            raise RuntimeError("conversion counts do not reconcile")
        coverage = converted / eligible_verbal if eligible_verbal else 0.0
        unindexed_pro_records = structural[
            "records_with_unlinked_unindexed_pro"
        ]
        optimistic_ceiling = (
            (eligible_verbal - unindexed_pro_records) / eligible_verbal
            if eligible_verbal
            else 0.0
        )

        g2_exclusions = {
            document: {
                "reasons": sorted(reasons),
                "canonical_prop_rows": g2_exclusion_rows[document],
            }
            for document, reasons in sorted(g2_exclusion_reasons.items())
        }
        g2_pass = bool(g2_documents) and not g2_exclusions
        g3_pass = coverage >= CONVERSION_THRESHOLD
        rights_pass = False
        overall_go = (
            acquisition_pass
            and archive_safe
            and rights_pass
            and g2_pass
            and g3_pass
        )
        blocking_gates: list[str] = []
        if not acquisition_pass:
            blocking_gates.append("acquisition_pin")
        if not archive_safe:
            blocking_gates.append("archive_safety")
        if not rights_pass:
            blocking_gates.append("g1_rights")
        if not g2_pass:
            blocking_gates.append("g2_join")
        if not g3_pass:
            blocking_gates.append("g3_annotation_fit")
        if not acquisition_pass:
            decision_reason = "archive_pin_mismatch"
        elif not archive_safe:
            decision_reason = "archive_safety_check_failed"
        elif sum((not rights_pass, not g2_pass, not g3_pass)) > 1:
            decision_reason = "multiple_feasibility_gates_failed"
        elif not rights_pass:
            decision_reason = "rights_gate_incomplete"
        elif not g2_pass:
            decision_reason = "retained_subset_join_gate_failed"
        elif not g3_pass:
            decision_reason = "gold_span_conversion_below_predeclared_threshold"
        else:
            decision_reason = "all_preparation_gates_met"

        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "acquisition": {
                "filename": path.name,
                "bytes": archive_bytes,
                "sha256": archive_sha256,
                "matches_pinned_filename": path.name == PINNED_ARCHIVE_FILENAME,
                "matches_pinned_bytes": archive_bytes == PINNED_ARCHIVE_BYTES,
                "matches_pinned_sha256": archive_sha256
                == PINNED_ARCHIVE_SHA256,
            },
            "archive": {
                "status": "pass" if archive_safe else "fail",
                "entries": len(infos),
                "files": len(files),
                "clean_files": len(clean_files),
                "svn_files_ignored": len(files) - len(clean_files),
                "uncompressed_bytes": uncompressed_bytes,
                "compressed_member_bytes": compressed_member_bytes,
                "member_compression_ratio": (
                    uncompressed_bytes / compressed_member_bytes
                    if compressed_member_bytes
                    else None
                ),
                "crc_error_member": bad_member,
                "unsafe_paths": unsafe_paths,
                "symlinks": symlinks,
                "encrypted_files": encrypted,
                "duplicate_entry_names": duplicate_names,
                "casefold_name_collisions": casefold_name_collisions,
                "readme_present": _README_PATH in archive.namelist(),
                "archive_level_legal_notice_files": (
                    archive_level_legal_notice_files
                ),
                "frame_legal_notice_files": frame_legal_notice_files,
                "suffix_counts": dict(sorted(suffix_counts.items())),
            },
            "layers": layer_summary,
            "join": {
                "canonical_union_documents": len(document_union),
                "all_three_layer_documents": len(joined_documents),
                "all_three_layer_non_wsj_documents": sum(
                    not document.casefold().startswith("wsj_")
                    for document in joined_documents
                ),
                "wsj_union_documents": len(wsj_union),
                "missing_by_layer": missing_by_layer,
                "invalid_genre_paths": sorted(invalid_genre_paths),
                "cross_layer_genre_conflicts": genre_conflicts,
                "canonical_document_genre_counts": dict(
                    sorted(canonical_genre_counts.items())
                ),
                "duplicate_canonical_paths": duplicate_canonical_paths,
                "identical_alternate_prop_rows": duplicate_prop_rows,
                "ambiguous_prop_documents": sorted(
                    ambiguous_prop_documents
                ),
                "physical_prop_rows": physical_prop_rows,
                "physical_prop_dialects": dict(
                    sorted(physical_prop_dialects.items())
                ),
                "canonical_prop_rows": sum(
                    len(rows) for rows in primary_prop_rows.values()
                ),
                "canonical_prop_dialects": dict(
                    sorted(canonical_prop_dialects.items())
                ),
                "ptb_structurally_valid_documents": len(ptb_terminal_counts),
                "ptb_structural_failure_documents": ptb_structure_failures,
                "ptb_structurally_valid_sentences": sum(
                    len(counts) for counts in ptb_terminal_counts.values()
                ),
                "full_joined_prop_rows": full_joined_rows,
                "full_join_failures": dict(sorted(full_join_failures.items())),
                "non_wsj_canonical_prop_rows": full_non_wsj_rows,
                "non_wsj_fully_joined_prop_rows": full_non_wsj_joined_rows,
                "rows_without_exactly_one_rel": invalid_rel_rows,
                "exact_duplicate_predicate_instances": (
                    exact_duplicate_predicates
                ),
                "conflicting_duplicate_predicate_instances": (
                    conflicting_duplicate_predicates
                ),
                "conflicting_duplicate_predicate_documents": sorted(
                    conflicting_duplicate_documents
                ),
            },
            "rights": {
                "status": "incomplete_provisional_diagnostic_manifest",
                "requested_allowed_documents": len(
                    requested_allowed_documents
                ),
                "effective_allowed_documents": len(allowed_documents),
                "denied_override_documents": sorted(
                    denied_override_documents
                ),
                "joined_allowed_documents": len(rights_joined),
                "source_family_union_counts": dict(
                    sorted(rights_family_counts.items())
                ),
                "disposition_union_counts": dict(
                    sorted(rights_disposition_counts.items())
                ),
                "wsj_policy": "deny_all_layers",
            },
            "g2": {
                "status": "pass" if g2_pass else "fail",
                "scope": "explicit_allowed_document_manifest",
                "full_archive_status": "hold_for_join_and_rights_exclusions",
                "candidate_joined_documents": len(rights_joined),
                "retained_documents": len(g2_documents),
                "candidate_rows_after_tree_parse": g2_candidate_rows,
                "retained_rows": g2_retained_rows,
                "exclusions": g2_exclusions,
                "excluded_documents": len(g2_exclusions),
                "excluded_canonical_prop_rows": sum(
                    item["canonical_prop_rows"]
                    for item in g2_exclusions.values()
                ),
                "tree_failure_documents": g2_tree_failure_documents,
                "tree_failure_reasons": dict(sorted(g2_tree_failures.items())),
                "ambiguous_prop_documents": sorted(
                    g2_ambiguous_prop_documents
                ),
                "row_failures": dict(sorted(g2_row_failures.items())),
                "exact_duplicate_predicate_instances": (
                    g2_exact_duplicate_predicates
                ),
                "conflicting_duplicate_predicate_instances": (
                    g2_conflicting_duplicate_predicates
                ),
            },
            "g3": {
                "status": "pass" if g3_pass else "fail",
                "threshold": CONVERSION_THRESHOLD,
                "eligible_verbal_records": eligible_verbal,
                "nonverbal_records_excluded": nonverbal,
                "converted_records": converted,
                "conversion_failures": dict(
                    sorted(conversion_failures.items())
                ),
                "conversion_coverage": coverage,
                "annotation_inventory": dict(sorted(inventory.items())),
                "structural_findings": dict(sorted(structural.items())),
                "optimistic_exact_span_ceiling": optimistic_ceiling,
                "ceiling_assumption": (
                    "every non-*PRO* failure is repaired or exactly resolved"
                ),
            },
            "decision": {
                "overall": "go" if overall_go else "no_go",
                "reason": decision_reason,
                "blocking_gates": blocking_gates,
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and print deterministic aggregate JSON."""

    parser = argparse.ArgumentParser(
        description="Audit MASC PropBank without extracting corpus files."
    )
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    report = audit_masc_archive(args.archive)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
