"""Validate and convert the normalized candidate-pilot annotation workbook.

The pilot workbook is a human-annotation interface, not a QA-SRL or QANom
release.  A completed pass therefore converts directly into the model-neutral
evaluation bundle without inventing release-specific inflection or judgment
fields.  Exact character spans are checked against the authored note and mapped
to a frozen, dependency-free tokenizer before any scorer artifact is produced.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
import posixpath
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree
import zipfile

from .annotation_schema import QASRLQuestionSlots
from .datasets.common import DatasetFormatError
from .datasets.io import sha256_file
from .evaluation.bundle import EvaluationBundle
from .evaluation.types import (
    EvaluationArgument,
    EvaluationCorpus,
    EvaluationMentionQualifier,
    EvaluationPredicate,
    EvaluationQAPair,
    EvaluationQuestion,
    PredicateKey,
)
from .schema import MENTION_QUALIFIER_KINDS


PILOT_WORKBOOK_CONTRACT_VERSION = "candidate-pilot-normalized-v1"
PILOT_WORKBOOK_ADAPTER_VERSION = "candidate-pilot-evaluation-adapter-v1"
PILOT_TOKENIZATION_VERSION = "pilot-source-tokenizer-v1"
PILOT_CONSOLIDATION_RULE = "pilot-single-annotation-v1"
PILOT_PREDICATE_SOURCE = "single-human-pilot-annotation"
PILOT_RECORD_QUARANTINE = "challenge-record-quarantine-v1"
PILOT_AUTHORING_FINGERPRINT = (
    "d0d7fc850d923ad99c4800bfab0f9014e2c23d1eb81183012167f3566bc00c40"
)

_SHEET_TABLES: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "Authoring": (
        "PilotAuthoringTable",
        (
            "record_id",
            "document_id",
            "split",
            "scenario",
            "raw_note",
            "target_phenomena",
            "author",
            "authored_on",
            "rights_basis",
            "release_allowed",
            "privacy_reviewed",
            "derived_from",
            "predicate_families",
            "source_reference",
            "license_id",
            "notes",
            "ready",
        ),
    ),
    "Annotation": (
        "PilotCandidateTable",
        (
            "record_id",
            "annotator_id",
            "pass_id",
            "candidate_id",
            "predicate_text",
            "predicate_start",
            "predicate_end",
            "predicate_type",
            "lemma",
            "eventive",
            "related_verbal_form",
            "qualifier_assessed",
            "decision",
            "ambiguity_or_exclusion",
            "rationale",
        ),
    ),
    "Qualifier Evidence": (
        "PilotQualifierEvidenceTable",
        (
            "record_id",
            "annotator_id",
            "pass_id",
            "candidate_id",
            "qualifier_kind",
            "span_order",
            "evidence_text",
            "evidence_start",
            "evidence_end",
            "rationale",
        ),
    ),
    "Questions": (
        "PilotQuestionTable",
        (
            "record_id",
            "annotator_id",
            "pass_id",
            "candidate_id",
            "question_id",
            "question_surface",
            "wh",
            "aux",
            "subj",
            "verb",
            "obj",
            "prep",
            "obj2",
            "is_passive",
            "is_negated",
            "rationale",
        ),
    ),
    "Answer Spans": (
        "PilotAnswerSpanTable",
        (
            "record_id",
            "annotator_id",
            "pass_id",
            "candidate_id",
            "question_id",
            "alternative_id",
            "span_order",
            "answer_text",
            "answer_start",
            "answer_end",
            "rationale",
        ),
    ),
    "Review Log": (
        "PilotReviewTable",
        (
            "record_id",
            "candidate_id",
            "field",
            "pass_1_value",
            "pass_2_value",
            "resolution",
            "guide_change",
            "excluded",
            "reviewed_by",
            "reviewed_on",
        ),
    ),
    "Contract": (
        "PilotContractTable",
        ("key", "value"),
    ),
}

_CONTRACT_VALUES = {
    "annotation_contract_version": PILOT_WORKBOOK_CONTRACT_VERSION,
    "tokenizer_version": PILOT_TOKENIZATION_VERSION,
    "consolidation_rule": PILOT_CONSOLIDATION_RULE,
    "record_quarantine_policy": PILOT_RECORD_QUARANTINE,
    "evaluation_use": "guide-development-only",
    "not_for_model_selection": "true",
    "protocol_version": "candidate-pilot-v1",
    "guide_version": "candidate-pilot-v1",
    "workbook_pass_policy": "one-annotator-one-pass",
}

_SCENARIOS = {
    "customer_support",
    "it_incident_change",
    "billing_approvals",
    "logistics_fulfillment",
    "project_administration",
}
_RIGHTS_BASES = {"newly_authored", "public_domain", "licensed"}
_DECISIONS = {"include", "exclude"}
_PILOT_WH_VALUES = {
    "how",
    "how long",
    "how much",
    "what",
    "when",
    "where",
    "who",
    "why",
}
_REQUIRED_PHENOMENA = {
    "verbal_predicate",
    "nominal_predicate",
    "passive_voice",
    "coordination",
    "multiple_predicates",
    "temporal_circumstance",
    "long_distance_argument",
    "non_eventive_nominal",
    *MENTION_QUALIFIER_KINDS,
}
_ALLOWED_PHENOMENA = _REQUIRED_PHENOMENA | {"embedded_argument"}
_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_TAG_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_TOKEN_PATTERN = re.compile(
    r"\d+(?:[.:]\d+)+|\w+(?:[-'’]\w+)*|[^\w\s]",
    re.UNICODE,
)
_CELL_REFERENCE = re.compile(r"([A-Z]+)([1-9][0-9]*)\Z")
_TABLE_REFERENCE = re.compile(r"A1:([A-Z]+)([1-9][0-9]*)\Z")
_XML_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
_OFFICE_RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_MAX_ARCHIVE_FILES = 512
_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_XML_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PilotWorkbookValidation:
    """Machine-readable validation summary for one workbook snapshot."""

    workbook_sha256: str
    authoring_fingerprint: str
    authoring_record_count: int
    candidate_row_count: int
    qualifier_evidence_row_count: int
    question_row_count: int
    answer_span_row_count: int
    active_review_row_count: int
    annotator_id: str | None = None
    pass_id: str | None = None
    candidate_count: int | None = None
    excluded_candidate_count: int | None = None
    quarantined_record_count: int | None = None
    annotated_qa_pair_count: int | None = None
    qa_pair_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        selection = None
        if self.annotator_id is not None:
            selection = {
                "annotator_id": self.annotator_id,
                "pass_id": self.pass_id,
                "candidate_count": self.candidate_count,
                "excluded_candidate_count": self.excluded_candidate_count,
                "quarantined_record_count": self.quarantined_record_count,
                "annotated_qa_pair_count": self.annotated_qa_pair_count,
                "qa_pair_count": self.qa_pair_count,
            }
        return {
            "valid": True,
            "annotation_status": (
                "annotation_pending" if selection is None else "single_annotator_pass"
            ),
            "contract_version": PILOT_WORKBOOK_CONTRACT_VERSION,
            "adapter_version": PILOT_WORKBOOK_ADAPTER_VERSION,
            "tokenization_version": PILOT_TOKENIZATION_VERSION,
            "workbook_sha256": self.workbook_sha256,
            "authoring_fingerprint": self.authoring_fingerprint,
            "authoring_record_count": self.authoring_record_count,
            "candidate_row_count": self.candidate_row_count,
            "qualifier_evidence_row_count": self.qualifier_evidence_row_count,
            "question_row_count": self.question_row_count,
            "answer_span_row_count": self.answer_span_row_count,
            "active_review_row_count": self.active_review_row_count,
            "selection": selection,
            "ready_for_annotation": True,
            "ready_for_scoring": selection is not None,
            "not_for_model_selection": True,
        }


@dataclass(frozen=True, slots=True)
class PilotWorkbookConversion:
    """A scorer-ready bundle plus compact conversion counts."""

    bundle: EvaluationBundle
    workbook_sha256: str
    authoring_fingerprint: str
    annotation_fingerprint: str
    authoring_record_count: int
    candidate_count: int
    excluded_candidate_count: int
    quarantined_record_count: int
    annotated_qa_pair_count: int
    qa_pair_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": PILOT_WORKBOOK_CONTRACT_VERSION,
            "adapter_version": PILOT_WORKBOOK_ADAPTER_VERSION,
            "tokenization_version": PILOT_TOKENIZATION_VERSION,
            "workbook_sha256": self.workbook_sha256,
            "authoring_fingerprint": self.authoring_fingerprint,
            "annotation_fingerprint": self.annotation_fingerprint,
            "authoring_record_count": self.authoring_record_count,
            "candidate_count": self.candidate_count,
            "excluded_candidate_count": self.excluded_candidate_count,
            "quarantined_record_count": self.quarantined_record_count,
            "annotated_qa_pair_count": self.annotated_qa_pair_count,
            "qa_pair_count": self.qa_pair_count,
        }


@dataclass(frozen=True, slots=True)
class _SheetRow:
    sheet_name: str
    row_number: int
    values: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _WorkbookSnapshot:
    workbook_sha256: str
    rows: Mapping[str, tuple[_SheetRow, ...]]


@dataclass(frozen=True, slots=True)
class _AuthoringRecord:
    row_number: int
    record_id: str
    document_id: str
    text: str
    scenario: str
    values: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _SourceToken:
    index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _GroundedSpan:
    text: str
    character_start: int
    character_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    row: _SheetRow
    record_id: str
    candidate_id: str
    decision: str
    predicate_type: str
    predicate_span: _GroundedSpan
    lemma: str | None
    eventive: bool | None
    related_verbal_form: str | None
    qualifier_assessed: bool
    ambiguity_or_exclusion: str | None
    rationale: str | None


def validate_pilot_workbook(path: str | Path) -> PilotWorkbookValidation:
    """Validate the checked-in template or one completed single-human pass."""

    snapshot = _read_snapshot(path)
    records, authoring_fingerprint = _parse_authoring_rows(
        snapshot.rows["Authoring"]
    )
    _validate_contract(snapshot.rows["Contract"])
    active_review_rows = _active_review_rows(snapshot.rows["Review Log"])
    if active_review_rows:
        raise DatasetFormatError(
            "Review Log must remain blank in a one-annotator, one-pass workbook"
        )
    annotation_rows = _annotation_rows(snapshot)
    if not annotation_rows:
        return PilotWorkbookValidation(
            workbook_sha256=snapshot.workbook_sha256,
            authoring_fingerprint=authoring_fingerprint,
            authoring_record_count=len(records),
            candidate_row_count=0,
            qualifier_evidence_row_count=0,
            question_row_count=0,
            answer_span_row_count=0,
            active_review_row_count=len(active_review_rows),
        )

    annotator_id, pass_id = _single_annotation_group(annotation_rows)
    conversion = _convert_snapshot(
        snapshot,
        records,
        authoring_fingerprint=authoring_fingerprint,
        annotator_id=annotator_id,
        pass_id=pass_id,
    )
    return PilotWorkbookValidation(
        workbook_sha256=snapshot.workbook_sha256,
        authoring_fingerprint=authoring_fingerprint,
        authoring_record_count=len(records),
        candidate_row_count=len(snapshot.rows["Annotation"]),
        qualifier_evidence_row_count=len(snapshot.rows["Qualifier Evidence"]),
        question_row_count=len(snapshot.rows["Questions"]),
        answer_span_row_count=len(snapshot.rows["Answer Spans"]),
        active_review_row_count=len(active_review_rows),
        annotator_id=annotator_id,
        pass_id=pass_id,
        candidate_count=conversion.candidate_count,
        excluded_candidate_count=conversion.excluded_candidate_count,
        quarantined_record_count=conversion.quarantined_record_count,
        annotated_qa_pair_count=conversion.annotated_qa_pair_count,
        qa_pair_count=conversion.qa_pair_count,
    )


def convert_pilot_workbook(
    path: str | Path,
    *,
    annotator_id: str,
    pass_id: str,
) -> PilotWorkbookConversion:
    """Convert exactly one completed workbook pass into an evaluation bundle."""

    snapshot = _read_snapshot(path)
    records, authoring_fingerprint = _parse_authoring_rows(
        snapshot.rows["Authoring"]
    )
    _validate_contract(snapshot.rows["Contract"])
    annotation_rows = _annotation_rows(snapshot)
    if not annotation_rows:
        raise DatasetFormatError(
            "pilot annotation is pending; no completed pass can be converted"
        )
    actual_annotator, actual_pass = _single_annotation_group(annotation_rows)
    expected_annotator = _identifier(annotator_id, label="annotator_id")
    expected_pass = _pass_identifier(pass_id, label="pass_id")
    if (actual_annotator, actual_pass) != (expected_annotator, expected_pass):
        raise DatasetFormatError(
            "workbook annotation group does not match the requested selection; "
            f"found=({actual_annotator}, {actual_pass}), "
            f"requested=({expected_annotator}, {expected_pass})"
        )
    return _convert_snapshot(
        snapshot,
        records,
        authoring_fingerprint=authoring_fingerprint,
        annotator_id=actual_annotator,
        pass_id=actual_pass,
    )


def tokenize_pilot_text(text: str) -> tuple[_SourceToken, ...]:
    """Apply the frozen Unicode/code-point tokenization used by the pilot."""

    if not isinstance(text, str) or not text:
        raise DatasetFormatError("pilot source text cannot be empty")
    return tuple(
        _SourceToken(
            index=index,
            text=match.group(0),
            start=match.start(),
            end=match.end(),
        )
        for index, match in enumerate(_TOKEN_PATTERN.finditer(text))
    )


def _convert_snapshot(
    snapshot: _WorkbookSnapshot,
    records: tuple[_AuthoringRecord, ...],
    *,
    authoring_fingerprint: str,
    annotator_id: str,
    pass_id: str,
) -> PilotWorkbookConversion:
    if _active_review_rows(snapshot.rows["Review Log"]):
        raise DatasetFormatError(
            "Review Log must remain blank in a one-annotator, one-pass workbook"
        )
    record_by_id = {record.record_id: record for record in records}
    candidates = _parse_candidates(
        snapshot.rows["Annotation"],
        record_by_id=record_by_id,
        annotator_id=annotator_id,
        pass_id=pass_id,
    )
    candidate_by_key = {
        (candidate.record_id, candidate.candidate_id): candidate
        for candidate in candidates
    }
    qualifier_rows = _foreign_key_rows(
        snapshot.rows["Qualifier Evidence"],
        record_by_id=record_by_id,
        candidate_by_key=candidate_by_key,
        annotator_id=annotator_id,
        pass_id=pass_id,
    )
    question_rows = _foreign_key_rows(
        snapshot.rows["Questions"],
        record_by_id=record_by_id,
        candidate_by_key=candidate_by_key,
        annotator_id=annotator_id,
        pass_id=pass_id,
    )
    answer_rows = _foreign_key_rows(
        snapshot.rows["Answer Spans"],
        record_by_id=record_by_id,
        candidate_by_key=candidate_by_key,
        annotator_id=annotator_id,
        pass_id=pass_id,
    )

    candidates_by_record: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_record[candidate.record_id].append(candidate)
    missing_records = set(record_by_id) - set(candidates_by_record)
    if missing_records:
        raise DatasetFormatError(
            "a complete pilot pass must contain at least one candidate for every "
            f"Authoring record; missing={sorted(missing_records)}"
        )

    qualifier_rows_by_candidate: dict[tuple[str, str], list[_SheetRow]] = (
        defaultdict(list)
    )
    for row in qualifier_rows:
        qualifier_rows_by_candidate[_candidate_key(row)].append(row)
    question_by_key = _parse_questions(
        question_rows,
        candidate_by_key=candidate_by_key,
    )
    answer_rows_by_question: dict[tuple[str, str, str], list[_SheetRow]] = (
        defaultdict(list)
    )
    for row in answer_rows:
        key = (*_candidate_key(row), _id_cell(row, "question_id"))
        if key not in question_by_key:
            raise DatasetFormatError(
                f"{_cell_label(row, 'question_id')} does not match Questions: "
                f"{row.values['question_id']}"
            )
        answer_rows_by_question[key].append(row)
    questions_by_candidate: dict[
        tuple[str, str], list[tuple[tuple[str, str, str], EvaluationQuestion]]
    ] = defaultdict(list)
    for key, question in question_by_key.items():
        questions_by_candidate[key[:2]].append((key, question))

    for key in question_by_key:
        if key not in answer_rows_by_question:
            raise DatasetFormatError(
                f"question {'/'.join(key)} has no Answer Spans rows"
            )

    for candidate in candidates:
        key = (candidate.record_id, candidate.candidate_id)
        if (
            candidate.decision == "include"
            and candidate.eventive is True
            and key not in questions_by_candidate
        ):
            raise DatasetFormatError(
                f"included eventive candidate {candidate.record_id}/"
                f"{candidate.candidate_id} has no Questions row"
            )

    excluded_source_ids = tuple(
        record.record_id
        for record in records
        if any(
            candidate.decision == "exclude"
            for candidate in candidates_by_record[record.record_id]
        )
    )
    excluded_source_set = set(excluded_source_ids)
    included_source_ids = tuple(
        record.record_id
        for record in records
        if record.record_id not in excluded_source_set
    )

    predicates: list[EvaluationPredicate] = []
    candidate_audit: list[dict[str, Any]] = []
    annotated_qa_pair_count = 0
    scored_qa_pair_count = 0
    seen_predicate_keys: set[PredicateKey] = set()
    record_order = {record.record_id: index for index, record in enumerate(records)}
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            record_order[candidate.record_id],
            candidate.predicate_span.character_start,
            candidate.predicate_span.character_end,
            candidate.predicate_type,
            candidate.candidate_id,
        ),
    )
    for candidate in ordered_candidates:
        key = (candidate.record_id, candidate.candidate_id)
        q_rows = qualifier_rows_by_candidate.get(key, [])
        questions = questions_by_candidate.get(key, [])
        if candidate.decision == "exclude":
            if q_rows or questions:
                raise DatasetFormatError(
                    f"excluded candidate {candidate.record_id}/{candidate.candidate_id} "
                    "cannot have qualifier evidence or questions"
                )
        if candidate.eventive is False and (q_rows or questions):
            raise DatasetFormatError(
                f"non-eventive candidate {candidate.record_id}/"
                f"{candidate.candidate_id} cannot have qualifier evidence or questions"
            )

        qualifiers = _convert_qualifiers(
            q_rows,
            candidate=candidate,
            record=record_by_id[candidate.record_id],
        )
        pairs: list[EvaluationQAPair] = []
        for question_key, question in sorted(
            questions,
            key=lambda item: item[0][2],
        ):
            pairs.extend(
                _convert_answers(
                    answer_rows_by_question[question_key],
                    question=question,
                    record=record_by_id[candidate.record_id],
                    candidate_id=candidate.candidate_id,
                    question_id=question_key[2],
                )
            )
        annotated_qa_pair_count += len(pairs)

        audit = {
            "record_id": candidate.record_id,
            "candidate_id": candidate.candidate_id,
            "decision": candidate.decision,
            "predicate_text": candidate.predicate_span.text,
            "predicate_character_start": candidate.predicate_span.character_start,
            "predicate_character_end": candidate.predicate_span.character_end,
            "predicate_token_start": candidate.predicate_span.token_start,
            "predicate_token_end": candidate.predicate_span.token_end,
            "predicate_type": candidate.predicate_type,
            "lemma": candidate.lemma,
            "eventive": candidate.eventive,
            "related_verbal_form": candidate.related_verbal_form,
            "qualifier_assessed": candidate.qualifier_assessed,
            "ambiguity_or_exclusion": candidate.ambiguity_or_exclusion,
            "rationale": candidate.rationale,
            "sheet_row": candidate.row.row_number,
            "record_quarantined": candidate.record_id in excluded_source_set,
        }
        candidate_audit.append(audit)

        if candidate.record_id in excluded_source_set:
            continue
        if candidate.decision != "include":
            raise AssertionError("all excluded candidates quarantine their source")
        predicate_key = PredicateKey(
            source_id=candidate.record_id,
            token_start=candidate.predicate_span.token_start,
            token_end=candidate.predicate_span.token_end,
            predicate_type=candidate.predicate_type,
        )
        if predicate_key in seen_predicate_keys:
            raise DatasetFormatError(
                f"duplicate predicate identity in {candidate.record_id}: tokens "
                f"[{predicate_key.token_start}, {predicate_key.token_end}), "
                f"{candidate.predicate_type}"
            )
        seen_predicate_keys.add(predicate_key)
        scored_qa_pair_count += len(pairs)
        predicates.append(
            EvaluationPredicate(
                key=predicate_key,
                is_eventive=bool(candidate.eventive),
                lemma=candidate.lemma,
                pairs=tuple(pairs),
                mention_qualifiers=qualifiers,
            )
        )

    if not predicates:
        raise DatasetFormatError(
            "record quarantine removed every predicate from the completed pass"
        )
    if scored_qa_pair_count == 0:
        raise DatasetFormatError(
            "completed pilot pass contains no scorer-ready question-answer pairs"
        )

    annotation_rows = _annotation_rows(snapshot)
    normalized_annotation = sorted(
        (_semantic_row(row) for row in annotation_rows),
        key=lambda item: (
            item["sheet"],
            json.dumps(
                item["values"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )
    annotation_fingerprint = _json_fingerprint(normalized_annotation)
    source_records = []
    for record in records:
        source_records.append(
            {
                "record_id": record.record_id,
                "document_id": record.document_id,
                "text": record.text,
                "scenario": record.scenario,
                "tokens": [
                    {
                        "index": token.index,
                        "text": token.text,
                        "start": token.start,
                        "end": token.end,
                    }
                    for token in tokenize_pilot_text(record.text)
                ],
            }
        )
    metadata = {
        "annotation_status": "single-annotator",
        "evaluation_use": "guide-development-only",
        "not_for_model_selection": True,
        "contract_version": PILOT_WORKBOOK_CONTRACT_VERSION,
        "adapter_version": PILOT_WORKBOOK_ADAPTER_VERSION,
        "tokenization_version": PILOT_TOKENIZATION_VERSION,
        "record_quarantine_policy": PILOT_RECORD_QUARANTINE,
        "workbook_sha256": snapshot.workbook_sha256,
        "authoring_fingerprint": authoring_fingerprint,
        "annotation_fingerprint": annotation_fingerprint,
        "selection": {"annotator_id": annotator_id, "pass_id": pass_id},
        "excluded_source_ids": list(excluded_source_ids),
        "included_source_ids": list(included_source_ids),
        "counts": {
            "authoring_records": len(records),
            "candidate_rows": len(candidates),
            "included_candidates": sum(
                candidate.decision == "include" for candidate in candidates
            ),
            "excluded_candidates": sum(
                candidate.decision == "exclude" for candidate in candidates
            ),
            "quarantined_records": len(excluded_source_ids),
            "scored_predicates": len(predicates),
            "qualifier_evidence_rows": len(qualifier_rows),
            "question_rows": len(question_rows),
            "answer_span_rows": len(answer_rows),
            "annotated_qa_pairs": annotated_qa_pair_count,
            "scored_qa_pairs": scored_qa_pair_count,
        },
        "source_records": source_records,
        "authoring_rows": [
            _normalized_row(row) for row in snapshot.rows["Authoring"]
        ],
        "normalized_annotation_rows": normalized_annotation,
        "annotation_row_audit": [
            _normalized_row(row) for row in annotation_rows
        ],
        "candidate_audit": candidate_audit,
    }
    bundle = EvaluationBundle(
        corpus=EvaluationCorpus(predicates=tuple(predicates)),
        predicate_source=PILOT_PREDICATE_SOURCE,
        consolidation_rule=PILOT_CONSOLIDATION_RULE,
        metadata=metadata,
    )
    return PilotWorkbookConversion(
        bundle=bundle,
        workbook_sha256=snapshot.workbook_sha256,
        authoring_fingerprint=authoring_fingerprint,
        annotation_fingerprint=annotation_fingerprint,
        authoring_record_count=len(records),
        candidate_count=len(candidates),
        excluded_candidate_count=sum(
            candidate.decision == "exclude" for candidate in candidates
        ),
        quarantined_record_count=len(excluded_source_ids),
        annotated_qa_pair_count=annotated_qa_pair_count,
        qa_pair_count=scored_qa_pair_count,
    )


def _parse_candidates(
    rows: Sequence[_SheetRow],
    *,
    record_by_id: Mapping[str, _AuthoringRecord],
    annotator_id: str,
    pass_id: str,
) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    seen_ids: set[tuple[str, str]] = set()
    seen_spans: set[tuple[str, int, int, str]] = set()
    for row in rows:
        _require_group(row, annotator_id=annotator_id, pass_id=pass_id)
        record_id = _record_id(row, record_by_id)
        candidate_id = _id_cell(row, "candidate_id")
        candidate_key = (record_id, candidate_id)
        if candidate_key in seen_ids:
            raise DatasetFormatError(
                f"duplicate Annotation candidate key: {record_id}/{candidate_id}"
            )
        seen_ids.add(candidate_key)
        record = record_by_id[record_id]
        tokens = tokenize_pilot_text(record.text)
        predicate_span = _grounded_span(
            row=row,
            text=record.text,
            tokens=tokens,
            text_field="predicate_text",
            start_field="predicate_start",
            end_field="predicate_end",
        )
        predicate_type = _choice_cell(
            row,
            "predicate_type",
            choices={"verbal", "nominal"},
        )
        span_key = (
            record_id,
            predicate_span.character_start,
            predicate_span.character_end,
            predicate_type,
        )
        if span_key in seen_spans:
            raise DatasetFormatError(
                f"two candidate IDs share predicate identity in {record_id}: "
                f"characters [{predicate_span.character_start}, "
                f"{predicate_span.character_end}), {predicate_type}"
            )
        seen_spans.add(span_key)
        decision = _choice_cell(row, "decision", choices=_DECISIONS)
        related = _optional_controlled_cell(row, "related_verbal_form")
        qualifier_assessed = _boolean_cell(row, "qualifier_assessed")
        ambiguity = _optional_free_text_cell(row, "ambiguity_or_exclusion")
        rationale = _optional_free_text_cell(row, "rationale")
        if decision == "exclude":
            _require_blank(row, "lemma")
            _require_blank(row, "eventive")
            _require_blank(row, "related_verbal_form")
            if qualifier_assessed:
                raise DatasetFormatError(
                    f"{_cell_label(row, 'qualifier_assessed')} must be false for "
                    "an excluded candidate"
                )
            if ambiguity is None or rationale is None:
                raise DatasetFormatError(
                    f"excluded candidate {record_id}/{candidate_id} requires both "
                    "ambiguity_or_exclusion and rationale"
                )
            lemma = None
            eventive = None
        else:
            if ambiguity is not None:
                raise DatasetFormatError(
                    f"{_cell_label(row, 'ambiguity_or_exclusion')} must be blank "
                    "for an included candidate"
                )
            lemma = _controlled_cell(row, "lemma")
            eventive = _boolean_cell(row, "eventive")
            if predicate_type == "verbal" and not eventive:
                raise DatasetFormatError(
                    f"included verbal candidate {record_id}/{candidate_id} must "
                    "set eventive to true"
                )
            if predicate_type == "verbal" and related is not None:
                raise DatasetFormatError(
                    f"{_cell_label(row, 'related_verbal_form')} applies only to "
                    "nominal candidates"
                )
            if not eventive and qualifier_assessed:
                raise DatasetFormatError(
                    f"non-eventive candidate {record_id}/{candidate_id} must set "
                    "qualifier_assessed to false"
                )
            if eventive and not qualifier_assessed:
                raise DatasetFormatError(
                    f"included eventive candidate {record_id}/{candidate_id} must "
                    "complete qualifier assessment"
                )
        candidates.append(
            _Candidate(
                row=row,
                record_id=record_id,
                candidate_id=candidate_id,
                decision=decision,
                predicate_type=predicate_type,
                predicate_span=predicate_span,
                lemma=lemma,
                eventive=eventive,
                related_verbal_form=related,
                qualifier_assessed=qualifier_assessed,
                ambiguity_or_exclusion=ambiguity,
                rationale=rationale,
            )
        )
    return tuple(candidates)


def _foreign_key_rows(
    rows: Sequence[_SheetRow],
    *,
    record_by_id: Mapping[str, _AuthoringRecord],
    candidate_by_key: Mapping[tuple[str, str], _Candidate],
    annotator_id: str,
    pass_id: str,
) -> tuple[_SheetRow, ...]:
    result = []
    for row in rows:
        _require_group(row, annotator_id=annotator_id, pass_id=pass_id)
        record_id = _record_id(row, record_by_id)
        candidate_id = _id_cell(row, "candidate_id")
        if (record_id, candidate_id) not in candidate_by_key:
            raise DatasetFormatError(
                f"{_cell_label(row, 'candidate_id')} does not match Annotation: "
                f"{record_id}/{candidate_id}"
            )
        result.append(row)
    return tuple(result)


def _parse_questions(
    rows: Sequence[_SheetRow],
    *,
    candidate_by_key: Mapping[tuple[str, str], _Candidate],
) -> Mapping[tuple[str, str, str], EvaluationQuestion]:
    result: dict[tuple[str, str, str], EvaluationQuestion] = {}
    signatures: dict[tuple[str, str], set[tuple[object, ...]]] = defaultdict(set)
    for row in rows:
        record_id, candidate_id = _candidate_key(row)
        candidate = candidate_by_key[(record_id, candidate_id)]
        if candidate.decision == "exclude" or candidate.eventive is not True:
            raise DatasetFormatError(
                f"question row {_row_label(row)} refers to an excluded or "
                "non-eventive candidate"
            )
        question_id = _id_cell(row, "question_id")
        key = (record_id, candidate_id, question_id)
        if key in result:
            raise DatasetFormatError(f"duplicate question key: {'/'.join(key)}")
        slots = tuple(
            _slot_cell(row, field)
            for field in ("wh", "aux", "subj", "verb", "obj", "prep", "obj2")
        )
        try:
            QASRLQuestionSlots(
                wh=slots[0],
                aux=slots[1],
                subj=slots[2],
                verb=slots[3],
                obj=slots[4],
                prep=slots[5],
                obj2=slots[6],
            )
        except ValueError as error:
            raise DatasetFormatError(
                f"{_row_label(row)} has invalid QA-SRL slots: {error}"
            ) from error
        if slots[0] not in _PILOT_WH_VALUES or slots[3] == "_":
            raise DatasetFormatError(
                f"{_row_label(row)} must contain a substantive role question "
                "with a supported wh slot and non-placeholder verb slot"
            )
        surface_form = _controlled_cell(row, "question_surface")
        if not surface_form.endswith("?"):
            raise DatasetFormatError(
                f"{_cell_label(row, 'question_surface')} must end with '?'"
            )
        question = EvaluationQuestion(
            surface_form=surface_form,
            wh=slots[0],
            aux=slots[1],
            subj=slots[2],
            verb=slots[3],
            obj=slots[4],
            prep=slots[5],
            obj2=slots[6],
            is_passive=_boolean_cell(row, "is_passive"),
            is_negated=_boolean_cell(row, "is_negated"),
        )
        signature = (
            *("" if value == "_" else value.casefold() for value in slots),
            question.is_passive,
            question.is_negated,
        )
        candidate_key = (record_id, candidate_id)
        if signature in signatures[candidate_key]:
            raise DatasetFormatError(
                f"candidate {record_id}/{candidate_id} repeats a scorer-equivalent "
                "question under a different question_id"
            )
        signatures[candidate_key].add(signature)
        result[key] = question
        _optional_free_text_cell(row, "rationale")
    return result


def _convert_qualifiers(
    rows: Sequence[_SheetRow],
    *,
    candidate: _Candidate,
    record: _AuthoringRecord,
) -> tuple[EvaluationMentionQualifier, ...] | None:
    if not candidate.qualifier_assessed:
        if rows:
            raise DatasetFormatError(
                f"candidate {candidate.record_id}/{candidate.candidate_id} has "
                "Qualifier Evidence rows but qualifier_assessed is false"
            )
        return None
    if candidate.eventive is not True or candidate.decision != "include":
        if rows:
            raise DatasetFormatError(
                f"candidate {candidate.record_id}/{candidate.candidate_id} cannot "
                "contain qualifier evidence"
            )
        return None

    tokens = tokenize_pilot_text(record.text)
    grouped: dict[str, list[tuple[int, _GroundedSpan]]] = defaultdict(list)
    for row in rows:
        kind = _choice_cell(
            row,
            "qualifier_kind",
            choices=set(MENTION_QUALIFIER_KINDS),
        )
        span_order = _nonnegative_integer_cell(row, "span_order")
        span = _grounded_span(
            row=row,
            text=record.text,
            tokens=tokens,
            text_field="evidence_text",
            start_field="evidence_start",
            end_field="evidence_end",
        )
        _optional_free_text_cell(row, "rationale")
        grouped[kind].append((span_order, span))

    qualifiers = []
    for kind in sorted(grouped):
        ordered = _ordered_spans(
            grouped[kind],
            label=(
                f"qualifier {candidate.record_id}/{candidate.candidate_id}/{kind}"
            ),
        )
        qualifiers.append(
            EvaluationMentionQualifier(
                kind=kind,
                evidence=_evaluation_argument(ordered),
            )
        )
    return tuple(qualifiers)


def _convert_answers(
    rows: Sequence[_SheetRow],
    *,
    question: EvaluationQuestion,
    record: _AuthoringRecord,
    candidate_id: str,
    question_id: str,
) -> tuple[EvaluationQAPair, ...]:
    tokens = tokenize_pilot_text(record.text)
    grouped: dict[str, list[tuple[int, _GroundedSpan]]] = defaultdict(list)
    for row in rows:
        alternative_id = _id_cell(row, "alternative_id")
        span_order = _nonnegative_integer_cell(row, "span_order")
        span = _grounded_span(
            row=row,
            text=record.text,
            tokens=tokens,
            text_field="answer_text",
            start_field="answer_start",
            end_field="answer_end",
        )
        _optional_free_text_cell(row, "rationale")
        grouped[alternative_id].append((span_order, span))

    pairs = []
    seen_signatures: set[tuple[tuple[int, int], ...]] = set()
    for alternative_id in sorted(grouped):
        ordered = _ordered_spans(
            grouped[alternative_id],
            label=(
                f"answer {record.record_id}/{candidate_id}/"
                f"{question_id}/{alternative_id}"
            ),
        )
        argument = _evaluation_argument(ordered)
        if argument.token_spans in seen_signatures:
            raise DatasetFormatError(
                f"question {record.record_id}/{candidate_id}/{question_id} has "
                "duplicate answer alternatives"
            )
        seen_signatures.add(argument.token_spans)
        digest = hashlib.sha256(
            json.dumps(
                [record.record_id, candidate_id, question_id, alternative_id],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        pairs.append(
            EvaluationQAPair(
                pair_id=f"pilot-pair-{digest}",
                role_id=question_id,
                question=question,
                argument=argument,
                metadata={
                    "candidate_id": candidate_id,
                    "question_id": question_id,
                    "alternative_id": alternative_id,
                    "contract_version": PILOT_WORKBOOK_CONTRACT_VERSION,
                },
            )
        )
    return tuple(pairs)


def _ordered_spans(
    indexed_spans: Sequence[tuple[int, _GroundedSpan]],
    *,
    label: str,
) -> tuple[_GroundedSpan, ...]:
    by_order: dict[int, _GroundedSpan] = {}
    for order, span in indexed_spans:
        if order in by_order:
            raise DatasetFormatError(f"{label} repeats span_order {order}")
        by_order[order] = span
    expected = list(range(len(by_order)))
    if sorted(by_order) != expected:
        raise DatasetFormatError(
            f"{label} span_order values must be contiguous from 0; "
            f"found={sorted(by_order)}"
        )
    ordered = tuple(by_order[index] for index in expected)
    previous_end = -1
    seen: set[tuple[int, int]] = set()
    for span in ordered:
        signature = (span.character_start, span.character_end)
        if signature in seen:
            raise DatasetFormatError(f"{label} contains a duplicate span")
        if span.character_start < previous_end:
            raise DatasetFormatError(
                f"{label} spans must be in source order and non-overlapping"
            )
        seen.add(signature)
        previous_end = span.character_end
    return ordered


def _evaluation_argument(spans: Sequence[_GroundedSpan]) -> EvaluationArgument:
    if not spans:
        raise DatasetFormatError("an evaluation argument requires at least one span")
    return EvaluationArgument(
        token_spans=tuple((span.token_start, span.token_end) for span in spans),
        character_spans=tuple(
            (span.character_start, span.character_end) for span in spans
        ),
    )


def _parse_authoring_rows(
    rows: Sequence[_SheetRow],
) -> tuple[tuple[_AuthoringRecord, ...], str]:
    if len(rows) != 20:
        raise DatasetFormatError(
            f"Authoring must contain exactly 20 records; found {len(rows)}"
        )
    records: list[_AuthoringRecord] = []
    seen_records: set[str] = set()
    seen_documents: set[str] = set()
    seen_notes: set[str] = set()
    scenarios: Counter[str] = Counter()
    all_phenomena: set[str] = set()
    for row in rows:
        record_id = _id_cell(row, "record_id")
        document_id = _id_cell(row, "document_id")
        if _controlled_cell(row, "split") != "pilot":
            raise DatasetFormatError(f"{_cell_label(row, 'split')} must be pilot")
        scenario = _choice_cell(row, "scenario", choices=_SCENARIOS)
        text = _free_text_cell(row, "raw_note")
        phenomena = _parse_tags(row, "target_phenomena")
        unsupported_phenomena = set(phenomena) - _ALLOWED_PHENOMENA
        if unsupported_phenomena:
            raise DatasetFormatError(
                f"{_cell_label(row, 'target_phenomena')} contains unsupported "
                f"values: {sorted(unsupported_phenomena)}"
            )
        all_phenomena.update(phenomena)
        if _controlled_cell(row, "author") != "Ayo Adetayo":
            raise DatasetFormatError(
                f"{_cell_label(row, 'author')} must be Ayo Adetayo"
            )
        _date_cell(row, "authored_on")
        rights_basis = _choice_cell(row, "rights_basis", choices=_RIGHTS_BASES)
        if _controlled_cell(row, "release_allowed") != "yes":
            raise DatasetFormatError(
                f"{_cell_label(row, 'release_allowed')} must be yes"
            )
        if _controlled_cell(row, "privacy_reviewed") != "yes":
            raise DatasetFormatError(
                f"{_cell_label(row, 'privacy_reviewed')} must be yes"
            )
        derived_from = _id_cell(row, "derived_from")
        if rights_basis == "newly_authored" and derived_from != record_id:
            raise DatasetFormatError(
                f"{_cell_label(row, 'derived_from')} must equal record_id for "
                "newly authored notes"
            )
        _parse_tags(row, "predicate_families")
        _free_text_cell(row, "source_reference")
        _controlled_cell(row, "license_id")
        _optional_free_text_cell(row, "notes")
        expected_ready = "ready"
        if _controlled_cell(row, "ready") != expected_ready:
            raise DatasetFormatError(
                f"{_cell_label(row, 'ready')} conflicts with recomputed readiness"
            )
        if record_id in seen_records:
            raise DatasetFormatError(f"duplicate Authoring record_id: {record_id}")
        if document_id in seen_documents:
            raise DatasetFormatError(f"duplicate Authoring document_id: {document_id}")
        if text in seen_notes:
            raise DatasetFormatError("Authoring raw_note values must be unique")
        seen_records.add(record_id)
        seen_documents.add(document_id)
        seen_notes.add(text)
        scenarios[scenario] += 1
        if not tokenize_pilot_text(text):
            raise DatasetFormatError(
                f"{_cell_label(row, 'raw_note')} contains no tokens"
            )
        records.append(
            _AuthoringRecord(
                row_number=row.row_number,
                record_id=record_id,
                document_id=document_id,
                text=text,
                scenario=scenario,
                values=row.values,
            )
        )
    expected_scenarios = {scenario: 4 for scenario in _SCENARIOS}
    if dict(scenarios) != expected_scenarios:
        raise DatasetFormatError(
            "Authoring must contain four records per scenario; "
            f"found={dict(sorted(scenarios.items()))}"
        )
    missing_phenomena = _REQUIRED_PHENOMENA - all_phenomena
    if missing_phenomena:
        raise DatasetFormatError(
            "Authoring target_phenomena does not cover the protocol; "
            f"missing={sorted(missing_phenomena)}"
        )
    fingerprint = _json_fingerprint(
        sorted(
            (_semantic_row(row) for row in rows),
            key=lambda item: item["values"]["record_id"],
        )
    )
    if fingerprint != PILOT_AUTHORING_FINGERPRINT:
        raise DatasetFormatError(
            "Authoring rows differ from the accepted candidate-pilot source set; "
            f"expected={PILOT_AUTHORING_FINGERPRINT}, found={fingerprint}"
        )
    return tuple(records), fingerprint


def _validate_contract(rows: Sequence[_SheetRow]) -> None:
    values: dict[str, str] = {}
    for row in rows:
        key = _controlled_cell(row, "key")
        value = _controlled_cell(row, "value")
        if key in values:
            raise DatasetFormatError(f"duplicate Contract key: {key}")
        values[key] = value
    if values != _CONTRACT_VALUES:
        raise DatasetFormatError(
            "Contract values differ from the supported workbook contract; "
            f"missing={sorted(set(_CONTRACT_VALUES) - set(values))}, "
            f"extra={sorted(set(values) - set(_CONTRACT_VALUES))}, "
            "changed="
            f"{sorted(key for key in set(values) & set(_CONTRACT_VALUES) if values[key] != _CONTRACT_VALUES[key])}"
        )


def _annotation_rows(snapshot: _WorkbookSnapshot) -> tuple[_SheetRow, ...]:
    return tuple(
        row
        for sheet_name in (
            "Annotation",
            "Qualifier Evidence",
            "Questions",
            "Answer Spans",
        )
        for row in snapshot.rows[sheet_name]
    )


def _active_review_rows(rows: Sequence[_SheetRow]) -> tuple[_SheetRow, ...]:
    active_fields = {
        "record_id",
        "candidate_id",
        "field",
        "pass_1_value",
        "pass_2_value",
        "resolution",
        "guide_change",
        "reviewed_on",
    }
    return tuple(
        row
        for row in rows
        if (
            any(not _is_blank(row.values[field]) for field in active_fields)
            or row.values["excluded"] not in {None, "", "no"}
            or row.values["reviewed_by"] not in {None, "", "Ayo Adetayo"}
        )
    )


def _single_annotation_group(
    rows: Sequence[_SheetRow],
) -> tuple[str, str]:
    groups = {
        (
            _id_cell(row, "annotator_id"),
            _pass_identifier(
                row.values["pass_id"],
                label=_cell_label(row, "pass_id"),
            ),
        )
        for row in rows
    }
    if len(groups) != 1:
        raise DatasetFormatError(
            "one workbook must contain exactly one annotator/pass group; "
            f"found={sorted(groups)}"
        )
    return next(iter(groups))


def _require_group(
    row: _SheetRow,
    *,
    annotator_id: str,
    pass_id: str,
) -> None:
    actual = (
        _id_cell(row, "annotator_id"),
        _pass_identifier(
            row.values["pass_id"],
            label=_cell_label(row, "pass_id"),
        ),
    )
    if actual != (annotator_id, pass_id):
        raise DatasetFormatError(
            f"{_row_label(row)} belongs to a different annotator/pass group"
        )


def _record_id(
    row: _SheetRow,
    record_by_id: Mapping[str, _AuthoringRecord],
) -> str:
    record_id = _id_cell(row, "record_id")
    if record_id not in record_by_id:
        raise DatasetFormatError(
            f"{_cell_label(row, 'record_id')} does not match Authoring: {record_id}"
        )
    return record_id


def _candidate_key(row: _SheetRow) -> tuple[str, str]:
    return _id_cell(row, "record_id"), _id_cell(row, "candidate_id")


def _grounded_span(
    *,
    row: _SheetRow,
    text: str,
    tokens: Sequence[_SourceToken],
    text_field: str,
    start_field: str,
    end_field: str,
) -> _GroundedSpan:
    surface = _free_text_cell(row, text_field)
    start = _nonnegative_integer_cell(row, start_field)
    end = _nonnegative_integer_cell(row, end_field)
    if end <= start or end > len(text):
        raise DatasetFormatError(
            f"{_row_label(row)} has invalid {text_field} range [{start}, {end})"
        )
    if text[start:end] != surface:
        raise DatasetFormatError(
            f"{_cell_label(row, text_field)} does not match Authoring raw_note "
            "at its declared offsets"
        )
    token_start = next((token.index for token in tokens if token.start == start), None)
    token_end_index = next((token.index for token in tokens if token.end == end), None)
    if token_start is None or token_end_index is None or token_end_index < token_start:
        raise DatasetFormatError(
            f"{_cell_label(row, text_field)} must align to complete "
            f"{PILOT_TOKENIZATION_VERSION} tokens"
        )
    return _GroundedSpan(
        text=surface,
        character_start=start,
        character_end=end,
        token_start=token_start,
        token_end=token_end_index + 1,
    )


def _normalized_row(row: _SheetRow) -> dict[str, Any]:
    return {
        "sheet": row.sheet_name,
        "sheet_row": row.row_number,
        "values": dict(row.values),
    }


def _semantic_row(row: _SheetRow) -> dict[str, Any]:
    """Return row content without its presentation-only worksheet position."""

    return {
        "sheet": row.sheet_name,
        "values": dict(row.values),
    }


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _row_label(row: _SheetRow) -> str:
    return f"{row.sheet_name} row {row.row_number}"


def _cell_label(row: _SheetRow, field: str) -> str:
    return f"{_row_label(row)} {field}"


def _is_blank(value: Any) -> bool:
    return value is None or value == ""


def _require_blank(row: _SheetRow, field: str) -> None:
    if not _is_blank(row.values[field]):
        raise DatasetFormatError(f"{_cell_label(row, field)} must be blank")


def _controlled_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetFormatError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise DatasetFormatError(f"{label} cannot have surrounding whitespace")
    return value


def _controlled_cell(row: _SheetRow, field: str) -> str:
    return _controlled_text(row.values[field], label=_cell_label(row, field))


def _optional_controlled_cell(row: _SheetRow, field: str) -> str | None:
    value = row.values[field]
    if _is_blank(value):
        return None
    return _controlled_text(value, label=_cell_label(row, field))


def _id_cell(row: _SheetRow, field: str) -> str:
    return _identifier(row.values[field], label=_cell_label(row, field))


def _identifier(value: Any, *, label: str) -> str:
    text = _controlled_text(value, label=label)
    if not _ID_PATTERN.fullmatch(text):
        raise DatasetFormatError(
            f"{label} must match lowercase identifier pattern {_ID_PATTERN.pattern}"
        )
    return text


def _pass_identifier(value: Any, *, label: str) -> str:
    text = _identifier(value, label=label)
    if text not in {"pass-1", "pass-2"}:
        raise DatasetFormatError(f"{label} must be pass-1 or pass-2")
    return text


def _free_text_cell(row: _SheetRow, field: str) -> str:
    value = row.values[field]
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(f"{_cell_label(row, field)} cannot be blank")
    return value


def _optional_free_text_cell(row: _SheetRow, field: str) -> str | None:
    value = row.values[field]
    if _is_blank(value):
        return None
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(
            f"{_cell_label(row, field)} must be meaningful text"
        )
    return value


def _choice_cell(row: _SheetRow, field: str, *, choices: set[str]) -> str:
    value = _controlled_cell(row, field)
    if value not in choices:
        raise DatasetFormatError(
            f"{_cell_label(row, field)} must be one of: "
            f"{', '.join(sorted(choices))}"
        )
    return value


def _boolean_cell(row: _SheetRow, field: str) -> bool:
    value = row.values[field]
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise DatasetFormatError(f"{_cell_label(row, field)} must be true or false")


def _nonnegative_integer_cell(row: _SheetRow, field: str) -> int:
    value = row.values[field]
    if isinstance(value, bool):
        raise DatasetFormatError(
            f"{_cell_label(row, field)} must be a non-negative integer"
        )
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isdigit():
        result = int(value)
    else:
        raise DatasetFormatError(
            f"{_cell_label(row, field)} must be a non-negative integer"
        )
    if result < 0:
        raise DatasetFormatError(
            f"{_cell_label(row, field)} must be a non-negative integer"
        )
    return result


def _slot_cell(row: _SheetRow, field: str) -> str:
    return _controlled_cell(row, field)


def _parse_tags(row: _SheetRow, field: str) -> tuple[str, ...]:
    raw = _controlled_cell(row, field)
    values = tuple(raw.split(";"))
    if any(not _TAG_PATTERN.fullmatch(value) for value in values):
        raise DatasetFormatError(
            f"{_cell_label(row, field)} must contain semicolon-separated "
            "lowercase identifiers"
        )
    if len(values) != len(set(values)):
        raise DatasetFormatError(f"{_cell_label(row, field)} contains duplicate tags")
    return values


def _date_cell(row: _SheetRow, field: str) -> str:
    value = row.values[field]
    if isinstance(value, bool):
        raise DatasetFormatError(f"{_cell_label(row, field)} must be a date")
    if isinstance(value, int):
        if value < 1 or value > 2_958_465:
            raise DatasetFormatError(f"{_cell_label(row, field)} has invalid date serial")
        try:
            return (date(1899, 12, 30) + timedelta(days=value)).isoformat()
        except OverflowError as error:
            raise DatasetFormatError(
                f"{_cell_label(row, field)} has invalid date serial"
            ) from error
    text = _controlled_cell(row, field)
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as error:
        raise DatasetFormatError(
            f"{_cell_label(row, field)} must use yyyy-mm-dd"
        ) from error


def _read_snapshot(path: str | Path) -> _WorkbookSnapshot:
    source = Path(path)
    initial_digest = sha256_file(source)
    try:
        with zipfile.ZipFile(source) as archive:
            _validate_archive(archive)
            shared_strings = _read_shared_strings(archive)
            sheet_parts = _sheet_parts(archive)
            rows: dict[str, tuple[_SheetRow, ...]] = {}
            for sheet_name, (table_name, headers) in _SHEET_TABLES.items():
                raw_rows, table = _read_sheet(
                    archive,
                    sheet_parts,
                    sheet_name,
                    shared_strings=shared_strings,
                    allowed_formula_columns=(
                        {headers.index("ready")} if sheet_name == "Authoring" else set()
                    ),
                )
                table_last_row = _validate_table(
                    archive,
                    table,
                    expected_name=table_name,
                    expected_headers=headers,
                    sheet_name=sheet_name,
                )
                rows[sheet_name] = _rows_with_headers(
                    raw_rows,
                    expected_headers=headers,
                    sheet_name=sheet_name,
                    table_last_row=table_last_row,
                )
    except DatasetFormatError:
        raise
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise DatasetFormatError(f"invalid pilot workbook {source}: {error}") from error
    final_digest = sha256_file(source)
    if final_digest != initial_digest:
        raise RuntimeError(f"pilot workbook changed while it was read: {source}")
    return _WorkbookSnapshot(workbook_sha256=initial_digest, rows=rows)


def _validate_archive(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if len(infos) > _MAX_ARCHIVE_FILES:
        raise DatasetFormatError("pilot workbook contains too many archive entries")
    total = 0
    seen_names: set[str] = set()
    for info in infos:
        if info.filename in seen_names:
            raise DatasetFormatError(
                f"pilot workbook repeats archive entry {info.filename}"
            )
        seen_names.add(info.filename)
        normalized = posixpath.normpath(info.filename)
        if (
            info.filename.startswith("/")
            or normalized == ".."
            or normalized.startswith("../")
        ):
            raise DatasetFormatError("pilot workbook contains an unsafe archive path")
        if info.flag_bits & 0x1:
            raise DatasetFormatError("encrypted pilot workbooks are not supported")
        lowered = info.filename.lower()
        if "vbaproject" in lowered or "/externallinks/" in lowered:
            raise DatasetFormatError(
                "pilot workbook cannot contain macros or external links"
            )
        total += info.file_size
        if info.file_size > _MAX_XML_BYTES:
            raise DatasetFormatError("pilot workbook contains an oversized part")
    if total > _MAX_UNCOMPRESSED_BYTES:
        raise DatasetFormatError("pilot workbook expands beyond the size limit")
    for info in infos:
        if not info.filename.lower().endswith(".rels"):
            continue
        root = _read_xml(archive, info.filename)
        for relation in root.findall(
            f"{{{_RELATIONSHIP_NAMESPACE}}}Relationship"
        ):
            if relation.attrib.get("TargetMode") == "External":
                raise DatasetFormatError(
                    "pilot workbook contains an external relationship"
                )


def _sheet_parts(archive: zipfile.ZipFile) -> Mapping[str, str]:
    workbook = _read_xml(archive, "xl/workbook.xml")
    relationships = _relationship_targets(
        archive,
        "xl/_rels/workbook.xml.rels",
        parent="xl/workbook.xml",
    )
    parts: dict[str, str] = {}
    for sheet in workbook.findall(f".//{{{_XML_NAMESPACE}}}sheet"):
        name = sheet.attrib.get("name")
        relation_id = sheet.attrib.get(
            f"{{{_OFFICE_RELATIONSHIP_NAMESPACE}}}id"
        )
        if name and relation_id:
            if name in parts:
                raise DatasetFormatError(
                    f"pilot workbook repeats worksheet name {name}"
                )
            try:
                parts[name] = relationships[relation_id]
            except KeyError as error:
                raise DatasetFormatError(
                    f"worksheet {name} has no package relationship"
                ) from error
    required = set(_SHEET_TABLES) | {"Instructions", "Lists"}
    actual = set(parts)
    if actual != required:
        raise DatasetFormatError(
            "pilot workbook sheet names differ from the contract; "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )
    return parts


def _relationship_targets(
    archive: zipfile.ZipFile,
    relationship_part: str,
    *,
    parent: str,
) -> Mapping[str, str]:
    root = _read_xml(archive, relationship_part)
    targets: dict[str, str] = {}
    for relation in root.findall(f"{{{_RELATIONSHIP_NAMESPACE}}}Relationship"):
        if relation.attrib.get("TargetMode") == "External":
            raise DatasetFormatError("pilot workbook contains an external relationship")
        relation_id = relation.attrib.get("Id")
        target = relation.attrib.get("Target")
        if relation_id and target:
            if relation_id in targets:
                raise DatasetFormatError(
                    f"relationship part {relationship_part} repeats {relation_id}"
                )
            targets[relation_id] = _resolve_part(parent, target)
    return targets


def _resolve_part(parent: str, target: str) -> str:
    if target.startswith("/"):
        resolved = posixpath.normpath(target.lstrip("/"))
    else:
        resolved = posixpath.normpath(
            posixpath.join(posixpath.dirname(parent), target)
        )
    if resolved == ".." or resolved.startswith("../"):
        raise DatasetFormatError("pilot workbook relationship leaves the archive")
    return resolved


def _read_shared_strings(archive: zipfile.ZipFile) -> tuple[str, ...]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return ()
    root = _read_xml(archive, "xl/sharedStrings.xml")
    return tuple(
        "".join(text.text or "" for text in item.iter(f"{{{_XML_NAMESPACE}}}t"))
        for item in root.findall(f"{{{_XML_NAMESPACE}}}si")
    )


def _read_sheet(
    archive: zipfile.ZipFile,
    parts: Mapping[str, str],
    sheet_name: str,
    *,
    shared_strings: Sequence[str],
    allowed_formula_columns: set[int],
) -> tuple[
    tuple[tuple[int, Mapping[int, Any]], ...],
    tuple[str, str],
]:
    try:
        part = parts[sheet_name]
    except KeyError as error:
        raise DatasetFormatError(f"pilot workbook is missing {sheet_name} sheet") from error
    root = _read_xml(archive, part)
    result: list[tuple[int, Mapping[int, Any]]] = []
    seen_rows: set[int] = set()
    for row in root.findall(f".//{{{_XML_NAMESPACE}}}sheetData/{{{_XML_NAMESPACE}}}row"):
        raw_number = row.attrib.get("r")
        if raw_number is None or not raw_number.isdigit() or int(raw_number) < 1:
            raise DatasetFormatError(f"{sheet_name} contains an invalid row number")
        row_number = int(raw_number)
        if row_number in seen_rows:
            raise DatasetFormatError(f"{sheet_name} row {row_number} is duplicated")
        seen_rows.add(row_number)
        cells: dict[int, Any] = {}
        for cell in row.findall(f"{{{_XML_NAMESPACE}}}c"):
            reference = cell.attrib.get("r", "")
            match = _CELL_REFERENCE.fullmatch(reference)
            if match is None or int(match.group(2)) != row_number:
                raise DatasetFormatError(
                    f"{sheet_name} row {row_number} contains invalid cell {reference!r}"
                )
            column = _column_index(match.group(1))
            if column in cells:
                raise DatasetFormatError(
                    f"{sheet_name} row {row_number} repeats column {match.group(1)}"
                )
            if (
                cell.find(f"{{{_XML_NAMESPACE}}}f") is not None
                and column not in allowed_formula_columns
            ):
                raise DatasetFormatError(
                    f"{sheet_name}!{reference} contains a forbidden formula"
                )
            cells[column] = _cell_value(
                cell,
                shared_strings=shared_strings,
                label=f"{sheet_name}!{reference}",
            )
        result.append((row_number, cells))

    table_parts = root.findall(
        f".//{{{_XML_NAMESPACE}}}tableParts/{{{_XML_NAMESPACE}}}tablePart"
    )
    if len(table_parts) != 1:
        raise DatasetFormatError(
            f"{sheet_name} must contain exactly one declared table"
        )
    relation_id = table_parts[0].attrib.get(
        f"{{{_OFFICE_RELATIONSHIP_NAMESPACE}}}id"
    )
    if relation_id is None:
        raise DatasetFormatError(f"{sheet_name} table has no relationship")
    sheet_relationship_part = posixpath.join(
        posixpath.dirname(part),
        "_rels",
        f"{posixpath.basename(part)}.rels",
    )
    relationships = _relationship_targets(
        archive,
        sheet_relationship_part,
        parent=part,
    )
    try:
        table_part = relationships[relation_id]
    except KeyError as error:
        raise DatasetFormatError(f"{sheet_name} table relationship is missing") from error
    return tuple(sorted(result)), (part, table_part)


def _validate_table(
    archive: zipfile.ZipFile,
    table_reference: tuple[str, str],
    *,
    expected_name: str,
    expected_headers: Sequence[str],
    sheet_name: str,
) -> int:
    _, table_part = table_reference
    root = _read_xml(archive, table_part)
    if root.attrib.get("name") != expected_name:
        raise DatasetFormatError(
            f"{sheet_name} table must be named {expected_name}"
        )
    if root.attrib.get("displayName") != expected_name:
        raise DatasetFormatError(
            f"{sheet_name} table displayName must be {expected_name}"
        )
    reference = root.attrib.get("ref", "")
    match = _TABLE_REFERENCE.fullmatch(reference)
    expected_last_column = _column_letters(len(expected_headers) - 1)
    if (
        match is None
        or match.group(1) != expected_last_column
        or int(match.group(2)) < 2
    ):
        raise DatasetFormatError(
            f"{sheet_name} table range must start at A1 and end in "
            f"column {expected_last_column}; found {reference!r}"
        )
    columns = root.findall(
        f".//{{{_XML_NAMESPACE}}}tableColumns/{{{_XML_NAMESPACE}}}tableColumn"
    )
    names = tuple(column.attrib.get("name") for column in columns)
    if names != tuple(expected_headers):
        raise DatasetFormatError(
            f"{sheet_name} table columns differ from the contract; "
            f"expected={list(expected_headers)}, found={list(names)}"
        )
    return int(match.group(2))


def _read_xml(archive: zipfile.ZipFile, part: str) -> ElementTree.Element:
    try:
        data = archive.read(part)
    except KeyError as error:
        raise DatasetFormatError(f"pilot workbook is missing package part {part}") from error
    if len(data) > _MAX_XML_BYTES:
        raise DatasetFormatError(f"pilot workbook part is oversized: {part}")
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise DatasetFormatError(f"pilot workbook part contains a forbidden DTD: {part}")
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as error:
        raise DatasetFormatError(f"pilot workbook part is invalid XML: {part}") from error


def _cell_value(
    cell: ElementTree.Element,
    *,
    shared_strings: Sequence[str],
    label: str,
) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{_XML_NAMESPACE}}}is")
        if inline is None:
            return None
        return "".join(
            item.text or "" for item in inline.iter(f"{{{_XML_NAMESPACE}}}t")
        )
    value_node = cell.find(f"{{{_XML_NAMESPACE}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type in {"str", "d"}:
        return raw
    if cell_type == "s":
        try:
            index = int(raw)
            if index < 0:
                raise ValueError
            return shared_strings[index]
        except (ValueError, IndexError) as error:
            raise DatasetFormatError(f"{label} has an invalid shared string") from error
    if cell_type == "b":
        if raw not in {"0", "1"}:
            raise DatasetFormatError(f"{label} has an invalid boolean")
        return raw == "1"
    if cell_type == "e":
        raise DatasetFormatError(f"{label} contains spreadsheet error {raw}")
    try:
        number = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(number):
        raise DatasetFormatError(f"{label} contains a non-finite number")
    return int(number) if number.is_integer() else number


def _rows_with_headers(
    rows: Sequence[tuple[int, Mapping[int, Any]]],
    *,
    expected_headers: Sequence[str],
    sheet_name: str,
    table_last_row: int,
) -> tuple[_SheetRow, ...]:
    by_number = dict(rows)
    header_cells = by_number.get(1)
    if header_cells is None:
        raise DatasetFormatError(f"{sheet_name} is missing its header row")
    headers = tuple(header_cells.get(index) for index in range(len(expected_headers)))
    extras = [
        value
        for index, value in header_cells.items()
        if index >= len(expected_headers) and not _is_blank(value)
    ]
    if headers != tuple(expected_headers) or extras:
        raise DatasetFormatError(
            f"{sheet_name} headers differ from {PILOT_WORKBOOK_CONTRACT_VERSION}; "
            f"expected={list(expected_headers)}, found={list(headers) + extras}"
        )
    result = []
    for row_number, cells in rows:
        if row_number == 1:
            continue
        values = {
            header: cells.get(index)
            for index, header in enumerate(expected_headers)
        }
        extras = [
            value
            for index, value in cells.items()
            if index >= len(expected_headers) and not _is_blank(value)
        ]
        if extras:
            raise DatasetFormatError(
                f"{sheet_name} row {row_number} contains values beyond the schema"
            )
        if row_number > table_last_row:
            raise DatasetFormatError(
                f"{sheet_name} row {row_number} contains data outside its table"
            )
        if all(_is_blank(value) for value in values.values()):
            continue
        result.append(
            _SheetRow(
                sheet_name=sheet_name,
                row_number=row_number,
                values=values,
            )
        )
    return tuple(result)


def _column_index(letters: str) -> int:
    value = 0
    for character in letters:
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _column_letters(index: int) -> str:
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result
