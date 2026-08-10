from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree
import zipfile

import semantic_action_extractor.challenge_workbook as workbook_module
from semantic_action_extractor.challenge_workbook import (
    PILOT_AUTHORING_FINGERPRINT,
    PILOT_RECORD_QUARANTINE,
    PILOT_TOKENIZATION_VERSION,
    PILOT_WORKBOOK_ADAPTER_VERSION,
    PILOT_WORKBOOK_CONTRACT_VERSION,
    _SHEET_TABLES,
    _SheetRow,
    _WorkbookSnapshot,
    _read_snapshot,
    convert_pilot_workbook,
    tokenize_pilot_text,
    validate_pilot_workbook,
)
from semantic_action_extractor.data_cli import main as data_main
from semantic_action_extractor.datasets.common import DatasetFormatError
from semantic_action_extractor.evaluation.bundle import load_evaluation_bundle
from semantic_action_extractor.evaluation.scorers import (
    PRIMARY_END_TO_END_V1,
    score_corpora,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_WORKBOOK = PROJECT_ROOT / "docs/challenge_set/pilot_worksheets.xlsx"

_PREDICATES = {
    "ops-pilot-001": ("emailed", "verbal", "email", True, None),
    "ops-pilot-002": ("approval", "nominal", "approval", True, "approve"),
    "ops-pilot-003": ("refund", "verbal", "refund", True, None),
    "ops-pilot-004": ("reopen", "verbal", "reopen", True, None),
    "ops-pilot-005": ("restarted", "verbal", "restart", True, None),
    "ops-pilot-006": ("detected", "verbal", "detect", True, None),
    "ops-pilot-007": ("fails", "verbal", "fail", True, None),
    "ops-pilot-008": ("reported", "verbal", "report", True, None),
    "ops-pilot-009": ("approval", "nominal", "approval", True, "approve"),
    "ops-pilot-010": ("authorized", "verbal", "authorize", True, None),
    "ops-pilot-011": ("renew", "verbal", "renew", True, None),
    "ops-pilot-012": ("approved", "verbal", "approve", True, None),
    "ops-pilot-013": ("packed", "verbal", "pack", True, None),
    "ops-pilot-014": ("assigned", "verbal", "assign", True, None),
    "ops-pilot-015": ("deliver", "verbal", "deliver", True, None),
    "ops-pilot-016": ("delivery", "nominal", "delivery", False, "deliver"),
    "ops-pilot-017": ("approval", "nominal", "approval", True, "approve"),
    "ops-pilot-018": ("revise", "verbal", "revise", True, None),
    "ops-pilot-019": ("postpone", "verbal", "postpone", True, None),
    "ops-pilot-020": ("schedule", "verbal", "schedule", True, None),
}

_ANSWERS = {
    "ops-pilot-001": ("The support agent",),
    "ops-pilot-002": ("The supervisor's",),
    "ops-pilot-003": ("The agent",),
    "ops-pilot-004": ("Support",),
    "ops-pilot-005": ("the on-call engineer",),
    "ops-pilot-006": ("latency", "a ticket"),
    "ops-pilot-007": ("the engineer",),
    "ops-pilot-008": ("The team lead",),
    "ops-pilot-009": ("The billing team's",),
    "ops-pilot-010": ("the reviewer",),
    "ops-pilot-011": ("Procurement",),
    "ops-pilot-012": ("The reviewer",),
    "ops-pilot-013": ("warehouse staff",),
    "ops-pilot-014": ("The dispatcher",),
    "ops-pilot-015": ("The carrier",),
    "ops-pilot-017": ("the coordinator",),
    "ops-pilot-018": ("the team",),
    "ops-pilot-019": ("the committee",),
    "ops-pilot-020": ("The administrator",),
}


def _row(sheet_name: str, row_number: int, **values: object) -> _SheetRow:
    headers = _SHEET_TABLES[sheet_name][1]
    payload = {header: None for header in headers}
    payload.update(values)
    return _SheetRow(sheet_name=sheet_name, row_number=row_number, values=payload)


def _span(text: str, surface: str) -> tuple[str, int, int]:
    start = text.index(surface)
    return surface, start, start + len(surface)


def _completed_snapshot(*, quarantine_first_record: bool = False) -> _WorkbookSnapshot:
    base = _read_snapshot(PILOT_WORKBOOK)
    authoring = {row.values["record_id"]: row for row in base.rows["Authoring"]}
    candidates = []
    qualifiers = []
    questions = []
    answers = []

    for index, source_row in enumerate(base.rows["Authoring"], start=1):
        record_id = source_row.values["record_id"]
        text = source_row.values["raw_note"]
        surface, predicate_type, lemma, eventive, related = _PREDICATES[record_id]
        _, predicate_start, predicate_end = _span(text, surface)
        candidate_id = f"c-{index:03d}"
        decision = "include"
        ambiguity = None
        rationale = "fixture candidate"
        qualifier_assessed = eventive
        if quarantine_first_record and record_id == "ops-pilot-001":
            decision = "exclude"
            lemma = None
            eventive = None
            related = None
            qualifier_assessed = False
            ambiguity = "guide does not resolve this candidate"
            rationale = "exclude rather than invent a label"
        candidates.append(
            _row(
                "Annotation",
                index + 1,
                record_id=record_id,
                annotator_id="ayo-adetayo",
                pass_id="pass-1",
                candidate_id=candidate_id,
                predicate_text=surface,
                predicate_start=predicate_start,
                predicate_end=predicate_end,
                predicate_type=predicate_type,
                lemma=lemma,
                eventive=(None if eventive is None else str(eventive).lower()),
                related_verbal_form=related,
                qualifier_assessed=str(qualifier_assessed).lower(),
                decision=decision,
                ambiguity_or_exclusion=ambiguity,
                rationale=rationale,
            )
        )
        if eventive is not True:
            continue
        question_id = "q-role"
        questions.append(
            _row(
                "Questions",
                len(questions) + 2,
                record_id=record_id,
                annotator_id="ayo-adetayo",
                pass_id="pass-1",
                candidate_id=candidate_id,
                question_id=question_id,
                question_surface=f"What participates in the {lemma} event?",
                wh="what",
                aux="_",
                subj="_",
                verb="past",
                obj="something",
                prep="_",
                obj2="_",
                is_passive="false",
                is_negated="false",
                rationale="fixture role question",
            )
        )
        for span_order, answer_surface in enumerate(_ANSWERS[record_id]):
            _, answer_start, answer_end = _span(text, answer_surface)
            answers.append(
                _row(
                    "Answer Spans",
                    len(answers) + 2,
                    record_id=record_id,
                    annotator_id="ayo-adetayo",
                    pass_id="pass-1",
                    candidate_id=candidate_id,
                    question_id=question_id,
                    alternative_id="a-01",
                    span_order=span_order,
                    answer_text=answer_surface,
                    answer_start=answer_start,
                    answer_end=answer_end,
                    rationale="fixture answer",
                )
            )

    qualifier_specs = {
        "ops-pilot-003": {"negated": ("not",)},
        "ops-pilot-004": {"possible": ("may",)},
        "ops-pilot-011": {"planned": ("plans", "to")},
    }
    candidate_ids = {
        row.values["record_id"]: row.values["candidate_id"] for row in candidates
    }
    for record_id, grouped in qualifier_specs.items():
        text = authoring[record_id].values["raw_note"]
        for kind, surfaces in grouped.items():
            for span_order, surface in enumerate(surfaces):
                _, start, end = _span(text, surface)
                qualifiers.append(
                    _row(
                        "Qualifier Evidence",
                        len(qualifiers) + 2,
                        record_id=record_id,
                        annotator_id="ayo-adetayo",
                        pass_id="pass-1",
                        candidate_id=candidate_ids[record_id],
                        qualifier_kind=kind,
                        span_order=span_order,
                        evidence_text=surface,
                        evidence_start=start,
                        evidence_end=end,
                        rationale="fixture qualifier",
                    )
                )

    if quarantine_first_record:
        record_id = "ops-pilot-001"
        text = authoring[record_id].values["raw_note"]
        surface, start, end = _span(text, "Monday")
        candidates.append(
            _row(
                "Annotation",
                len(candidates) + 2,
                record_id=record_id,
                annotator_id="ayo-adetayo",
                pass_id="pass-1",
                candidate_id="c-extra",
                predicate_text=surface,
                predicate_start=start,
                predicate_end=end,
                predicate_type="nominal",
                lemma="monday",
                eventive="true",
                related_verbal_form=None,
                qualifier_assessed="true",
                decision="include",
                ambiguity_or_exclusion=None,
                rationale="fixture included candidate in quarantined record",
            )
        )
        questions.append(
            _row(
                "Questions",
                len(questions) + 2,
                record_id=record_id,
                annotator_id="ayo-adetayo",
                pass_id="pass-1",
                candidate_id="c-extra",
                question_id="q-extra",
                question_surface="What is temporally grounded?",
                wh="what",
                aux="_",
                subj="_",
                verb="past",
                obj="something",
                prep="_",
                obj2="_",
                is_passive="false",
                is_negated="false",
                rationale="fixture quarantine question",
            )
        )
        answer_surface, answer_start, answer_end = _span(text, "Monday")
        answers.append(
            _row(
                "Answer Spans",
                len(answers) + 2,
                record_id=record_id,
                annotator_id="ayo-adetayo",
                pass_id="pass-1",
                candidate_id="c-extra",
                question_id="q-extra",
                alternative_id="a-extra",
                span_order=0,
                answer_text=answer_surface,
                answer_start=answer_start,
                answer_end=answer_end,
                rationale="fixture quarantine answer",
            )
        )

    rows = dict(base.rows)
    rows.update(
        {
            "Annotation": tuple(candidates),
            "Qualifier Evidence": tuple(qualifiers),
            "Questions": tuple(questions),
            "Answer Spans": tuple(answers),
        }
    )
    return _WorkbookSnapshot(workbook_sha256="fixture-workbook-sha256", rows=rows)


def _replace_sheet(
    snapshot: _WorkbookSnapshot,
    sheet_name: str,
    rows: tuple[_SheetRow, ...],
) -> _WorkbookSnapshot:
    updated = dict(snapshot.rows)
    updated[sheet_name] = rows
    return _WorkbookSnapshot(
        workbook_sha256=snapshot.workbook_sha256,
        rows=updated,
    )


def _rewrite_package_member(
    source: Path,
    destination: Path,
    member_name: str,
    payload: bytes,
) -> None:
    with zipfile.ZipFile(source) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(destination, "w") as archive:
        for info, data in entries:
            archive.writestr(info, payload if info.filename == member_name else data)


def _column_letters(index: int) -> str:
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _worksheet_with_rows(
    payload: bytes,
    *,
    sheet_name: str,
    rows: tuple[_SheetRow, ...],
) -> bytes:
    namespace = workbook_module._XML_NAMESPACE
    root = ElementTree.fromstring(payload)
    sheet_data = root.find(f"{{{namespace}}}sheetData")
    for child in list(sheet_data):
        sheet_data.remove(child)

    headers = _SHEET_TABLES[sheet_name][1]

    def add_row(row_number: int, values: dict[str, object]) -> None:
        row = ElementTree.SubElement(
            sheet_data,
            f"{{{namespace}}}row",
            {"r": str(row_number)},
        )
        for index, header in enumerate(headers):
            value = values.get(header)
            if value is None:
                continue
            reference = f"{_column_letters(index)}{row_number}"
            if isinstance(value, bool):
                cell = ElementTree.SubElement(
                    row,
                    f"{{{namespace}}}c",
                    {"r": reference, "t": "b"},
                )
                ElementTree.SubElement(cell, f"{{{namespace}}}v").text = (
                    "1" if value else "0"
                )
            elif isinstance(value, int):
                cell = ElementTree.SubElement(
                    row,
                    f"{{{namespace}}}c",
                    {"r": reference, "t": "n"},
                )
                ElementTree.SubElement(cell, f"{{{namespace}}}v").text = str(value)
            else:
                cell = ElementTree.SubElement(
                    row,
                    f"{{{namespace}}}c",
                    {"r": reference, "t": "inlineStr"},
                )
                inline = ElementTree.SubElement(cell, f"{{{namespace}}}is")
                ElementTree.SubElement(inline, f"{{{namespace}}}t").text = str(value)

    add_row(1, {header: header for header in headers})
    for row_number, source_row in enumerate(rows, start=2):
        add_row(row_number, dict(source_row.values))
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _write_populated_workbook(
    destination: Path,
    snapshot: _WorkbookSnapshot,
) -> None:
    with zipfile.ZipFile(PILOT_WORKBOOK) as archive:
        parts = workbook_module._sheet_parts(archive)
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    replacements = {
        parts[sheet_name]: _worksheet_with_rows(
            next(data for info, data in entries if info.filename == parts[sheet_name]),
            sheet_name=sheet_name,
            rows=snapshot.rows[sheet_name],
        )
        for sheet_name in (
            "Annotation",
            "Qualifier Evidence",
            "Questions",
            "Answer Spans",
        )
    }
    with zipfile.ZipFile(destination, "w") as archive:
        for info, data in entries:
            archive.writestr(info, replacements.get(info.filename, data))


class ChallengeWorkbookTests(unittest.TestCase):
    def test_checked_in_workbook_is_valid_pending_template(self) -> None:
        report = validate_pilot_workbook(PILOT_WORKBOOK)

        self.assertEqual(report.authoring_record_count, 20)
        self.assertEqual(report.authoring_fingerprint, PILOT_AUTHORING_FINGERPRINT)
        self.assertEqual(report.candidate_row_count, 0)
        self.assertEqual(report.active_review_row_count, 0)
        self.assertFalse(report.to_dict()["ready_for_scoring"])
        with self.assertRaisesRegex(DatasetFormatError, "annotation is pending"):
            convert_pilot_workbook(
                PILOT_WORKBOOK,
                annotator_id="ayo-adetayo",
                pass_id="pass-1",
            )

    def test_completed_pass_converts_and_round_trips(self) -> None:
        snapshot = _completed_snapshot()
        with patch.object(workbook_module, "_read_snapshot", return_value=snapshot):
            report = validate_pilot_workbook("fixture.xlsx")
            conversion = convert_pilot_workbook(
                "fixture.xlsx",
                annotator_id="ayo-adetayo",
                pass_id="pass-1",
            )

        self.assertTrue(report.to_dict()["ready_for_scoring"])
        self.assertEqual(conversion.candidate_count, 20)
        self.assertEqual(conversion.annotated_qa_pair_count, 19)
        self.assertEqual(conversion.qa_pair_count, 19)
        self.assertEqual(len(conversion.bundle.corpus.predicates), 20)
        metadata = conversion.bundle.metadata
        self.assertEqual(metadata["annotation_status"], "single-annotator")
        self.assertEqual(metadata["evaluation_use"], "guide-development-only")
        self.assertTrue(metadata["not_for_model_selection"])
        self.assertEqual(metadata["adapter_version"], PILOT_WORKBOOK_ADAPTER_VERSION)
        self.assertEqual(metadata["contract_version"], PILOT_WORKBOOK_CONTRACT_VERSION)
        self.assertEqual(metadata["tokenization_version"], PILOT_TOKENIZATION_VERSION)

        by_source = {
            predicate.key.source_id: predicate
            for predicate in conversion.bundle.corpus.predicates
        }
        self.assertEqual(by_source["ops-pilot-004"].mention_qualifiers[0].kind, "possible")
        self.assertEqual(by_source["ops-pilot-001"].mention_qualifiers, ())
        self.assertIsNone(by_source["ops-pilot-016"].mention_qualifiers)
        self.assertEqual(
            len(by_source["ops-pilot-011"].mention_qualifiers[0].evidence.token_spans),
            2,
        )
        self.assertEqual(len(by_source["ops-pilot-006"].pairs[0].argument.token_spans), 2)

        score = score_corpora(
            conversion.bundle.corpus,
            conversion.bundle.corpus,
            mode=PRIMARY_END_TO_END_V1,
            predicate_source="fixture",
            consolidation_rule=conversion.bundle.consolidation_rule,
        )
        self.assertEqual(score.labeled_arguments.f1, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot-pass-1.json"
            conversion.bundle.write(output)
            self.assertEqual(
                load_evaluation_bundle(output).to_dict(),
                conversion.bundle.to_dict(),
            )

    def test_populated_real_xlsx_validates_and_converts_end_to_end(self) -> None:
        snapshot = _completed_snapshot()
        source_answer = next(
            row
            for row in snapshot.rows["Answer Spans"]
            if row.values["record_id"] == "ops-pilot-006"
            and row.values["span_order"] == 1
        )
        second_alternative = replace(
            source_answer,
            row_number=999,
            values={
                **source_answer.values,
                "alternative_id": "a-02",
                "span_order": 0,
            },
        )
        snapshot = _replace_sheet(
            snapshot,
            "Answer Spans",
            (*snapshot.rows["Answer Spans"], second_alternative),
        )

        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "completed-pass.xlsx"
            _write_populated_workbook(workbook, snapshot)

            report = validate_pilot_workbook(workbook)
            conversion = convert_pilot_workbook(
                workbook,
                annotator_id="ayo-adetayo",
                pass_id="pass-1",
            )

        self.assertTrue(report.to_dict()["ready_for_scoring"])
        self.assertEqual(conversion.candidate_count, 20)
        self.assertEqual(conversion.qa_pair_count, 20)
        by_source = {
            predicate.key.source_id: predicate
            for predicate in conversion.bundle.corpus.predicates
        }
        self.assertEqual(len(by_source["ops-pilot-006"].pairs), 2)
        note = next(
            item
            for item in conversion.bundle.metadata["source_records"]
            if item["record_id"] == "ops-pilot-004"
        )["text"]
        possible = by_source["ops-pilot-004"].mention_qualifiers[0].evidence
        self.assertEqual(possible.character_spans, ((note.index("may"), note.index("may") + 3),))
        self.assertEqual(possible.token_spans, ((1, 2),))
        candidate = next(
            item
            for item in conversion.bundle.metadata["candidate_audit"]
            if item["record_id"] == "ops-pilot-004"
        )
        self.assertEqual(
            (candidate["predicate_character_start"], candidate["predicate_character_end"]),
            (note.index("reopen"), note.index("reopen") + len("reopen")),
        )
        self.assertEqual(
            (candidate["predicate_token_start"], candidate["predicate_token_end"]),
            (2, 3),
        )

    def test_semantic_fingerprints_survive_row_reordering(self) -> None:
        original = _completed_snapshot()
        reordered_rows = dict(original.rows)
        for sheet_name in (
            "Authoring",
            "Annotation",
            "Qualifier Evidence",
            "Questions",
            "Answer Spans",
        ):
            reordered_rows[sheet_name] = tuple(reversed(reordered_rows[sheet_name]))
        reordered = _WorkbookSnapshot(
            workbook_sha256="reordered-workbook-sha256",
            rows=reordered_rows,
        )
        with patch.object(workbook_module, "_read_snapshot", return_value=original):
            first = convert_pilot_workbook(
                "fixture.xlsx", annotator_id="ayo-adetayo", pass_id="pass-1"
            )
        with patch.object(workbook_module, "_read_snapshot", return_value=reordered):
            second = convert_pilot_workbook(
                "fixture.xlsx", annotator_id="ayo-adetayo", pass_id="pass-1"
            )

        self.assertEqual(first.authoring_fingerprint, second.authoring_fingerprint)
        self.assertEqual(first.annotation_fingerprint, second.annotation_fingerprint)
        self.assertEqual(first.bundle.corpus.by_key(), second.bundle.corpus.by_key())

    def test_record_exclusion_quarantines_every_predicate_in_source(self) -> None:
        snapshot = _completed_snapshot(quarantine_first_record=True)
        with patch.object(workbook_module, "_read_snapshot", return_value=snapshot):
            conversion = convert_pilot_workbook(
                "fixture.xlsx", annotator_id="ayo-adetayo", pass_id="pass-1"
            )

        metadata = conversion.bundle.metadata
        self.assertEqual(metadata["record_quarantine_policy"], PILOT_RECORD_QUARANTINE)
        self.assertEqual(metadata["excluded_source_ids"], ["ops-pilot-001"])
        self.assertNotIn(
            "ops-pilot-001",
            {item.key.source_id for item in conversion.bundle.corpus.predicates},
        )
        self.assertEqual(conversion.candidate_count, 21)
        self.assertEqual(conversion.annotated_qa_pair_count, 19)
        self.assertEqual(conversion.qa_pair_count, 18)
        quarantined = [
            row
            for row in metadata["candidate_audit"]
            if row["record_id"] == "ops-pilot-001"
        ]
        self.assertEqual(len(quarantined), 2)
        self.assertTrue(all(row["record_quarantined"] for row in quarantined))

    def test_rejects_a_completed_pass_with_no_scored_predicates(self) -> None:
        snapshot = _completed_snapshot()
        excluded = tuple(
            replace(
                row,
                values={
                    **row.values,
                    "lemma": None,
                    "eventive": None,
                    "related_verbal_form": None,
                    "qualifier_assessed": "false",
                    "decision": "exclude",
                    "ambiguity_or_exclusion": "fixture unresolved candidate",
                    "rationale": "fixture exclusion",
                },
            )
            for row in snapshot.rows["Annotation"]
        )
        empty = _replace_sheet(snapshot, "Annotation", excluded)
        empty = _replace_sheet(empty, "Qualifier Evidence", ())
        empty = _replace_sheet(empty, "Questions", ())
        empty = _replace_sheet(empty, "Answer Spans", ())

        with patch.object(workbook_module, "_read_snapshot", return_value=empty):
            with self.assertRaisesRegex(DatasetFormatError, "removed every predicate"):
                validate_pilot_workbook("fixture.xlsx")

    def test_rejects_semantic_duplicates_and_incomplete_eventive_candidates(self) -> None:
        snapshot = _completed_snapshot()
        first_question = snapshot.rows["Questions"][0]
        duplicate = replace(
            first_question,
            row_number=999,
            values={**first_question.values, "question_id": "q-duplicate"},
        )
        duplicate_snapshot = _replace_sheet(
            snapshot,
            "Questions",
            (*snapshot.rows["Questions"], duplicate),
        )
        with patch.object(
            workbook_module, "_read_snapshot", return_value=duplicate_snapshot
        ):
            with self.assertRaisesRegex(DatasetFormatError, "scorer-equivalent"):
                validate_pilot_workbook("fixture.xlsx")

        first_candidate = snapshot.rows["Annotation"][0]
        record_id = first_candidate.values["record_id"]
        candidate_id = first_candidate.values["candidate_id"]
        no_questions = tuple(
            row
            for row in snapshot.rows["Questions"]
            if (row.values["record_id"], row.values["candidate_id"])
            != (record_id, candidate_id)
        )
        no_answers = tuple(
            row
            for row in snapshot.rows["Answer Spans"]
            if (row.values["record_id"], row.values["candidate_id"])
            != (record_id, candidate_id)
        )
        incomplete = _replace_sheet(snapshot, "Questions", no_questions)
        incomplete = _replace_sheet(incomplete, "Answer Spans", no_answers)
        with patch.object(workbook_module, "_read_snapshot", return_value=incomplete):
            with self.assertRaisesRegex(DatasetFormatError, "has no Questions row"):
                validate_pilot_workbook("fixture.xlsx")

        placeholder = replace(
            first_question,
            values={
                **first_question.values,
                "question_surface": "Anything?",
                "wh": "_",
                "aux": "_",
                "subj": "_",
                "verb": "_",
                "obj": "_",
                "prep": "_",
                "obj2": "_",
            },
        )
        placeholder_snapshot = _replace_sheet(
            snapshot,
            "Questions",
            (placeholder, *snapshot.rows["Questions"][1:]),
        )
        with patch.object(
            workbook_module, "_read_snapshot", return_value=placeholder_snapshot
        ):
            with self.assertRaisesRegex(DatasetFormatError, "substantive role question"):
                validate_pilot_workbook("fixture.xlsx")

        unsupported_wh = replace(
            first_question,
            values={
                **first_question.values,
                "question_surface": "Which role was involved?",
                "wh": "which",
            },
        )
        unsupported_wh_snapshot = _replace_sheet(
            snapshot,
            "Questions",
            (unsupported_wh, *snapshot.rows["Questions"][1:]),
        )
        with patch.object(
            workbook_module,
            "_read_snapshot",
            return_value=unsupported_wh_snapshot,
        ):
            with self.assertRaisesRegex(DatasetFormatError, "substantive role question"):
                validate_pilot_workbook("fixture.xlsx")

        missing_question_mark = replace(
            first_question,
            values={
                **first_question.values,
                "question_surface": first_question.values["question_surface"].removesuffix(
                    "?"
                ),
            },
        )
        missing_question_mark_snapshot = _replace_sheet(
            snapshot,
            "Questions",
            (missing_question_mark, *snapshot.rows["Questions"][1:]),
        )
        with patch.object(
            workbook_module,
            "_read_snapshot",
            return_value=missing_question_mark_snapshot,
        ):
            with self.assertRaisesRegex(DatasetFormatError, "must end with '\\?'"):
                validate_pilot_workbook("fixture.xlsx")

    def test_rejects_mixed_groups_and_bad_spans(self) -> None:
        snapshot = _completed_snapshot()
        last_candidate = snapshot.rows["Annotation"][-1]
        mixed_row = replace(
            last_candidate,
            values={**last_candidate.values, "annotator_id": "second-person"},
        )
        mixed = _replace_sheet(
            snapshot,
            "Annotation",
            (*snapshot.rows["Annotation"][:-1], mixed_row),
        )
        with patch.object(workbook_module, "_read_snapshot", return_value=mixed):
            with self.assertRaisesRegex(DatasetFormatError, "exactly one annotator/pass"):
                validate_pilot_workbook("fixture.xlsx")

        first_candidate = snapshot.rows["Annotation"][0]
        bad_span = replace(
            first_candidate,
            values={**first_candidate.values, "predicate_text": "not-the-source-text"},
        )
        malformed = _replace_sheet(
            snapshot,
            "Annotation",
            (bad_span, *snapshot.rows["Annotation"][1:]),
        )
        with patch.object(workbook_module, "_read_snapshot", return_value=malformed):
            with self.assertRaisesRegex(DatasetFormatError, "does not match Authoring"):
                validate_pilot_workbook("fixture.xlsx")

    def test_rejects_changed_authoring_and_active_review_log(self) -> None:
        snapshot = _completed_snapshot()
        first = snapshot.rows["Authoring"][0]
        changed_authoring = replace(
            first,
            values={**first.values, "raw_note": first.values["raw_note"] + " Extra"},
        )
        changed = _replace_sheet(
            snapshot,
            "Authoring",
            (changed_authoring, *snapshot.rows["Authoring"][1:]),
        )
        with patch.object(workbook_module, "_read_snapshot", return_value=changed):
            with self.assertRaisesRegex(DatasetFormatError, "accepted candidate-pilot"):
                validate_pilot_workbook("fixture.xlsx")

        review = snapshot.rows["Review Log"][0]
        active_review = replace(
            review,
            values={**review.values, "field": "predicate span"},
        )
        with_review = _replace_sheet(
            snapshot,
            "Review Log",
            (active_review, *snapshot.rows["Review Log"][1:]),
        )
        with patch.object(workbook_module, "_read_snapshot", return_value=with_review):
            with self.assertRaisesRegex(DatasetFormatError, "Review Log must remain blank"):
                validate_pilot_workbook("fixture.xlsx")
        with patch.object(workbook_module, "_read_snapshot", return_value=with_review):
            with self.assertRaisesRegex(DatasetFormatError, "Review Log must remain blank"):
                convert_pilot_workbook(
                    "fixture.xlsx",
                    annotator_id="ayo-adetayo",
                    pass_id="pass-1",
                )

    def test_real_xlsx_rejects_formula_and_out_of_table_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(PILOT_WORKBOOK) as archive:
                annotation_part = workbook_module._sheet_parts(archive)["Annotation"]
                worksheet = ElementTree.fromstring(archive.read(annotation_part))
            namespace = workbook_module._XML_NAMESPACE
            sheet_data = worksheet.find(f"{{{namespace}}}sheetData")
            row_two = next(
                row
                for row in sheet_data.findall(f"{{{namespace}}}row")
                if row.attrib.get("r") == "2"
            )
            cell_a2 = next(
                (
                    cell
                    for cell in row_two.findall(f"{{{namespace}}}c")
                    if cell.attrib.get("r") == "A2"
                ),
                None,
            )
            if cell_a2 is None:
                cell_a2 = ElementTree.SubElement(row_two, f"{{{namespace}}}c", {"r": "A2"})
            ElementTree.SubElement(cell_a2, f"{{{namespace}}}f").text = "1+1"
            formula_path = root / "formula.xlsx"
            _rewrite_package_member(
                PILOT_WORKBOOK,
                formula_path,
                annotation_part,
                ElementTree.tostring(worksheet, encoding="utf-8", xml_declaration=True),
            )
            with self.assertRaisesRegex(DatasetFormatError, "forbidden formula"):
                validate_pilot_workbook(formula_path)

            with zipfile.ZipFile(PILOT_WORKBOOK) as archive:
                worksheet = ElementTree.fromstring(archive.read(annotation_part))
            sheet_data = worksheet.find(f"{{{namespace}}}sheetData")
            row = ElementTree.SubElement(sheet_data, f"{{{namespace}}}row", {"r": "102"})
            cell = ElementTree.SubElement(
                row,
                f"{{{namespace}}}c",
                {"r": "A102", "t": "inlineStr"},
            )
            inline = ElementTree.SubElement(cell, f"{{{namespace}}}is")
            ElementTree.SubElement(inline, f"{{{namespace}}}t").text = "ops-pilot-001"
            outside_path = root / "outside.xlsx"
            _rewrite_package_member(
                PILOT_WORKBOOK,
                outside_path,
                annotation_part,
                ElementTree.tostring(worksheet, encoding="utf-8", xml_declaration=True),
            )
            with self.assertRaisesRegex(DatasetFormatError, "outside its table"):
                validate_pilot_workbook(outside_path)

    def test_cli_conversion_and_input_output_alias_guard(self) -> None:
        original_sha = workbook_module.sha256_file(PILOT_WORKBOOK)
        with self.assertRaisesRegex(ValueError, "must be different files"):
            data_main(
                [
                    "convert-pilot-workbook",
                    str(PILOT_WORKBOOK),
                    str(PILOT_WORKBOOK),
                    "--annotator-id",
                    "ayo-adetayo",
                    "--pass-id",
                    "pass-1",
                    "--overwrite",
                ]
            )
        self.assertEqual(workbook_module.sha256_file(PILOT_WORKBOOK), original_sha)

        snapshot = _completed_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pass-1.json"
            captured = io.StringIO()
            with patch.object(workbook_module, "_read_snapshot", return_value=snapshot):
                with redirect_stdout(captured):
                    status = data_main(
                        [
                            "convert-pilot-workbook",
                            "fixture.xlsx",
                            str(output),
                            "--annotator-id",
                            "ayo-adetayo",
                            "--pass-id",
                            "pass-1",
                        ]
                    )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(captured.getvalue())["output"], str(output))
            self.assertEqual(len(load_evaluation_bundle(output).corpus.predicates), 20)
            with patch.object(workbook_module, "_read_snapshot", return_value=snapshot):
                with self.assertRaises(FileExistsError):
                    data_main(
                        [
                            "convert-pilot-workbook",
                            "fixture.xlsx",
                            str(output),
                            "--annotator-id",
                            "ayo-adetayo",
                            "--pass-id",
                            "pass-1",
                        ]
                    )
            with patch.object(workbook_module, "_read_snapshot", return_value=snapshot):
                with redirect_stdout(io.StringIO()):
                    overwrite_status = data_main(
                        [
                            "convert-pilot-workbook",
                            "fixture.xlsx",
                            str(output),
                            "--annotator-id",
                            "ayo-adetayo",
                            "--pass-id",
                            "pass-1",
                            "--overwrite",
                        ]
                    )
            self.assertEqual(overwrite_status, 0)
            self.assertEqual(len(load_evaluation_bundle(output).corpus.predicates), 20)

    def test_frozen_tokenizer_preserves_exact_code_point_spans(self) -> None:
        text = "Supervisor’s on-call note at 02:15: 🚀"
        tokens = tokenize_pilot_text(text)

        self.assertEqual(
            [token.text for token in tokens],
            ["Supervisor’s", "on-call", "note", "at", "02:15", ":", "🚀"],
        )
        self.assertEqual(
            [text[token.start : token.end] for token in tokens],
            [token.text for token in tokens],
        )
        self.assertEqual([token.index for token in tokens], list(range(len(tokens))))


if __name__ == "__main__":
    unittest.main()
