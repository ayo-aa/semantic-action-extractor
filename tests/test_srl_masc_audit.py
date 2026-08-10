import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from semantic_action_extractor.srl.masc_audit import (
    PINNED_ALLOWED_DOCUMENTS,
    audit_masc_archive,
    canonical_document_id,
    source_disposition,
    source_family,
)


REAL_ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "Propbank-original-format.zip"
)


def _write_minimal_archive(
    archive_path: Path,
    *,
    document: str = "invented",
    text_genre: str = "written",
    ptb_genre: str = "written",
    prop_genre: str = "written",
    tree: str = (
        "(TOP (S (NP-SBJ (NNP Mira)) "
        "(VP (VBD mailed) (NP (DT a) (NN parcel))) (. .)))"
    ),
    rows: str = (
        "0 1 gold mail-v mail.01 ----- "
        "0:1-ARG0 1:0-rel 2:1-ARG1\n"
    ),
    include_text: bool = True,
    include_ptb: bool = True,
    include_prop: bool = True,
) -> None:
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("Propbank/README.txt", "invented readme")
        if include_text:
            archive.writestr(
                f"Propbank/MASC1_textfiles/{text_genre}/{document}.txt",
                "Mira mailed a parcel.",
            )
        if include_ptb:
            archive.writestr(
                "Propbank/Penn_Treebank-orig/data/"
                f"{ptb_genre}/{document}.mrg",
                tree,
            )
        if include_prop:
            archive.writestr(
                f"Propbank/Propbank-orig/data/{prop_genre}/{document}.prop",
                rows,
            )


class MascAuditTests(unittest.TestCase):
    def test_canonicalizes_only_verified_archive_aliases(self) -> None:
        self.assertEqual(
            canonical_document_id(
                "ptb", "x/sw2025-ms98-a-trans.ascii-1-NEW.mrg"
            ),
            "sw2025-ms98-a-trans",
        )
        self.assertEqual(
            canonical_document_id("ptb", "x/117CWL009.mrg.txt"),
            "117CWL009",
        )
        self.assertEqual(
            canonical_document_id("prop", "x/wsj_1640.prop-NEW.prop"),
            "wsj_1640",
        )
        self.assertEqual(
            canonical_document_id(
                "prop", "x/invented_LU_ANNOTATE.prop"
            ),
            "invented_LU_ANNOTATE",
        )
        self.assertEqual(
            canonical_document_id(
                "prop",
                "x/ENRON-pearson-email-25jul02_LU_ANNOTATE.prop",
            ),
            "ENRON-pearson-email-25jul02",
        )

    def test_source_policy_is_pinned_and_fails_closed(self) -> None:
        self.assertEqual(len(PINNED_ALLOWED_DOCUMENTS), 48)
        self.assertEqual(source_family("pmed.0010029"), "PLOS")
        self.assertEqual(source_disposition("pmed.0010029"), "diagnostic")
        self.assertEqual(source_disposition("wsj_synthetic"), "deny")
        self.assertEqual(source_disposition("invented"), "hold")

    def test_rejects_an_unpinned_archive_before_payload_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            _write_minimal_archive(archive_path)

            with self.assertRaisesRegex(ValueError, "does not match"):
                audit_masc_archive(archive_path)

    def test_audits_synthetic_zip_without_returning_corpus_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("Propbank/README.txt", "invented readme")
                archive.writestr(
                    "Propbank/MASC1_textfiles/written/invented.txt",
                    "Mira mailed a parcel.",
                )
                archive.writestr(
                    "Propbank/MASC1_textfiles/.svn/text-base/ignored.txt",
                    "legacy invented copy",
                )
                archive.writestr(
                    "Propbank/Penn_Treebank-orig/data/written/invented.mrg",
                    "(TOP (S (NP-SBJ (NNP Mira)) "
                    "(VP (VBD mailed) (NP (DT a) (NN parcel))) (. .)))",
                )
                archive.writestr(
                    "Propbank/Propbank-orig/data/written/invented.prop",
                    "0 1 gold mail-v mail.01 ----- "
                    "0:1-ARG0 1:0-rel 2:1-ARG1\n",
                )

            report = audit_masc_archive(
                archive_path,
                allowed_document_ids={"invented"},
                allow_unpinned_archive=True,
            )

        self.assertEqual(report["join"]["all_three_layer_documents"], 1)
        self.assertEqual(report["archive"]["svn_files_ignored"], 1)
        self.assertEqual(report["g2"]["retained_documents"], 1)
        self.assertEqual(report["g3"]["eligible_verbal_records"], 1)
        self.assertEqual(report["g3"]["converted_records"], 1)
        self.assertEqual(report["g3"]["conversion_coverage"], 1.0)
        serialized = json.dumps(report)
        self.assertNotIn("Mira mailed", serialized)
        self.assertNotIn("0:1-ARG0", serialized)

    def test_reports_unsafe_archive_path_without_extracting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "invented")

            report = audit_masc_archive(
                archive_path,
                allowed_document_ids=set(),
                allow_unpinned_archive=True,
            )

        self.assertEqual(report["archive"]["unsafe_paths"], ["../escape.txt"])
        self.assertEqual(report["archive"]["status"], "fail")
        self.assertEqual(report["decision"]["reason"], "archive_pin_mismatch")

    def test_treats_backslashes_as_unsafe_archive_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(r"nested\\escape.txt", "invented")

            report = audit_masc_archive(
                archive_path,
                allowed_document_ids=set(),
                allow_unpinned_archive=True,
            )

        self.assertEqual(
            report["archive"]["unsafe_paths"], [r"nested\\escape.txt"]
        )

    def test_ws_j_denial_cannot_be_overridden_by_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            _write_minimal_archive(
                archive_path,
                document="wsj_synthetic",
            )
            report = audit_masc_archive(
                archive_path,
                allowed_document_ids={"wsj_synthetic"},
                allow_unpinned_archive=True,
            )

        self.assertEqual(report["rights"]["effective_allowed_documents"], 0)
        self.assertEqual(
            report["rights"]["denied_override_documents"],
            ["wsj_synthetic"],
        )
        self.assertEqual(report["g2"]["status"], "fail")
        self.assertEqual(
            report["g2"]["exclusions"]["wsj_synthetic"]["reasons"],
            ["wsj_denied_override"],
        )
        self.assertEqual(
            report["g2"]["exclusions"]["wsj_synthetic"][
                "canonical_prop_rows"
            ],
            1,
        )

    def test_manifest_layer_absence_fails_g2_with_explicit_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            _write_minimal_archive(archive_path, include_prop=False)
            report = audit_masc_archive(
                archive_path,
                allowed_document_ids={"invented"},
                allow_unpinned_archive=True,
            )

        self.assertEqual(report["g2"]["status"], "fail")
        self.assertEqual(
            report["g2"]["exclusions"]["invented"],
            {
                "reasons": ["prop_layer_missing"],
                "canonical_prop_rows": 0,
            },
        )

    def test_tree_failure_cannot_be_treated_as_an_exclusion_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            _write_minimal_archive(
                archive_path,
                tree="((S (NNP Mira)) (S (VBD mailed)))",
            )
            report = audit_masc_archive(
                archive_path,
                allowed_document_ids={"invented"},
                allow_unpinned_archive=True,
            )

        self.assertEqual(report["g2"]["status"], "fail")
        self.assertEqual(report["g2"]["retained_documents"], 0)
        self.assertEqual(
            report["g2"]["tree_failure_documents"],
            {"invented": "unlabeled_tree_wrapper_invalid"},
        )
        exclusion = report["g2"]["exclusions"]["invented"]
        self.assertEqual(exclusion["canonical_prop_rows"], 1)
        self.assertEqual(
            exclusion["reasons"],
            ["ptb_unlabeled_tree_wrapper_invalid"],
        )

    def test_predicate_bounds_and_rel_count_fail_retained_gate(self) -> None:
        cases = {
            "predicate": (
                "0 9 gold mail-v mail.01 ----- 0:1-ARG0 9:0-rel\n",
                "predicate_terminal_out_of_range",
            ),
            "rel": (
                "0 1 gold mail-v mail.01 ----- 0:1-ARG0\n",
                "record_rel_count_invalid",
            ),
        }
        for label, (rows, reason) in cases.items():
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as directory,
            ):
                archive_path = Path(directory) / "invented.zip"
                _write_minimal_archive(archive_path, rows=rows)
                report = audit_masc_archive(
                    archive_path,
                    allowed_document_ids={"invented"},
                    allow_unpinned_archive=True,
                )

            self.assertEqual(report["g2"]["status"], "fail")
            self.assertEqual(report["g2"]["row_failures"], {reason: 1})
            self.assertEqual(
                report["g2"]["exclusions"]["invented"]["reasons"],
                [reason],
            )

    def test_cross_layer_genre_conflict_fails_g2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            _write_minimal_archive(archive_path, ptb_genre="spoken")
            report = audit_masc_archive(
                archive_path,
                allowed_document_ids={"invented"},
                allow_unpinned_archive=True,
            )

        self.assertEqual(
            report["join"]["cross_layer_genre_conflicts"]["invented"],
            {
                "text": ["written"],
                "ptb": ["spoken"],
                "prop": ["written"],
            },
        )
        self.assertEqual(report["g2"]["status"], "fail")
        self.assertEqual(
            report["g2"]["exclusions"]["invented"]["reasons"],
            ["cross_layer_genre_conflict"],
        )

    def test_distinguishes_archive_and_frame_legal_notices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("Propbank/LICENSE.txt", "invented terms")
                for name in (
                    "copying-n.xml",
                    "license-n.xml",
                    "license-v.xml",
                    "notice-n.xml",
                    "notice-v.xml",
                ):
                    archive.writestr(
                        f"Propbank/Propbank-orig/framefiles/{name}",
                        "<invented />",
                    )
            report = audit_masc_archive(
                archive_path,
                allowed_document_ids=set(),
                allow_unpinned_archive=True,
            )

        self.assertEqual(
            report["archive"]["archive_level_legal_notice_files"],
            ["Propbank/LICENSE.txt"],
        )
        self.assertEqual(
            len(report["archive"]["frame_legal_notice_files"]),
            5,
        )

    @unittest.skipUnless(
        REAL_ARCHIVE.is_file(),
        "pinned MASC archive is not available locally",
    )
    def test_pinned_archive_aggregate_regression(self) -> None:
        report = audit_masc_archive(REAL_ARCHIVE)

        self.assertTrue(report["acquisition"]["matches_pinned_sha256"])
        self.assertEqual(report["archive"]["entries"], 8_091)
        self.assertEqual(report["archive"]["files"], 8_068)
        self.assertEqual(report["archive"]["svn_files_ignored"], 67)
        self.assertEqual(
            {
                layer: values["files"]
                for layer, values in report["layers"].items()
            },
            {"text": 99, "ptb": 100, "prop": 100},
        )
        self.assertEqual(report["join"]["canonical_union_documents"], 102)
        self.assertEqual(report["join"]["all_three_layer_documents"], 96)
        self.assertEqual(report["join"]["physical_prop_rows"], 15_002)
        self.assertEqual(report["join"]["canonical_prop_rows"], 14_884)
        self.assertEqual(report["join"]["full_joined_prop_rows"], 14_698)
        self.assertEqual(
            report["join"]["non_wsj_fully_joined_prop_rows"],
            13_919,
        )
        self.assertEqual(
            report["join"]["conflicting_duplicate_predicate_instances"],
            3,
        )
        self.assertEqual(
            len(report["archive"]["frame_legal_notice_files"]),
            5,
        )
        self.assertEqual(
            report["archive"]["archive_level_legal_notice_files"],
            [],
        )
        self.assertEqual(report["join"]["cross_layer_genre_conflicts"], {})
        self.assertEqual(report["g2"]["status"], "fail")
        self.assertEqual(
            report["rights"]["status"],
            "incomplete_provisional_diagnostic_manifest",
        )
        self.assertEqual(
            report["decision"]["reason"],
            "multiple_feasibility_gates_failed",
        )
        self.assertEqual(
            report["decision"]["blocking_gates"],
            ["g1_rights", "g2_join", "g3_annotation_fit"],
        )
        self.assertEqual(
            set(report["g2"]["exclusions"]),
            {
                "VOL15_3",
                "ch5",
                "sw2025-ms98-a-trans",
                "sw2071-ms98-a-trans",
            },
        )


if __name__ == "__main__":
    unittest.main()
