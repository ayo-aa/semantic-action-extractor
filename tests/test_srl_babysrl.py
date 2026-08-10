import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from semantic_action_extractor.srl.babysrl import (
    BABYSRL_ARCHIVE_SHA256,
    BABYSRL_ARCHIVE_SIZE_BYTES,
    BabySRLArchiveError,
    BabySRLPreparationError,
    assign_babysrl_document_splits,
    audit_babysrl_archive,
    build_babysrl_dataset,
    build_babysrl_document_assignment_manifest,
    convert_babysrl_chat,
    main,
    prepare_babysrl_archive,
)
from semantic_action_extractor.srl.dataset_io import read_prepared_dataset


REAL_ARCHIVE = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "BabySRL.zip"
)


def _pin(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def _single_proposition(subject: str) -> str:
    return "\n".join(
        (
            "*MOT: invented utterance",
            f"%srl: {subject} - (A0*)",
            "%srl: crafts craft (V*)",
            "%srl: kites - (A1*)",
        )
    )


def _write_archive(path: Path, documents: dict[tuple[str, str], str]) -> None:
    with ZipFile(path, "w") as archive:
        for (child, document_name), source in documents.items():
            archive.writestr(
                f"BabySRL/{child}/{document_name}.srl.cha",
                source,
            )


def _ten_adam_documents() -> dict[tuple[str, str], str]:
    return {
        ("Adam", f"adam{index:02d}"): _single_proposition(f"Inventor{index}")
        for index in range(1, 11)
    }


class BabySRLChatConversionTests(unittest.TestCase):
    def test_converts_multiple_columns_and_normalizes_supported_roles(self) -> None:
        source = "\n".join(
            (
                "*MOT: invented utterance",
                "%srl: Vela - (A0*) (A0*)",
                "%srl: packs pack (V*) *",
                "%srl: crates - (A1*) (A1*)",
                "%srl: nightly schedule (AM-TMP*) (V*)",
            )
        )

        conversion = convert_babysrl_chat(
            source,
            child="Adam",
            document_name="invented01",
        )

        self.assertEqual(len(conversion.examples), 2)
        first, second = conversion.examples
        self.assertEqual(
            first.tags,
            ("B-ARG0", "B-V", "B-ARG1", "B-ARGM-TMP"),
        )
        self.assertEqual(first.predicate_index, 1)
        self.assertEqual(first.predicate_roleset, "pack.XX")
        self.assertEqual(
            second.tags,
            ("B-ARG0", "O", "B-ARG1", "B-V"),
        )
        self.assertEqual(second.predicate_index, 3)
        self.assertEqual(conversion.report["declared_proposition_columns"], 2)
        self.assertEqual(conversion.report["accepted_proposition_columns"], 2)

    def test_relation_union_selects_one_head_and_encodes_other_pieces(self) -> None:
        source = "\n".join(
            (
                "*FAT: invented utterance",
                "%srl: Ivo - (A0*)",
                "%srl: sets - (V*)",
                "%srl: up assemble (C-V*)",
                "%srl: displays - (A1*)",
            )
        )

        conversion = convert_babysrl_chat(
            source,
            child="Eve",
            document_name="invented02",
        )

        example = conversion.examples[0]
        self.assertEqual(example.predicate_index, 2)
        self.assertEqual(
            example.tags,
            ("B-ARG0", "B-C-V", "B-V", "B-ARG1"),
        )
        self.assertEqual(example.tags.count("B-V"), 1)

    def test_multiword_relation_has_one_anchor_and_continuation_tags(self) -> None:
        source = "\n".join(
            (
                "*MOT: invented utterance",
                "%srl: Orin - (A0*)",
                "%srl: maps map (V*",
                "%srl: out - *",
                "%srl: routes - *)",
            )
        )

        conversion = convert_babysrl_chat(
            source,
            child="Sarah",
            document_name="invented03",
        )

        self.assertEqual(
            conversion.examples[0].tags,
            ("B-ARG0", "B-V", "B-C-V", "I-C-V"),
        )

    def test_fail_closed_reasons_are_exclusive_and_aggregate_only(self) -> None:
        cases = {
            "invalid_bracket_sequence": (
                "*MOT: invented utterance\n"
                "%srl: Rhea - (A0*\n"
                "%srl: welds weld (V*)\n"
                "%srl: frames - (A1*)"
            ),
            "missing_relation_span": (
                "*MOT: invented utterance\n"
                "%srl: Sela - (A0*)\n"
                "%srl: sorts sort (A1*)"
            ),
            "ambiguous_predicate_head": (
                "*FAT: invented utterance\n"
                "%srl: Taro lift (V*)\n"
                "%srl: stacks move (C-V*)"
            ),
            "unsupported_role_label": (
                "*MOT: invented utterance\n"
                "%srl: Uma - (R-A1*)\n"
                "%srl: seals seal (V*)"
            ),
        }
        for reason, source in cases.items():
            with self.subTest(reason=reason):
                conversion = convert_babysrl_chat(
                    source,
                    child="Adam",
                    document_name="invented04",
                )
                self.assertEqual(conversion.examples, ())
                self.assertEqual(conversion.report["rejection_reasons"], {reason: 1})
                serialized = json.dumps(conversion.report)
                self.assertNotIn("Rhea", serialized)
                self.assertNotIn("R-A1", serialized)

    def test_width_mismatch_rejects_every_declared_column(self) -> None:
        source = "\n".join(
            (
                "*MOT: invented utterance",
                "%srl: Wren - (A0*)",
                "%srl: builds build (V*) (V*)",
            )
        )

        conversion = convert_babysrl_chat(
            source,
            child="Adam",
            document_name="invented05",
        )

        self.assertEqual(conversion.examples, ())
        self.assertEqual(conversion.report["declared_proposition_columns"], 2)
        self.assertEqual(conversion.report["first_row_proposition_columns"], 1)
        self.assertEqual(
            conversion.report["rejection_reasons"], {"row_width_mismatch": 2}
        )

    def test_rejects_argument_continuations_references_and_unknown_roles(self) -> None:
        for raw_label in ("C-A1", "R-A1", "A-A1", "A-LOC", "AV-R"):
            with self.subTest(raw_label=raw_label):
                source = "\n".join(
                    (
                        "*MOT: invented utterance",
                        f"%srl: Yori - ({raw_label}*)",
                        "%srl: molds mold (V*)",
                    )
                )
                conversion = convert_babysrl_chat(
                    source,
                    child="Adam",
                    document_name="invented06",
                )
                self.assertEqual(conversion.examples, ())
                self.assertEqual(
                    conversion.report["rejection_reasons"],
                    {"unsupported_role_label": 1},
                )

    def test_requires_a_parent_main_tier_but_allows_dependent_tiers(self) -> None:
        valid = "\n".join(
            (
                "*MOT: invented utterance",
                "%mor: invented dependent tier",
                "%trn: invented dependent tier",
                "%srl: Ziva - (A0*)",
                "%srl: sketches sketch (V*)",
            )
        )
        conversion = convert_babysrl_chat(
            valid,
            child="Adam",
            document_name="invented07",
        )
        self.assertEqual(len(conversion.examples), 1)

        invalid_sources = (
            "%srl: Ziva - (A0*)\n%srl: sketches sketch (V*)",
            (
                "*CHI: invented utterance\n"
                "%srl: Ziva - (A0*)\n"
                "%srl: sketches sketch (V*)"
            ),
            (
                "*OBS: invented utterance\n"
                "%srl: Ziva - (A0*)\n"
                "%srl: sketches sketch (V*)"
            ),
        )
        for source in invalid_sources:
            with self.subTest(source_kind=source.partition(":")[0]):
                with self.assertRaisesRegex(
                    BabySRLPreparationError, "MOT/FAT"
                ):
                    convert_babysrl_chat(
                        source,
                        child="Adam",
                        document_name="invented08",
                    )


class BabySRLArchiveTests(unittest.TestCase):
    def test_rejects_bad_size_digest_unsafe_and_unexpected_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "invented.zip"
            _write_archive(
                archive_path,
                {("Adam", "adam01"): _single_proposition("Xara")},
            )
            digest, size = _pin(archive_path)

            with self.assertRaisesRegex(BabySRLArchiveError, "size"):
                audit_babysrl_archive(
                    archive_path,
                    expected_sha256=digest,
                    expected_size_bytes=size + 1,
                )
            with self.assertRaisesRegex(BabySRLArchiveError, "SHA-256"):
                audit_babysrl_archive(
                    archive_path,
                    expected_sha256="0" * 64,
                    expected_size_bytes=size,
                )

            unsafe_path = root / "unsafe.zip"
            with ZipFile(unsafe_path, "w") as archive:
                archive.writestr("../escape.txt", "invented")
            unsafe_digest, unsafe_size = _pin(unsafe_path)
            with self.assertRaisesRegex(BabySRLArchiveError, "unsafe"):
                audit_babysrl_archive(
                    unsafe_path,
                    expected_sha256=unsafe_digest,
                    expected_size_bytes=unsafe_size,
                )

            unexpected_path = root / "unexpected.zip"
            with ZipFile(unexpected_path, "w") as archive:
                archive.writestr("notes.txt", "invented")
            unexpected_digest, unexpected_size = _pin(unexpected_path)
            with self.assertRaisesRegex(BabySRLArchiveError, "unexpected"):
                audit_babysrl_archive(
                    unexpected_path,
                    expected_sha256=unexpected_digest,
                    expected_size_bytes=unexpected_size,
                )

    def test_document_split_is_stable_and_has_predeclared_counts(self) -> None:
        documents = {
            "Adam": tuple(f"babysrl:adam:adam{index:02d}" for index in range(1, 24)),
            "Eve": tuple(f"babysrl:eve:eve{index:02d}" for index in range(1, 21)),
            "Sarah": tuple(
                f"babysrl:sarah:sarah{index:03d}" for index in range(1, 91)
            ),
        }

        first = assign_babysrl_document_splits(documents)
        second = assign_babysrl_document_splits(
            {child: tuple(reversed(items)) for child, items in documents.items()}
        )
        first_manifest = build_babysrl_document_assignment_manifest(
            documents, first
        )
        second_manifest = build_babysrl_document_assignment_manifest(
            {child: tuple(reversed(items)) for child, items in documents.items()},
            second,
        )

        self.assertEqual(dict(first), dict(second))
        self.assertEqual(
            first_manifest.canonical_bytes, second_manifest.canonical_bytes
        )
        self.assertEqual(first_manifest.sha256, second_manifest.sha256)
        self.assertEqual(first_manifest.document_count, 133)
        self.assertEqual(
            hashlib.sha256(first_manifest.canonical_bytes).hexdigest(),
            first_manifest.sha256,
        )
        manifest_payload = json.loads(first_manifest.canonical_bytes)
        self.assertEqual(len(manifest_payload["documents"]), 133)
        self.assertEqual(
            set(manifest_payload["documents"][0]),
            {"child", "document_id", "source_basename", "split"},
        )
        expected = {
            "Adam": {"train": 18, "development": 2, "test": 3},
            "Eve": {"train": 16, "development": 2, "test": 2},
            "Sarah": {"train": 72, "development": 9, "test": 9},
        }
        for child, document_ids in documents.items():
            counts = Counter(first[document_id] for document_id in document_ids)
            self.assertEqual(dict(counts), expected[child])

    def test_duplicate_policy_excludes_cross_split_and_keeps_train_copies(self) -> None:
        documents = _ten_adam_documents()
        ids = tuple(f"babysrl:adam:adam{index:02d}" for index in range(1, 11))
        assignments = assign_babysrl_document_splits({"Adam": ids})
        by_split: dict[str, list[str]] = {
            "train": [],
            "development": [],
            "test": [],
        }
        for document_id, split in assignments.items():
            by_split[split].append(document_id)

        shared_across = _single_proposition("SharedAcross")
        for split in ("train", "development", "test"):
            document_id = sorted(by_split[split])[0]
            name = document_id.rsplit(":", 1)[-1]
            documents[("Adam", name)] = shared_across
        two_train_ids = sorted(by_split["train"])[1:3]
        for document_id in two_train_ids:
            name = document_id.rsplit(":", 1)[-1]
            documents[("Adam", name)] = _single_proposition("TrainTwin")

        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            _write_archive(archive_path, documents)
            digest, size = _pin(archive_path)
            build = build_babysrl_dataset(
                archive_path,
                expected_sha256=digest,
                expected_size_bytes=size,
            )

        split_report = build.report["split"]
        self.assertEqual(split_report["cross_split_word_sequence_groups"], 1)
        self.assertEqual(
            split_report["cross_split_leakage_exclusions"],
            {"train": 1, "development": 1, "test": 1},
        )
        self.assertEqual(
            split_report["dev_test_exact_semantic_duplicates_removed"],
            {"train": 0, "development": 0, "test": 0},
        )
        train_twins = [
            example for example in build.examples if example.words[0] == "TrainTwin"
        ]
        self.assertEqual(len(train_twins), 2)

    def test_duplicate_policy_deduplicates_dev_and_test_deterministically(self) -> None:
        documents = {
            ("Eve", f"eve{index:02d}"): _single_proposition(f"Maker{index}")
            for index in range(1, 21)
        }
        ids = tuple(f"babysrl:eve:eve{index:02d}" for index in range(1, 21))
        assignments = assign_babysrl_document_splits({"Eve": ids})
        by_split: dict[str, list[str]] = {
            "train": [],
            "development": [],
            "test": [],
        }
        for document_id, split in assignments.items():
            by_split[split].append(document_id)
        for split, subject in (
            ("development", "DevelopmentTwin"),
            ("test", "TestTwin"),
        ):
            for document_id in by_split[split]:
                name = document_id.rsplit(":", 1)[-1]
                documents[("Eve", name)] = _single_proposition(subject)

        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            _write_archive(archive_path, documents)
            digest, size = _pin(archive_path)
            first = build_babysrl_dataset(
                archive_path,
                expected_sha256=digest,
                expected_size_bytes=size,
            )
            second = build_babysrl_dataset(
                archive_path,
                expected_sha256=digest,
                expected_size_bytes=size,
            )

        self.assertEqual(first.examples, second.examples)
        self.assertEqual(
            first.report["split"]["dev_test_exact_semantic_duplicates_removed"],
            {"train": 0, "development": 1, "test": 1},
        )
        self.assertEqual(
            first.report["split"]["final_examples"],
            {"train": 16, "development": 1, "test": 1},
        )

    def test_preparation_rejects_duplicate_sentence_predicate_identity(self) -> None:
        source = "\n".join(
            (
                "*MOT: invented utterance",
                "%srl: Nara - (A0*) (A0*)",
                "%srl: folds fold (V*) (V*)",
                "%srl: maps - (A1*) (A1*)",
            )
        )
        conversion = convert_babysrl_chat(
            source,
            child="Adam",
            document_name="adam01",
        )
        self.assertEqual(conversion.report["accepted_proposition_columns"], 2)
        self.assertEqual(len(conversion.examples), 2)

        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            _write_archive(archive_path, {("Adam", "adam01"): source})
            digest, size = _pin(archive_path)
            with self.assertRaisesRegex(
                BabySRLPreparationError, "duplicate sentence/predicate"
            ):
                build_babysrl_dataset(
                    archive_path,
                    expected_sha256=digest,
                    expected_size_bytes=size,
                )

    def test_preparation_requires_talkbank_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "invented.zip"
            output_path = root / "prepared"
            _write_archive(archive_path, _ten_adam_documents())
            digest, size = _pin(archive_path)

            with self.assertRaisesRegex(BabySRLPreparationError, "TalkBank"):
                prepare_babysrl_archive(
                    archive_path,
                    output_path,
                    expected_sha256=digest,
                    expected_size_bytes=size,
                )
            self.assertFalse(output_path.exists())

            receipt = prepare_babysrl_archive(
                archive_path,
                output_path,
                talkbank_access_and_rules_confirmed=True,
                expected_sha256=digest,
                expected_size_bytes=size,
            )
            loaded = read_prepared_dataset(output_path)

        self.assertEqual(len(loaded.examples), 10)
        self.assertEqual(
            loaded.manifest.dataset_fingerprint,
            receipt.manifest.dataset_fingerprint,
        )

    def test_cli_requires_explicit_talkbank_flag_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error_output = io.StringIO()
            with contextlib.redirect_stderr(error_output):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            str(root / "missing.zip"),
                            "--output-directory",
                            str(root / "prepared"),
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "--confirm-talkbank-access-and-rules", error_output.getvalue()
        )

    def test_cli_audit_outputs_only_aggregate_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "invented.zip"
            _write_archive(archive_path, _ten_adam_documents())
            digest, size = _pin(archive_path)
            self.assertNotEqual(digest, BABYSRL_ARCHIVE_SHA256)
            self.assertNotEqual(size, BABYSRL_ARCHIVE_SIZE_BYTES)

            # The public CLI intentionally has no unpinned override.  Exercise
            # its serializer through the pinned low-level audit instead.
            report = audit_babysrl_archive(
                archive_path,
                expected_sha256=digest,
                expected_size_bytes=size,
            )
            serialized = json.dumps(report)

        self.assertNotIn("Inventor", serialized)
        self.assertNotIn("crafts", serialized)
        self.assertEqual(report["gate"]["status"], "pass")

    @unittest.skipUnless(REAL_ARCHIVE.is_file(), "ignored BabySRL archive absent")
    def test_pinned_real_archive_reconciles_aggregate_counts(self) -> None:
        report = audit_babysrl_archive(REAL_ARCHIVE)

        conversion = report["conversion"]
        self.assertEqual(conversion["declared_proposition_columns"], 18_536)
        self.assertEqual(conversion["accepted_proposition_columns"], 18_397)
        self.assertEqual(conversion["rejected_proposition_columns"], 139)
        self.assertEqual(
            conversion["rejection_reasons"],
            {
                "ambiguous_predicate_head": 5,
                "invalid_bracket_sequence": 15,
                "missing_relation_span": 99,
                "row_width_mismatch": 4,
                "unsupported_role_label": 16,
            },
        )
        self.assertEqual(conversion["first_row_proposition_columns"], 18_535)
        self.assertEqual(conversion["predicate_marker_rows"], 18_556)
        self.assertEqual(
            report["split"]["document_assignment_manifest"],
            {
                "schema_version": (
                    "semantic-action-extractor.babysrl-document-splits/v1"
                ),
                "document_count": 133,
                "sha256": (
                    "73ecae9f81d1d2b9f8495b13b297da9c"
                    "3d24e387630e71a38ef2420d4c9a5de7"
                ),
            },
        )
        self.assertEqual(report["gate"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
