import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

from semantic_action_extractor.srl import manual_review
from semantic_action_extractor.srl.babysrl import build_babysrl_dataset
from semantic_action_extractor.srl.dataset_io import write_prepared_dataset
from semantic_action_extractor.srl.manual_review import (
    REVIEW_AGGREGATE_FILENAME,
    REVIEW_DECISIONS_FILENAME,
    REVIEW_ITEMS_FILENAME,
    REVIEW_MANIFEST_FILENAME,
    ReviewAuthorizationError,
    ReviewFormatError,
    ReviewIntegrityError,
    ReviewPathError,
    TalkBankReviewAuthorization,
    create_manual_review_package,
    finalize_manual_review,
    main,
)


class BabySRLManualReviewTests(unittest.TestCase):
    def test_requires_explicit_registration_and_rules_before_any_read(self) -> None:
        with self.assertRaisesRegex(ReviewAuthorizationError, "TalkBank"):
            create_manual_review_package(
                "/not/read/raw.zip",
                "/not/read/prepared",
                "/not/written/review",
                repository_root="/not/read/repository",
                authorization=None,
            )
        with self.assertRaisesRegex(ReviewAuthorizationError, "registration"):
            TalkBankReviewAuthorization(
                registration_confirmed=False,
                ground_rules_confirmed=True,
                accepted_on="2026-08-10",
            )
        with self.assertRaisesRegex(ReviewAuthorizationError, "ground rules"):
            TalkBankReviewAuthorization(
                registration_confirmed=True,
                ground_rules_confirmed=False,
                accepted_on="2026-08-10",
            )
        with self.assertRaisesRegex(ReviewAuthorizationError, "YYYY-MM-DD"):
            TalkBankReviewAuthorization(
                registration_confirmed=True,
                ground_rules_confirmed=True,
                accepted_on="08/10/2026",
            )

    def test_selection_is_deterministic_stratified_and_raw_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._new_repository(Path(directory))
            archive, prepared, digest, size = self._prepared_synthetic_data(root)
            first = root / "data" / "review" / "first"
            second = root / "data" / "review" / "second"

            with self._patched_archive_pin(digest, size):
                first_receipt = create_manual_review_package(
                    archive,
                    prepared,
                    first,
                    repository_root=root,
                    authorization=self._authorization(),
                    per_stratum=1,
                )
                second_receipt = create_manual_review_package(
                    archive,
                    prepared,
                    second,
                    repository_root=root,
                    authorization=self._authorization(),
                    per_stratum=1,
                )

            self.assertEqual(first_receipt, second_receipt)
            self.assertEqual(
                (first / REVIEW_ITEMS_FILENAME).read_bytes(),
                (second / REVIEW_ITEMS_FILENAME).read_bytes(),
            )
            self.assertTrue(first_receipt.sample_counts_by_stratum)
            self.assertTrue(
                all(
                    value == 1
                    for value in first_receipt.sample_counts_by_stratum.values()
                )
            )
            item_text = (first / REVIEW_ITEMS_FILENAME).read_text(
                encoding="utf-8"
            )
            item = json.loads(
                item_text.splitlines()[0]
            )
            self.assertEqual(
                set(item),
                {
                    "schema_version",
                    "review_id",
                    "stratum",
                    "predicate_index",
                    "predicate_roleset",
                    "rows",
                },
            )
            self.assertEqual(
                set(item["rows"][0]),
                {"word", "predicate_marker", "raw_role_cell", "prepared_tag"},
            )
            predicate_row = item["rows"][item["predicate_index"]]
            self.assertNotEqual(predicate_row["predicate_marker"], "-")
            self.assertEqual(predicate_row["prepared_tag"], "B-V")
            self.assertIn("V", predicate_row["raw_role_cell"])
            self.assertIn("Invented", item_text)
            self.assertNotIn(
                "Invented", (first / REVIEW_MANIFEST_FILENAME).read_text()
            )
            self.assertNotIn(
                "Invented", (first / REVIEW_DECISIONS_FILENAME).read_text()
            )
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o700)
            for filename in (
                REVIEW_ITEMS_FILENAME,
                REVIEW_DECISIONS_FILENAME,
                REVIEW_MANIFEST_FILENAME,
            ):
                self.assertEqual(
                    stat.S_IMODE((first / filename).stat().st_mode),
                    0o600,
                )

    def test_rejects_nonignored_tracked_and_symlink_escape_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._new_repository(Path(directory))
            archive, prepared, digest, size = self._prepared_synthetic_data(root)
            authorization = self._authorization()

            with self._patched_archive_pin(digest, size):
                with self.assertRaisesRegex(ReviewPathError, "data/review"):
                    create_manual_review_package(
                        archive,
                        prepared,
                        root / "docs" / "review",
                        repository_root=root,
                        authorization=authorization,
                    )

                tracked = root / "data" / "review" / "tracked"
                tracked.mkdir(parents=True)
                (tracked / "tracked.txt").write_text("no corpus", encoding="utf-8")
                subprocess.run(
                    ["git", "add", "-f", "data/review/tracked/tracked.txt"],
                    cwd=root,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                with self.assertRaisesRegex(ReviewPathError, "tracked"):
                    create_manual_review_package(
                        archive,
                        prepared,
                        tracked,
                        repository_root=root,
                        authorization=authorization,
                    )

                outside = root / "outside"
                outside.mkdir()
                symlink = root / "data" / "review" / "escape"
                symlink.symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ReviewPathError, "data/review"):
                    create_manual_review_package(
                        archive,
                        prepared,
                        symlink / "package",
                        repository_root=root,
                        authorization=authorization,
                    )

    def test_rejects_unpinned_archive_and_prepared_raw_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._new_repository(Path(directory))
            archive, prepared, digest, size = self._prepared_synthetic_data(root)
            authorization = self._authorization()

            with self.assertRaisesRegex(ValueError, "pin"):
                create_manual_review_package(
                    archive,
                    prepared,
                    root / "data" / "review" / "wrong-pin",
                    repository_root=root,
                    authorization=authorization,
                )

            build = build_babysrl_dataset(
                archive,
                expected_sha256=digest,
                expected_size_bytes=size,
            )
            write_prepared_dataset(prepared, build.examples[1:])
            with self._patched_archive_pin(digest, size):
                with self.assertRaisesRegex(ReviewIntegrityError, "exactly match"):
                    create_manual_review_package(
                        archive,
                        prepared,
                        root / "data" / "review" / "mismatch",
                        repository_root=root,
                        authorization=authorization,
                    )

    def test_finalize_is_aggregate_only_and_passes_only_all_approved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._new_repository(Path(directory))
            archive, prepared, digest, size = self._prepared_synthetic_data(root)
            review = root / "data" / "review" / "approval"
            authorization = self._authorization()

            with self._patched_archive_pin(digest, size):
                create_manual_review_package(
                    archive,
                    prepared,
                    review,
                    repository_root=root,
                    authorization=authorization,
                    per_stratum=1,
                )
                pending = finalize_manual_review(
                    review,
                    repository_root=root,
                    authorization=authorization,
                )
                self.assertEqual(pending["status"], "hold")
                self.assertEqual(
                    pending["decision_counts"]["pending"],
                    pending["sample_size"],
                )

                self._set_all_decisions(review, decision="approve")
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    approved = finalize_manual_review(
                        review,
                        repository_root=root,
                        authorization=authorization,
                    )

            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(approved["status"], "pass")
            self.assertEqual(
                approved["decision_counts"]["approve"], approved["sample_size"]
            )
            aggregate_bytes = (review / REVIEW_AGGREGATE_FILENAME).read_bytes()
            for forbidden in (
                b"Invented",
                b'"rows"',
                b'"word"',
                b'"prepared_tag"',
                b'"predicate_roleset"',
                b'"review_id"',
            ):
                self.assertNotIn(forbidden, aggregate_bytes)
            self.assertEqual(
                stat.S_IMODE((review / REVIEW_AGGREGATE_FILENAME).stat().st_mode),
                0o600,
            )

    def test_reject_and_uncertain_require_structured_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._new_repository(Path(directory))
            archive, prepared, digest, size = self._prepared_synthetic_data(root)
            review = root / "data" / "review" / "decision-codes"
            authorization = self._authorization()

            with self._patched_archive_pin(digest, size):
                create_manual_review_package(
                    archive,
                    prepared,
                    review,
                    repository_root=root,
                    authorization=authorization,
                    per_stratum=1,
                )
                self._set_first_decision(
                    review,
                    decision="reject",
                    reason_codes=[],
                )
                with self.assertRaisesRegex(ReviewFormatError, "require a reason"):
                    finalize_manual_review(
                        review,
                        repository_root=root,
                        authorization=authorization,
                    )

                self._set_first_decision(
                    review,
                    decision="reject",
                    reason_codes=["argument_boundary_error"],
                )
                failed = finalize_manual_review(
                    review,
                    repository_root=root,
                    authorization=authorization,
                )

            self.assertEqual(failed["status"], "fail")
            self.assertEqual(failed["reason_counts"]["argument_boundary_error"], 1)

    def test_finalize_detects_private_item_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._new_repository(Path(directory))
            archive, prepared, digest, size = self._prepared_synthetic_data(root)
            review = root / "data" / "review" / "tampered"
            authorization = self._authorization()

            with self._patched_archive_pin(digest, size):
                create_manual_review_package(
                    archive,
                    prepared,
                    review,
                    repository_root=root,
                    authorization=authorization,
                    per_stratum=1,
                )
                item_path = review / REVIEW_ITEMS_FILENAME
                item_path.write_bytes(item_path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ReviewIntegrityError, "manifest digest"):
                    finalize_manual_review(
                        review,
                        repository_root=root,
                        authorization=authorization,
                    )

    def test_cli_never_prints_sample_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._new_repository(Path(directory))
            archive, prepared, digest, size = self._prepared_synthetic_data(root)
            review = root / "data" / "review" / "cli"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                self._patched_archive_pin(digest, size),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "--repository-root",
                        str(root),
                        "create",
                        str(archive),
                        str(prepared),
                        str(review),
                        "--per-stratum",
                        "1",
                        "--confirm-talkbank-registration",
                        "--confirm-talkbank-ground-rules",
                        "--rules-accepted-on",
                        "2026-08-10",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr.getvalue(), "")
            safe_receipt = json.loads(stdout.getvalue())
            self.assertEqual(safe_receipt["status"], "review_pending")
            self.assertNotIn("Invented", stdout.getvalue())
            self.assertNotIn("prepared_tag", stdout.getvalue())

    def test_cli_fails_on_access_confirmation_before_paths(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main(
                [
                    "create",
                    "/must/not/read/BabySRL.zip",
                    "/must/not/read/prepared",
                    "/must/not/write/review",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("TalkBank", stderr.getvalue())
        self.assertNotIn("review_items", stderr.getvalue())

    @staticmethod
    def _new_repository(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", "-q"],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        (root / ".gitignore").write_text(
            "data/raw/\ndata/processed/\ndata/review/\n",
            encoding="utf-8",
        )
        return root

    @classmethod
    def _prepared_synthetic_data(
        cls,
        root: Path,
    ) -> tuple[Path, Path, str, int]:
        archive = root / "data" / "raw" / "BabySRL.zip"
        archive.parent.mkdir(parents=True)
        with ZipFile(archive, "w") as target:
            for child in ("Adam", "Eve", "Sarah"):
                for index in range(1, 11):
                    basename = (
                        f"sarah{index:03d}"
                        if child == "Sarah"
                        else f"{child.lower()}{index:02d}"
                    )
                    target.writestr(
                        f"BabySRL/{child}/{basename}.srl.cha",
                        cls._synthetic_chat(child, index),
                    )
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        size = archive.stat().st_size
        build = build_babysrl_dataset(
            archive,
            expected_sha256=digest,
            expected_size_bytes=size,
        )
        prepared = root / "data" / "processed" / "babysrl"
        write_prepared_dataset(prepared, build.examples)
        return archive, prepared, digest, size

    @staticmethod
    def _synthetic_chat(child: str, index: int) -> str:
        token = f"{child}{index:03d}Invented"
        lemma = f"craft{child.lower()}{index:03d}"
        shape = index % 4
        if shape == 0:
            rows = (
                f"%srl: {token}Subject - (A0*)",
                f"%srl: {token}Builds {lemma} (V*)",
                f"%srl: {token}Object - (A1*)",
            )
        elif shape == 1:
            rows = (
                f"%srl: {token}Subject - (A0*)",
                f"%srl: {token}Builds {lemma} (V*)",
                f"%srl: {token}Bright - (A1*",
                f"%srl: {token}Object - *)",
            )
        elif shape == 2:
            rows = (
                f"%srl: {token}Subject - (A0*)",
                f"%srl: {token}Sets - (V*)",
                f"%srl: {token}Up {lemma} (C-V*)",
                f"%srl: {token}Object - (A1*)",
            )
        else:
            rows = (
                f"%srl: {token}Subject - (A0*)",
                f"%srl: {token}Builds {lemma} (V*)",
                f"%srl: {token}First - (A1*)",
                f"%srl: {token}Quietly - *",
                f"%srl: {token}Second - (A1*)",
            )
        return "\n".join(("*MOT: wholly invented fixture", *rows))

    @staticmethod
    def _authorization() -> TalkBankReviewAuthorization:
        return TalkBankReviewAuthorization(
            registration_confirmed=True,
            ground_rules_confirmed=True,
            accepted_on="2026-08-10",
        )

    @staticmethod
    @contextlib.contextmanager
    def _patched_archive_pin(digest: str, size: int):
        with (
            mock.patch.object(manual_review, "BABYSRL_ARCHIVE_SHA256", digest),
            mock.patch.object(manual_review, "BABYSRL_ARCHIVE_SIZE_BYTES", size),
        ):
            yield

    @staticmethod
    def _set_all_decisions(review: Path, *, decision: str) -> None:
        path = review / REVIEW_DECISIONS_FILENAME
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["decision"] = decision
            row["reason_codes"] = []
        BabySRLManualReviewTests._write_decisions(path, rows)

    @staticmethod
    def _set_first_decision(
        review: Path,
        *,
        decision: str,
        reason_codes: list[str],
    ) -> None:
        path = review / REVIEW_DECISIONS_FILENAME
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["decision"] = decision
        rows[0]["reason_codes"] = reason_codes
        BabySRLManualReviewTests._write_decisions(path, rows)

    @staticmethod
    def _write_decisions(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    unittest.main()
