import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from semantic_action_extractor.annotation_schema import (
    AnnotationProvenance,
    AnnotationRecord,
)
from semantic_action_extractor.datasets.common import (
    DatasetFormatError,
    canonicalize_tokens,
)
from semantic_action_extractor.datasets.io import (
    download_verified,
    extract_verified,
    sha256_file,
    verify_sha256,
)
from semantic_action_extractor.datasets.leakage import (
    DatasetPartition,
    build_training_quarantine,
    check_cross_partition_leakage,
    iter_after_document_quarantine,
    load_training_quarantine_report,
)
from semantic_action_extractor.datasets.manifest import (
    ManifestBuilder,
    SourceArtifactIdentity,
)
from semantic_action_extractor.datasets.prepare import _publish_pair
from semantic_action_extractor.datasets.registry import (
    ARTIFACTS,
    DatasetArtifact,
    get_artifact,
)


def _record(
    record_id: str,
    *,
    split: str,
    source_id: str,
    document_id: str | None,
    text_token: str = "Synthetic",
) -> AnnotationRecord:
    text, tokens = canonicalize_tokens((text_token, "."), label="tokens")
    return AnnotationRecord(
        text=text,
        tokens=tokens,
        provenance=AnnotationProvenance(
            dataset="synthetic",
            release="1",
            split=split,
            source_id=source_id,
            record_id=record_id,
            document_id=document_id,
        ),
    )


class RegistryAndIOTests(unittest.TestCase):
    def test_registry_contains_the_three_pinned_archives(self) -> None:
        self.assertEqual(
            set(ARTIFACTS),
            {"qa-srl-bank-2.1", "qa-srl-gold-standard", "qanom-2020"},
        )
        self.assertEqual(get_artifact("qa-srl-bank-2.1").release, "2.1")

    def test_checksum_failure_reports_expected_and_actual_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.bin"
            path.write_bytes(b"verified bytes")

            with self.assertRaisesRegex(DatasetFormatError, "checksum mismatch"):
                verify_sha256(path, "0" * 64)

    def test_download_overwrite_replaces_a_corrupt_cached_file(self) -> None:
        payload = b"verified replacement"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.bin"
            path.write_bytes(b"corrupt cache")
            artifact = DatasetArtifact(
                key="fixture",
                dataset="synthetic",
                release="1",
                url="https://example.invalid/archive.bin",
                sha256=hashlib.sha256(payload).hexdigest(),
                archive_format="zip",
            )

            with patch(
                "semantic_action_extractor.datasets.io.urlopen",
                return_value=io.BytesIO(payload),
            ):
                result = download_verified(artifact, path, overwrite=True)

            self.assertEqual(result.read_bytes(), payload)

    def test_verified_zip_extracts_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("dataset/example.txt", "synthetic")
            digest = sha256_file(archive)
            artifact = DatasetArtifact(
                key="fixture",
                dataset="synthetic",
                release="1",
                url="https://example.invalid/fixture.zip",
                sha256=digest,
                archive_format="zip",
            )

            output = extract_verified(artifact, archive, root / "output")

            self.assertEqual(
                (output / "dataset" / "example.txt").read_text(encoding="utf-8"),
                "synthetic",
            )

    def test_tar_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.tar"
            payload = b"escape"
            with tarfile.open(archive, "w") as handle:
                member = tarfile.TarInfo("../escape.txt")
                member.size = len(payload)
                handle.addfile(member, io.BytesIO(payload))
            artifact = DatasetArtifact(
                key="unsafe",
                dataset="synthetic",
                release="1",
                url="https://example.invalid/unsafe.tar",
                sha256=sha256_file(archive),
                archive_format="tar",
            )

            with self.assertRaisesRegex(DatasetFormatError, "unsafe archive member"):
                extract_verified(artifact, archive, root / "output")
            self.assertFalse((root / "escape.txt").exists())


class ManifestTests(unittest.TestCase):
    def _builder(self) -> ManifestBuilder:
        return ManifestBuilder(
            dataset="synthetic",
            release="1",
            split="train",
            adapter="synthetic-adapter",
            adapter_version="1",
            source_artifacts=(SourceArtifactIdentity("fixture", "a" * 64),),
        )

    def test_manifest_fingerprint_is_deterministic(self) -> None:
        record = _record(
            "record-1",
            split="train",
            source_id="source-1",
            document_id="document-1",
        )
        first = self._builder()
        second = self._builder()

        first.add(record)
        second.add(record)

        self.assertEqual(
            first.finish().record_fingerprint,
            second.finish().record_fingerprint,
        )
        self.assertEqual(first.finish().counts["records"], 1)

    def test_manifest_rejects_duplicate_record_ids(self) -> None:
        record = _record(
            "record-1",
            split="train",
            source_id="source-1",
            document_id="document-1",
        )
        builder = self._builder()
        builder.add(record)

        with self.assertRaisesRegex(DatasetFormatError, "duplicate adapted"):
            builder.add(record)

    def test_manifest_omits_zero_valued_anomalies(self) -> None:
        record = _record(
            "record-1",
            split="train",
            source_id="source-1",
            document_id="document-1",
        )
        builder = self._builder()

        builder.add(record)

        self.assertEqual(builder.finish().anomalies, {})

    def test_manifest_rejects_invalid_source_identities_and_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "name cannot be empty"):
            SourceArtifactIdentity("", "a" * 64)
        with self.assertRaisesRegex(ValueError, "64 lowercase"):
            SourceArtifactIdentity("fixture", "not-a-checksum")
        with self.assertRaisesRegex(ValueError, "at least one"):
            ManifestBuilder(
                dataset="synthetic",
                release="1",
                split="train",
                adapter="synthetic-adapter",
                adapter_version="1",
                source_artifacts=(),
            )

        builder = self._builder()
        with self.assertRaisesRegex(DatasetFormatError, "non-negative integers"):
            builder.add_exclusion("quarantined_records", True)

    def test_paired_publication_restores_previous_files_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "adapted.jsonl"
            manifest = root / "manifest.json"
            staged_output = root / ".adapted.partial"
            missing_staged_manifest = root / ".missing-manifest.partial"
            output.write_text("old data\n", encoding="utf-8")
            manifest.write_text("old manifest\n", encoding="utf-8")
            staged_output.write_text("new data\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                _publish_pair(
                    staged_output,
                    output,
                    missing_staged_manifest,
                    manifest,
                    transaction_id="fixture",
                )

            self.assertEqual(output.read_text(encoding="utf-8"), "old data\n")
            self.assertEqual(
                manifest.read_text(encoding="utf-8"),
                "old manifest\n",
            )

    def test_manifest_write_has_stable_sorted_json(self) -> None:
        record = _record(
            "record-1",
            split="train",
            source_id="source-1",
            document_id="document-1",
        )
        builder = self._builder()
        builder.add(record)
        with tempfile.TemporaryDirectory() as directory:
            path = builder.finish().write(Path(directory) / "manifest.json")
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["counts"]["records"], 1)
        self.assertEqual(payload["schema_version"], "0.3.0")


class LeakageTests(unittest.TestCase):
    def test_cross_role_document_overlap_is_reported(self) -> None:
        train = DatasetPartition(
            name="qa-srl-train",
            role="train",
            records=(
                _record(
                    "train-1",
                    split="train",
                    source_id="train-source",
                    document_id="shared-document",
                ),
            ),
        )
        development = DatasetPartition(
            name="qanom-development",
            role="development",
            records=(
                _record(
                    "dev-1",
                    split="development",
                    source_id="dev-source",
                    document_id="shared-document",
                    text_token="Different",
                ),
            ),
        )

        report = check_cross_partition_leakage((train, development))

        self.assertFalse(report.is_clean)
        self.assertIn(
            "document_id",
            {finding.identity_type for finding in report.findings},
        )
        with self.assertRaisesRegex(DatasetFormatError, "leakage detected"):
            report.raise_if_found()

    def test_same_role_cross_task_overlap_is_permitted(self) -> None:
        shared = _record(
            "dev-1",
            split="development",
            source_id="shared-source",
            document_id="shared-document",
        )
        report = check_cross_partition_leakage(
            (
                DatasetPartition("qa-srl-development", "development", (shared,)),
                DatasetPartition("qanom-development", "development", (shared,)),
            )
        )

        self.assertTrue(report.is_clean)

    def test_missing_document_id_fails_closed(self) -> None:
        partition = DatasetPartition(
            name="train",
            role="train",
            records=(
                _record(
                    "train-1",
                    split="train",
                    source_id="source",
                    document_id=None,
                ),
            ),
        )

        with self.assertRaisesRegex(DatasetFormatError, "no document ID"):
            check_cross_partition_leakage((partition,))

    def test_exact_evaluation_text_quarantines_the_full_training_document(self) -> None:
        trigger = _record(
            "train-trigger",
            split="train",
            source_id="train-source-1",
            document_id="training-document",
            text_token="Copied",
        )
        sibling = _record(
            "train-sibling",
            split="train",
            source_id="train-source-2",
            document_id="training-document",
            text_token="Sibling",
        )
        clean = _record(
            "train-clean",
            split="train",
            source_id="train-source-3",
            document_id="clean-document",
            text_token="Clean",
        )
        development = _record(
            "dev-copy",
            split="dev",
            source_id="dev-source",
            document_id="different-evaluation-document",
            text_token="Copied",
        )

        report = build_training_quarantine(
            (
                DatasetPartition(
                    "training",
                    "train",
                    iter((trigger, sibling, clean)),
                ),
            ),
            (
                DatasetPartition(
                    "development",
                    "development",
                    iter((development,)),
                ),
            ),
        )
        excluded: list[str] = []
        retained = tuple(
            iter_after_document_quarantine(
                (trigger, sibling, clean),
                report,
                on_exclude=lambda record: excluded.append(
                    record.provenance.record_id
                ),
            )
        )

        self.assertEqual(
            report.contaminated_document_ids,
            ("training-document",),
        )
        self.assertEqual([item.reason for item in report.triggers], ["exact_text_match"])
        self.assertEqual(
            [record.provenance.record_id for record in retained],
            ["train-clean"],
        )
        self.assertEqual(excluded, ["train-trigger", "train-sibling"])

        with tempfile.TemporaryDirectory() as directory:
            path = report.write(Path(directory) / "quarantine.json")
            loaded = load_training_quarantine_report(path)

        self.assertEqual(loaded.to_dict(), report.to_dict())
        self.assertEqual(loaded.training_record_counts["training"], 3)
        self.assertEqual(loaded.quarantined_record_counts["training"], 2)

    def test_development_test_overlap_fails_before_training_check(self) -> None:
        development = _record(
            "dev-copy",
            split="dev",
            source_id="dev-source",
            document_id="dev-document",
            text_token="Copied",
        )
        test = _record(
            "test-copy",
            split="test",
            source_id="test-source",
            document_id="test-document",
            text_token="Copied",
        )

        with self.assertRaisesRegex(DatasetFormatError, "development/test"):
            build_training_quarantine(
                (),
                (
                    DatasetPartition("development", "development", (development,)),
                    DatasetPartition("test", "test", (test,)),
                ),
            )


if __name__ == "__main__":
    unittest.main()
