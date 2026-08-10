import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from semantic_action_extractor.srl.dataset_io import (
    DATASET_SCHEMA_VERSION,
    DatasetFormatError,
    compute_dataset_fingerprint,
    read_prepared_dataset,
    write_prepared_dataset,
)
from semantic_action_extractor.srl.example import (
    DatasetSplit,
    PreparedWordLevelSRLExample,
    WordLevelSRLExample,
)


_SPLIT_FILENAMES = ("train.jsonl", "development.jsonl", "test.jsonl")


class PreparedDatasetIOTests(unittest.TestCase):
    def test_round_trip_is_canonical_deterministic_and_utf8_stable(self) -> None:
        examples = self._dataset_examples()

        with (
            tempfile.TemporaryDirectory() as first_name,
            tempfile.TemporaryDirectory() as second_name,
        ):
            first = Path(first_name)
            second = Path(second_name)
            manifest = write_prepared_dataset(first, reversed(examples))
            write_prepared_dataset(second, examples)

            for filename in (*_SPLIT_FILENAMES, "manifest.json"):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )

            train_bytes = (first / "train.jsonl").read_bytes()
            self.assertTrue(train_bytes.endswith(b"\n"))
            self.assertNotIn(b"\r", train_bytes)
            self.assertIn("Zoë".encode("utf-8"), train_bytes)
            self.assertNotIn(b"Zo\\u00eb", train_bytes)

            loaded = read_prepared_dataset(first)

        self.assertEqual(
            [item.example_id for item in loaded.examples],
            ["train:send", "train:sign", "dev:review", "test:file"],
        )
        self.assertEqual(
            loaded.examples[0].words,
            ("Zoë", "sent", "and", "signed", "forms"),
        )
        self.assertEqual(
            dict(manifest.record_counts),
            {"train": 2, "development": 1, "test": 1},
        )
        self.assertEqual(loaded.manifest, manifest)

        expected_input = bytearray(DATASET_SCHEMA_VERSION.encode("utf-8") + b"\n")
        for filename in _SPLIT_FILENAMES:
            expected_input.extend(filename.encode("utf-8"))
            expected_input.extend(b"\0")
            expected_input.extend(manifest.file_sha256[filename].encode("ascii"))
            expected_input.extend(b"\n")
        self.assertEqual(
            manifest.dataset_fingerprint,
            hashlib.sha256(expected_input).hexdigest(),
        )

    def test_accepts_multiple_predicates_in_one_sentence_and_split(self) -> None:
        examples = tuple(
            item for item in self._dataset_examples() if item.split == "train"
        )

        with tempfile.TemporaryDirectory() as directory:
            write_prepared_dataset(directory, examples)
            loaded = read_prepared_dataset(directory)

        self.assertEqual(len(loaded.examples), 2)
        self.assertEqual({item.predicate_index for item in loaded.examples}, {1, 3})

    def test_accepts_only_final_examples(self) -> None:
        prepared = PreparedWordLevelSRLExample(
            example_id="prepared:send",
            document_id="prepared",
            sentence_id="prepared:0",
            words=("Mira", "sent"),
            predicate_index=1,
            tags=("B-ARG0", "B-V"),
            predicate_roleset="send.01",
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "final WordLevelSRLExample"):
                write_prepared_dataset(directory, [prepared])

    def test_rejects_duplicate_example_and_semantic_identities(self) -> None:
        first = self._example(
            example_id="one",
            document_id="doc-one",
            sentence_id="doc-one:0",
            split="train",
        )
        duplicate_id = self._example(
            example_id="one",
            document_id="doc-two",
            sentence_id="doc-two:0",
            split="train",
        )
        duplicate_semantics = self._example(
            example_id="two",
            document_id="doc-one",
            sentence_id="doc-one:0",
            split="train",
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DatasetFormatError, "duplicate example ID"):
                write_prepared_dataset(directory, [first, duplicate_id])
            with self.assertRaisesRegex(
                DatasetFormatError, "duplicate semantic identity"
            ):
                write_prepared_dataset(directory, [first, duplicate_semantics])

    def test_rejects_document_and_sentence_leakage_across_splits(self) -> None:
        train = self._example(
            example_id="train",
            document_id="shared-doc",
            sentence_id="shared-doc:0",
            split="train",
        )
        leaked_document = self._example(
            example_id="dev",
            document_id="shared-doc",
            sentence_id="shared-doc:1",
            split="development",
        )
        leaked_sentence = self._example(
            example_id="dev",
            document_id="other-doc",
            sentence_id="shared-doc:0",
            split="development",
        )
        leaked_sentence_text = self._example(
            example_id="dev",
            document_id="other-doc",
            sentence_id="other-doc:0",
            split="development",
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DatasetFormatError, "document leakage"):
                write_prepared_dataset(directory, [train, leaked_document])
            with self.assertRaisesRegex(DatasetFormatError, "sentence leakage"):
                write_prepared_dataset(directory, [train, leaked_sentence])
            with self.assertRaisesRegex(DatasetFormatError, "sentence-text leakage"):
                write_prepared_dataset(directory, [train, leaked_sentence_text])

    def test_reader_rejects_unknown_missing_and_duplicate_json_keys(self) -> None:
        mutations = {
            "unknown keys": lambda payload: {**payload, "source_path": "forbidden"},
            "missing keys": lambda payload: {
                key: value for key, value in payload.items() if key != "tags"
            },
            "duplicate JSON key": lambda payload: (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))[:-1]
                + ',"tags":["B-ARG0","B-V"]}'
            ),
        }

        for expected_error, mutate in mutations.items():
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as directory_name:
                    directory = Path(directory_name)
                    write_prepared_dataset(directory, [self._example()])
                    payload = json.loads((directory / "train.jsonl").read_text())
                    mutated = mutate(payload)
                    if isinstance(mutated, dict):
                        line = json.dumps(
                            mutated,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    else:
                        line = mutated
                    self._replace_split_bytes(
                        directory,
                        "train.jsonl",
                        (line + "\n").encode("utf-8"),
                    )

                    with self.assertRaisesRegex(DatasetFormatError, expected_error):
                        read_prepared_dataset(directory)

    def test_reader_rejects_malformed_json_and_noncanonical_newlines(self) -> None:
        cases = (
            (b"{not-json}\n", "malformed JSON"),
            (b"{}", "end with an LF"),
            (b"{}\r\n", "LF newlines"),
        )

        for contents, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as directory_name:
                    directory = Path(directory_name)
                    write_prepared_dataset(directory, [self._example()])
                    self._replace_split_bytes(directory, "train.jsonl", contents)

                    with self.assertRaisesRegex(DatasetFormatError, expected_error):
                        read_prepared_dataset(directory)

    def test_reader_rejects_split_mismatch_and_invalid_dataclass_record(self) -> None:
        mutations = (
            ({"split": "development"}, "declares split"),
            ({"tags": ["I-ARG0", "B-V"]}, "word 0"),
        )

        for updates, expected_error in mutations:
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as directory_name:
                    directory = Path(directory_name)
                    write_prepared_dataset(directory, [self._example()])
                    payload = json.loads((directory / "train.jsonl").read_text())
                    payload.update(updates)
                    contents = (
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                    self._replace_split_bytes(directory, "train.jsonl", contents)

                    with self.assertRaisesRegex(DatasetFormatError, expected_error):
                        read_prepared_dataset(directory)

    def test_reader_rejects_noncanonical_row_order(self) -> None:
        examples = tuple(
            item for item in self._dataset_examples() if item.split == "train"
        )

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_prepared_dataset(directory, examples)
            lines = (directory / "train.jsonl").read_bytes().splitlines(
                keepends=True
            )
            self._replace_split_bytes(
                directory,
                "train.jsonl",
                b"".join(reversed(lines)),
            )

            with self.assertRaisesRegex(DatasetFormatError, "ordered by example_id"):
                read_prepared_dataset(directory)

    def test_reader_verifies_file_digest_count_and_manifest_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_prepared_dataset(directory, [self._example()])
            (directory / "train.jsonl").write_bytes(b"")
            with self.assertRaisesRegex(DatasetFormatError, "SHA-256 mismatch"):
                read_prepared_dataset(directory)

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_prepared_dataset(directory, [self._example()])
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest["record_counts"]["train"] = 2
            self._write_manifest(directory, manifest)
            with self.assertRaisesRegex(DatasetFormatError, "record count mismatch"):
                read_prepared_dataset(directory)

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_prepared_dataset(directory, [self._example()])
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest["schema_version"] = "future/v2"
            self._write_manifest(directory, manifest)
            with self.assertRaisesRegex(
                DatasetFormatError, "unsupported schema_version"
            ):
                read_prepared_dataset(directory)

    @staticmethod
    def _replace_split_bytes(directory: Path, filename: str, contents: bytes) -> None:
        (directory / filename).write_bytes(contents)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["file_sha256"][filename] = hashlib.sha256(contents).hexdigest()
        manifest["dataset_fingerprint"] = compute_dataset_fingerprint(
            manifest["file_sha256"]
        )
        PreparedDatasetIOTests._write_manifest(directory, manifest)

    @staticmethod
    def _write_manifest(directory: Path, manifest: dict[str, object]) -> None:
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def _example(
        *,
        example_id: str = "train:send",
        document_id: str = "train-doc",
        sentence_id: str = "train-doc:0",
        split: DatasetSplit = "train",
    ) -> WordLevelSRLExample:
        return WordLevelSRLExample(
            example_id=example_id,
            document_id=document_id,
            sentence_id=sentence_id,
            split=split,
            words=("Mira", "sent"),
            predicate_index=1,
            tags=("B-ARG0", "B-V"),
            predicate_roleset="send.01",
        )

    @staticmethod
    def _dataset_examples() -> tuple[WordLevelSRLExample, ...]:
        words = ("Zoë", "sent", "and", "signed", "forms")
        return (
            WordLevelSRLExample(
                example_id="train:sign",
                document_id="train-doc",
                sentence_id="train-doc:0",
                split="train",
                words=words,
                predicate_index=3,
                tags=("B-ARG0", "O", "O", "B-V", "B-ARG1"),
                predicate_roleset="sign.01",
            ),
            WordLevelSRLExample(
                example_id="train:send",
                document_id="train-doc",
                sentence_id="train-doc:0",
                split="train",
                words=words,
                predicate_index=1,
                tags=("B-ARG0", "B-V", "O", "O", "B-ARG1"),
                predicate_roleset="send.01",
            ),
            WordLevelSRLExample(
                example_id="dev:review",
                document_id="dev-doc",
                sentence_id="dev-doc:0",
                split="development",
                words=("Noah", "reviewed"),
                predicate_index=1,
                tags=("B-ARG0", "B-V"),
                predicate_roleset="review.01",
            ),
            WordLevelSRLExample(
                example_id="test:file",
                document_id="test-doc",
                sentence_id="test-doc:0",
                split="test",
                words=("They", "filed"),
                predicate_index=1,
                tags=("B-ARG0", "B-V"),
                predicate_roleset="file.01",
            ),
        )


if __name__ == "__main__":
    unittest.main()
