import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from semantic_action_extractor.srl.dataset_io import read_prepared_dataset
from semantic_action_extractor.srl.ewt import (
    EWT_PROVENANCE_FILENAME,
    EWTPreparationError,
    EWTSourceError,
    build_ewt_dataset,
    main,
    prepare_ewt_dataset,
)


SplitName = str
Sentence = tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str, str, tuple[str, ...]], ...],
]

_REPOSITORY = Path(__file__).resolve().parents[1]
_REAL_PROPBANK = _REPOSITORY / "data" / "raw" / "ewt_sources" / "propbank-release"
_REAL_UD_EWT = _REPOSITORY / "data" / "raw" / "ewt_sources" / "UD_English-EWT"
_REAL_SOURCES_AVAILABLE = _REAL_PROPBANK.is_dir() and _REAL_UD_EWT.is_dir()


def _single_predicate_sentence(
    words: tuple[str, ...],
    xpos: tuple[str, ...],
    *,
    predicate_index: int = 1,
    predicate_xpos: str | None = None,
    roleset: str = "craft.01",
) -> Sentence:
    rows = []
    for index, pos in enumerate(xpos):
        if index == predicate_index:
            rows.append((predicate_xpos or pos, "craft", roleset, ("(V*)",)))
        elif index < predicate_index:
            rows.append((pos, "-", "-", ("(ARG0*)",)))
        else:
            rows.append((pos, "-", "-", ("(ARG1*)",)))
    return words, xpos, tuple(rows)


def _multiword_primary_v_sentence(
    *, subject: str = "They", object_word: str = "files"
) -> Sentence:
    return (
        (subject, "cross", "reference", "linked", object_word),
        ("PRP", "VB", "VB", "VBN", "NNS"),
        (
            ("PRP", "-", "-", ("(ARG0*)",)),
            ("VB", "-", "-", ("(V*",)),
            ("VB", "crossreference", "crossreference.01", ("*)",)),
            ("VBN", "-", "-", ("(C-V*)",)),
            ("NNS", "-", "-", ("(ARG1-DSP*)",)),
        ),
    )


def _same_anchor_two_column_sentence(
    *,
    subject: str,
    conflicting_targets: bool,
) -> Sentence:
    second_subject_cell = (
        "(ARGM-TMP*)" if conflicting_targets else "(ARG0*)"
    )
    return (
        (subject, "cross", "reference"),
        ("NNS", "VB", "VB"),
        (
            ("NNS", "-", "-", ("(ARG0*)", second_subject_cell)),
            ("VB", "cross", "cross.01", ("(V*", "(V*")),
            ("VB", "reference", "reference.01", ("*)", "*)")),
        ),
    )


def _commit(root: Path) -> str:
    subprocess.run(("git", "init", "-q", root), check=True)
    subprocess.run(("git", "-C", root, "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            root,
            "-c",
            "user.name=Invented Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "invented fixture",
        ),
        check=True,
    )
    return subprocess.run(
        ("git", "-C", root, "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_fixture(
    base: Path,
    documents: dict[SplitName, dict[str, tuple[Sentence, ...]]],
) -> tuple[Path, Path, str, str]:
    propbank = base / "propbank-release"
    ud = base / "UD_English-EWT"
    skeleton_directory = propbank / "data" / "google" / "ewt" / "weblog" / "00"
    evaluation_directory = propbank / "docs" / "evaluation"
    skeleton_directory.mkdir(parents=True)
    evaluation_directory.mkdir(parents=True)
    ud.mkdir()

    pb_split_names = {
        "train": "train",
        "development": "dev",
        "test": "test",
    }
    ud_split_names = {
        "train": "train",
        "development": "dev",
        "test": "test",
    }
    for split in ("train", "development", "test"):
        document_rows = documents[split]
        split_lines = []
        conllu_blocks = []
        for document_name, sentences in document_rows.items():
            suffix = ".xml.conllu" if split == "development" else ".xml"
            split_lines.append(f"weblog/00/{document_name}{suffix}")
            raw_document = f"google/ewt/weblog/00/{document_name}.xml"
            skeleton_lines = []
            conllu_blocks.append(
                f"# newdoc id = weblog-{document_name}"
            )
            for sentence_index, (words, ud_xpos, rows) in enumerate(sentences):
                for token_index, (pos, lemma, roleset, cells) in enumerate(rows):
                    skeleton_lines.append(
                        " ".join(
                            (
                                raw_document,
                                str(sentence_index),
                                str(token_index),
                                "[WORD]",
                                pos,
                                "*",
                                lemma,
                                roleset,
                                *cells,
                            )
                        )
                    )
                skeleton_lines.append("")
                conllu_blocks.append(
                    f"# sent_id = weblog-{document_name}-{sentence_index + 1:04d}"
                )
                for token_index, (word, pos) in enumerate(
                    zip(words, ud_xpos, strict=True), start=1
                ):
                    conllu_blocks.append(
                        "\t".join(
                            (
                                str(token_index),
                                word,
                                "_",
                                "X",
                                pos,
                                "_",
                                "0",
                                "dep",
                                "_",
                                "_",
                            )
                        )
                    )
                conllu_blocks.append("")
            (skeleton_directory / f"{document_name}.xml.gold_skel").write_text(
                "\n".join(skeleton_lines), encoding="utf-8"
            )
        (evaluation_directory / f"ewt.{pb_split_names[split]}.txt").write_text(
            "\n".join(split_lines) + "\n", encoding="utf-8"
        )
        (ud / f"en_ewt-ud-{ud_split_names[split]}.conllu").write_text(
            "\n".join(conllu_blocks) + "\n", encoding="utf-8"
        )

    propbank_commit = _commit(propbank)
    ud_commit = _commit(ud)
    return propbank, ud, propbank_commit, ud_commit


def _minimal_documents() -> dict[SplitName, dict[str, tuple[Sentence, ...]]]:
    return {
        "train": {
            "invented_train": (
                _single_predicate_sentence(
                    ("Mira", "crafts", "kites"), ("NNP", "VBZ", "NNS")
                ),
            )
        },
        "development": {
            "invented_dev": (
                _single_predicate_sentence(
                    ("Niko", "sorts", "tiles"), ("NNP", "VBZ", "NNS")
                ),
            )
        },
        "test": {
            "invented_test": (
                _single_predicate_sentence(
                    ("Oona", "paints", "signs"), ("NNP", "VBZ", "NNS")
                ),
            )
        },
    }


class EWTDatasetTests(unittest.TestCase):
    @unittest.skipUnless(
        _REAL_SOURCES_AVAILABLE,
        "ignored pinned EWT source checkouts are not available",
    )
    def test_pinned_real_sources_reproduce_aggregate_gate(self) -> None:
        build = build_ewt_dataset(_REAL_PROPBANK, _REAL_UD_EWT)

        self.assertEqual(build.report["gate"]["status"], "pass")
        self.assertEqual(
            build.report["propbank_skeleton"]["documents"], 1_145
        )
        self.assertEqual(
            build.report["propbank_skeleton"]["all_predicate_columns"],
            50_262,
        )
        self.assertEqual(
            build.report["propbank_skeleton"]["verbal_predicate_columns"],
            38_639,
        )
        self.assertEqual(
            build.report["conversion"]["accepted_before_leakage_policy"],
            38_635,
        )
        self.assertEqual(
            build.report["split"]["final_examples"],
            {"train": 31_101, "development": 3_775, "test": 3_610},
        )
        self.assertEqual(build.report["label_vocabulary"]["size"], 111)
        self.assertEqual(
            build.report["label_vocabulary"][
                "development_test_compatibility"
            ],
            "pass",
        )

        ud_source = (_REAL_UD_EWT / "en_ewt-ud-train.conllu").read_text(
            encoding="utf-8"
        )
        sample_document = next(
            line.removeprefix("# newdoc id = ")
            for line in ud_source.splitlines()
            if line.startswith("# newdoc id = ")
        )
        sample_word = next(
            fields[1]
            for line in ud_source.splitlines()
            if not line.startswith("#") and "\t" in line
            if (fields := line.split("\t"))[0].isdecimal()
            if len(fields[1]) >= 12
        )
        serialized = json.dumps(build.report, ensure_ascii=True, sort_keys=True)
        self.assertNotIn(json.dumps(sample_document), serialized)
        self.assertNotIn(json.dumps(sample_word), serialized)

    def test_joins_converts_primary_v_and_excludes_mismatch_and_leakage(self) -> None:
        shared = _single_predicate_sentence(
            ("Shared", "teams", "build"),
            ("JJ", "NNS", "VB"),
            predicate_index=2,
        )
        mismatch = (
            ("Width", "differs"),
            ("NN", "VBZ"),
            (
                ("NN", "-", "-", ("(ARG0*)",)),
                ("VBZ", "differ", "differ.01", ("(V*)",)),
                ("NNS", "-", "-", ("(ARG1*)",)),
            ),
        )
        nonverbal = _single_predicate_sentence(
            ("Their", "plan", "works"),
            ("PRP$", "NN", "VBZ"),
            predicate_xpos="NN",
            roleset="plan.01",
        )
        documents = {
            "train": {
                "invented_train": (
                    _multiword_primary_v_sentence(
                        subject="We", object_word="records"
                    ),
                    shared,
                    mismatch,
                    nonverbal,
                )
            },
            "development": {
                "invented_dev": (
                    _single_predicate_sentence(
                        ("Niko", "sorts", "tiles"),
                        ("NNP", "VBZ", "NNS"),
                    ),
                    shared,
                )
            },
            "test": {"invented_test": (_multiword_primary_v_sentence(),)},
        }
        with tempfile.TemporaryDirectory() as directory:
            propbank, ud, propbank_commit, ud_commit = _write_fixture(
                Path(directory), documents
            )
            build = build_ewt_dataset(
                propbank,
                ud,
                expected_propbank_commit=propbank_commit,
                expected_ud_commit=ud_commit,
                minimum_verbal_predicate_coverage=0.8,
            )

        self.assertEqual(len(build.examples), 3)
        self.assertEqual(
            [example.split for example in build.examples],
            ["development", "test", "train"],
        )
        test_example = next(
            example for example in build.examples if example.split == "test"
        )
        self.assertEqual(test_example.predicate_index, 1)
        self.assertEqual(test_example.predicate_roleset, "crossreference.01")
        self.assertEqual(
            test_example.tags,
            ("B-ARG0", "B-V", "I-V", "B-C-V", "B-ARG1"),
        )
        self.assertEqual(
            build.report["propbank_skeleton"]["verbal_predicate_columns"], 6
        )
        self.assertEqual(
            build.report["propbank_skeleton"]["nonverbal_predicate_columns"], 1
        )
        self.assertEqual(
            build.report["propbank_skeleton"]["multiword_primary_v_columns"], 2
        )
        self.assertEqual(
            build.report["propbank_skeleton"][
                "metadata_primary_anchor_differences"
            ],
            2,
        )
        self.assertEqual(build.report["join"]["token_count_mismatches"], 1)
        self.assertEqual(
            build.report["conversion"]["rejection_reasons"],
            {"token_count_mismatch": 1},
        )
        self.assertEqual(
            build.report["split"]["cross_split_word_sequence_groups"], 1
        )
        self.assertEqual(
            build.report["split"]["cross_split_example_exclusions"],
            {"train": 1, "development": 1, "test": 0},
        )
        self.assertEqual(build.report["gate"]["status"], "pass")
        serialized = json.dumps(build.report)
        self.assertNotIn("Mira", serialized)
        self.assertNotIn("crossreference.01", serialized)

    def test_excludes_conflicting_inputs_and_deduplicates_only_eval_exactness(
        self,
    ) -> None:
        conflict = _single_predicate_sentence(
            ("Inventors", "shape", "clay"), ("NNS", "VBP", "NN")
        )
        words, xpos, rows = conflict
        conflicting_rows = list(rows)
        first_pos, first_lemma, first_roleset, _ = conflicting_rows[0]
        conflicting_rows[0] = (
            first_pos,
            first_lemma,
            first_roleset,
            ("(ARGM-TMP*)",),
        )
        conflicting = (words, xpos, tuple(conflicting_rows))
        development_conflict = _single_predicate_sentence(
            ("Analysts", "shape", "clay"), ("NNS", "VBP", "NN")
        )
        dev_words, dev_xpos, dev_rows = development_conflict
        dev_conflicting_rows = list(dev_rows)
        dev_pos, dev_lemma, dev_roleset, _ = dev_conflicting_rows[0]
        dev_conflicting_rows[0] = (
            dev_pos,
            dev_lemma,
            dev_roleset,
            ("(ARGM-TMP*)",),
        )
        development_conflicting = (
            dev_words,
            dev_xpos,
            tuple(dev_conflicting_rows),
        )
        train_exact = _single_predicate_sentence(
            ("Builders", "stack", "blocks"), ("NNS", "VBP", "NNS")
        )
        development_exact = _single_predicate_sentence(
            ("Reviewers", "check", "plans"), ("NNS", "VBP", "NNS")
        )
        test_exact = _single_predicate_sentence(
            ("Painters", "mix", "colors"), ("NNS", "VBP", "NNS")
        )
        documents = {
            "train": {
                "invented_train": (
                    _single_predicate_sentence(
                        ("Mira", "crafts", "kites"),
                        ("NNP", "VBZ", "NNS"),
                    ),
                    development_conflict,
                    development_conflicting,
                    train_exact,
                    train_exact,
                )
            },
            "development": {
                "invented_dev": (
                    _single_predicate_sentence(
                        ("Niko", "sorts", "tiles"),
                        ("NNP", "VBZ", "NNS"),
                    ),
                    conflict,
                    conflicting,
                    development_exact,
                    development_exact,
                )
            },
            "test": {
                "invented_test": (
                    _single_predicate_sentence(
                        ("Oona", "paints", "signs"),
                        ("NNP", "VBZ", "NNS"),
                    ),
                    test_exact,
                    test_exact,
                )
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            propbank, ud, propbank_commit, ud_commit = _write_fixture(
                Path(directory), documents
            )
            build = build_ewt_dataset(
                propbank,
                ud,
                expected_propbank_commit=propbank_commit,
                expected_ud_commit=ud_commit,
            )

        self.assertEqual(
            build.report["split"]["conflicting_identical_input_groups"],
            {"train": 1, "development": 1, "test": 0},
        )
        self.assertEqual(
            build.report["split"]["conflicting_identical_input_exclusions"],
            {"train": 2, "development": 2, "test": 0},
        )
        self.assertEqual(
            build.report["split"][
                "dev_test_cross_source_exact_semantic_duplicates_removed"
            ],
            {"train": 0, "development": 1, "test": 1},
        )
        self.assertEqual(
            build.report["split"]["final_examples"],
            {"train": 3, "development": 2, "test": 2},
        )
        train_exact_examples = [
            example
            for example in build.examples
            if example.split == "train" and example.words == train_exact[0]
        ]
        self.assertEqual(len(train_exact_examples), 2)

    def test_same_anchor_conflicts_exclude_all_and_identical_targets_dedupe(
        self,
    ) -> None:
        train_frequency = _single_predicate_sentence(
            ("Builders", "stack", "blocks"), ("NNS", "VBP", "NNS")
        )
        documents = {
            "train": {
                "invented_train": (
                    _single_predicate_sentence(
                        ("Mira", "crafts", "kites"),
                        ("NNP", "VBZ", "NNS"),
                    ),
                    _same_anchor_two_column_sentence(
                        subject="Conflicts",
                        conflicting_targets=True,
                    ),
                    _same_anchor_two_column_sentence(
                        subject="Identicals",
                        conflicting_targets=False,
                    ),
                    train_frequency,
                    train_frequency,
                )
            },
            "development": {
                "invented_dev": (
                    _single_predicate_sentence(
                        ("Niko", "sorts", "tiles"),
                        ("NNP", "VBZ", "NNS"),
                    ),
                )
            },
            "test": {
                "invented_test": (
                    _single_predicate_sentence(
                        ("Oona", "paints", "signs"),
                        ("NNP", "VBZ", "NNS"),
                    ),
                )
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            propbank, ud, propbank_commit, ud_commit = _write_fixture(
                base, documents
            )
            build = build_ewt_dataset(
                propbank,
                ud,
                expected_propbank_commit=propbank_commit,
                expected_ud_commit=ud_commit,
            )
            prepared = base / "prepared"
            receipt = prepare_ewt_dataset(
                propbank,
                ud,
                prepared,
                expected_propbank_commit=propbank_commit,
                expected_ud_commit=ud_commit,
                output_path_policy=lambda path: None,
            )
            loaded = read_prepared_dataset(prepared)

        self.assertEqual(
            build.report["split"]["conflicting_identical_input_groups"],
            {"train": 1, "development": 0, "test": 0},
        )
        self.assertEqual(
            build.report["split"]["conflicting_identical_input_exclusions"],
            {"train": 2, "development": 0, "test": 0},
        )
        self.assertEqual(
            build.report["split"][
                "same_source_identical_target_groups_deduplicated"
            ],
            {"train": 1, "development": 0, "test": 0},
        )
        self.assertEqual(
            build.report["split"][
                "same_source_identical_target_duplicates_removed"
            ],
            {"train": 1, "development": 0, "test": 0},
        )
        self.assertEqual(
            build.report["split"]["final_examples"],
            {"train": 4, "development": 1, "test": 1},
        )
        self.assertFalse(
            any(example.words[0] == "Conflicts" for example in build.examples)
        )
        identicals = tuple(
            example
            for example in build.examples
            if example.words[0] == "Identicals"
        )
        self.assertEqual(len(identicals), 1)
        self.assertEqual(identicals[0].predicate_roleset, "cross.01")
        self.assertEqual(identicals[0].tags, ("B-ARG0", "B-V", "I-V"))
        retained_frequency = tuple(
            example
            for example in build.examples
            if example.words == train_frequency[0]
        )
        self.assertEqual(len(retained_frequency), 2)
        self.assertEqual(receipt.manifest.record_counts["train"], 4)
        self.assertEqual(
            sorted(example.example_id for example in loaded.examples),
            sorted(example.example_id for example in build.examples),
        )

    def test_prepares_canonical_splits_and_pinned_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            propbank, ud, propbank_commit, ud_commit = _write_fixture(
                base, _minimal_documents()
            )
            first_output = base / "first"
            second_output = base / "second"
            receipt = prepare_ewt_dataset(
                propbank,
                ud,
                first_output,
                expected_propbank_commit=propbank_commit,
                expected_ud_commit=ud_commit,
                output_path_policy=lambda path: None,
            )
            prepare_ewt_dataset(
                propbank,
                ud,
                second_output,
                expected_propbank_commit=propbank_commit,
                expected_ud_commit=ud_commit,
                output_path_policy=lambda path: None,
            )
            loaded = read_prepared_dataset(first_output)
            provenance_bytes = (first_output / EWT_PROVENANCE_FILENAME).read_bytes()
            provenance = json.loads(provenance_bytes)
            for filename in (
                "train.jsonl",
                "development.jsonl",
                "test.jsonl",
                "manifest.json",
                EWT_PROVENANCE_FILENAME,
            ):
                self.assertEqual(
                    (first_output / filename).read_bytes(),
                    (second_output / filename).read_bytes(),
                )

        self.assertEqual(len(loaded.examples), 3)
        self.assertEqual(receipt.manifest, loaded.manifest)
        self.assertEqual(
            receipt.provenance_sha256,
            hashlib.sha256(provenance_bytes).hexdigest(),
        )
        self.assertEqual(
            provenance["dataset"]["dataset_fingerprint"],
            loaded.manifest.dataset_fingerprint,
        )
        self.assertEqual(
            provenance["audit"]["source"]["propbank_release_commit"],
            propbank_commit,
        )

    def test_prepare_cli_emits_aggregate_receipt_with_artifact_identities(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            propbank, ud, propbank_commit, ud_commit = _write_fixture(
                base, _minimal_documents()
            )
            receipt = prepare_ewt_dataset(
                propbank,
                ud,
                base / "actual",
                expected_propbank_commit=propbank_commit,
                expected_ud_commit=ud_commit,
                output_path_policy=lambda path: None,
            )
            independently_loaded = read_prepared_dataset(base / "actual")
            provenance_bytes = (
                base / "actual" / EWT_PROVENANCE_FILENAME
            ).read_bytes()
            stdout = io.StringIO()
            with (
                patch(
                    "semantic_action_extractor.srl.ewt."
                    "_require_ignored_output_path"
                ),
                patch(
                    "semantic_action_extractor.srl.ewt.prepare_ewt_dataset",
                    return_value=receipt,
                ),
                redirect_stdout(stdout),
            ):
                status = main(
                    (
                        str(propbank),
                        str(ud),
                        "--output-directory",
                        str(base / "pretend"),
                    )
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(
            payload["prepared_dataset"]["dataset_fingerprint"],
            independently_loaded.manifest.dataset_fingerprint,
        )
        self.assertEqual(
            payload["prepared_dataset"]["record_counts"],
            {"train": 1, "development": 1, "test": 1},
        )
        self.assertEqual(
            payload["provenance"],
            {
                "filename": EWT_PROVENANCE_FILENAME,
                "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            },
        )
        self.assertNotIn(str(propbank), stdout.getvalue())
        self.assertNotIn("Mira", stdout.getvalue())

    def test_rejects_wrong_revision_and_relevant_dirty_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            propbank, ud, propbank_commit, ud_commit = _write_fixture(
                base, _minimal_documents()
            )
            with self.assertRaisesRegex(EWTSourceError, "immutable source pin"):
                build_ewt_dataset(
                    propbank,
                    ud,
                    expected_propbank_commit="0" * 40,
                    expected_ud_commit=ud_commit,
                )
            path = ud / "en_ewt-ud-train.conllu"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EWTSourceError, "modified or untracked"):
                build_ewt_dataset(
                    propbank,
                    ud,
                    expected_propbank_commit=propbank_commit,
                    expected_ud_commit=ud_commit,
                )

    def test_malformed_brackets_fail_the_preparation_gate_without_writing(self) -> None:
        documents = _minimal_documents()
        words, xpos, rows = documents["train"]["invented_train"][0]
        malformed_rows = tuple(
            (pos, lemma, roleset, ("(ARG0*",) if index == 0 else cells)
            for index, (pos, lemma, roleset, cells) in enumerate(rows)
        )
        documents["train"]["invented_train"] = (
            (words, xpos, malformed_rows),
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            propbank, ud, propbank_commit, ud_commit = _write_fixture(
                base, documents
            )
            build = build_ewt_dataset(
                propbank,
                ud,
                expected_propbank_commit=propbank_commit,
                expected_ud_commit=ud_commit,
            )
            output = base / "prepared"
            with self.assertRaisesRegex(EWTPreparationError, "gate failed"):
                prepare_ewt_dataset(
                    propbank,
                    ud,
                    output,
                    expected_propbank_commit=propbank_commit,
                    expected_ud_commit=ud_commit,
                    output_path_policy=lambda path: None,
                )

        self.assertEqual(
            build.report["propbank_skeleton"]["classification_rejection_reasons"],
            {"overlapping_role_spans": 1},
        )
        self.assertEqual(
            build.report["gate"]["predicate_column_classification"], "fail"
        )
        self.assertEqual(build.report["gate"]["status"], "fail")
        self.assertFalse(output.exists())

    def test_rejects_an_untracked_skeleton_injection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            propbank, ud, propbank_commit, ud_commit = _write_fixture(
                base, _minimal_documents()
            )
            injection = (
                propbank
                / "data"
                / "google"
                / "ewt"
                / "weblog"
                / "00"
                / "injected.xml.gold_skel"
            )
            injection.write_text("invented untracked payload\n", encoding="utf-8")
            with self.assertRaisesRegex(EWTSourceError, "modified or untracked"):
                build_ewt_dataset(
                    propbank,
                    ud,
                    expected_propbank_commit=propbank_commit,
                    expected_ud_commit=ud_commit,
                )


if __name__ == "__main__":
    unittest.main()
