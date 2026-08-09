from contextlib import redirect_stdout
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from semantic_action_extractor.data_cli import _write, build_parser, main
from semantic_action_extractor.datasets.io import sha256_file
from semantic_action_extractor.datasets.qasrl import adapt_qasrl_record
from semantic_action_extractor.datasets.serialization import (
    annotation_record_from_dict,
    load_adapted_jsonl,
)
from semantic_action_extractor.datasets.common import DatasetFormatError
from semantic_action_extractor.datasets.leakage import (
    DatasetPartition,
    build_training_quarantine,
)
from semantic_action_extractor.evaluation.bundle import load_evaluation_bundle


def _synthetic_qasrl_record() -> dict:
    return {
        "sentenceId": "Wiki1k:wikipedia:999:0:0",
        "sentenceTokens": ["Maya", "approved", "refunds", "."],
        "verbEntries": {
            "1": {
                "verbIndex": 1,
                "verbInflectedForms": {
                    "stem": "approve",
                    "presentSingular3rd": "approves",
                    "presentParticiple": "approving",
                    "past": "approved",
                    "pastParticiple": "approved",
                },
                "questionLabels": {
                    "What did someone approve?": {
                        "questionString": "What did someone approve?",
                        "questionSources": ["synthetic-writer"],
                        "answerJudgments": [
                            {
                                "sourceId": "synthetic-validator",
                                "isValid": True,
                                "spans": [[2, 3]],
                            }
                        ],
                        "questionSlots": {
                            "wh": "what",
                            "aux": "did",
                            "subj": "someone",
                            "verb": "stem",
                            "obj": "_",
                            "prep": "_",
                            "obj2": "_",
                        },
                        "tense": "past",
                        "isPerfect": False,
                        "isProgressive": False,
                        "isNegated": False,
                        "isPassive": False,
                    }
                },
            }
        },
    }


class DataCliTests(unittest.TestCase):
    def test_parser_keeps_data_commands_separate_from_extraction_cli(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            ["verify", "qa-srl-bank-2.1", "/tmp/archive.tar"]
        )

        self.assertEqual(args.command, "verify")
        self.assertEqual(args.artifact, "qa-srl-bank-2.1")

    def test_adapt_qasrl_writes_canonical_jsonl_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.jsonl"
            source.write_text(
                json.dumps(_synthetic_qasrl_record()) + "\n",
                encoding="utf-8",
            )
            output = root / "adapted.jsonl"
            captured = io.StringIO()
            with redirect_stdout(captured):
                status = main(
                    [
                        "adapt-qasrl",
                        str(source),
                        str(output),
                        "--release",
                        "synthetic-1",
                        "--split",
                        "dev",
                        "--layer",
                        "gold",
                    ]
                )

            manifest_path = root / "adapted.jsonl.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            reported = json.loads(captured.getvalue())

            self.assertEqual(status, 0)
            self.assertEqual(manifest["counts"]["records"], 1)
            self.assertEqual(reported["record_fingerprint"], sha256_file(output))
            adapted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(adapted["provenance"]["release"], "synthetic-1")
            loaded = load_adapted_jsonl(
                output,
                manifest_path=manifest_path,
            )
            self.assertEqual(loaded[0].to_dict(), adapted)

            evaluation_path = root / "gold-evaluation.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "consolidate",
                            str(output),
                            str(evaluation_path),
                            "--manifest",
                            str(manifest_path),
                        ]
                    ),
                    0,
                )
            evaluation = load_evaluation_bundle(evaluation_path)
            self.assertEqual(
                evaluation.consolidation_rule,
                "valid-judgment-union-v1",
            )
            self.assertEqual(len(evaluation.corpus.predicates), 1)

    def test_canonical_reader_rejects_unknown_fields_and_checksum_drift(self) -> None:
        raw = adapt_qasrl_record(
            _synthetic_qasrl_record(),
            release="synthetic-1",
            split="dev",
        ).to_dict()
        raw["unknown"] = True
        with self.assertRaisesRegex(DatasetFormatError, "extra"):
            annotation_record_from_dict(raw)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.jsonl"
            source.write_text(
                json.dumps(_synthetic_qasrl_record()) + "\n",
                encoding="utf-8",
            )
            output = root / "adapted.jsonl"
            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "adapt-qasrl",
                        str(source),
                        str(output),
                        "--release",
                        "synthetic-1",
                        "--split",
                        "dev",
                        "--layer",
                        "gold",
                    ]
                )
            output.write_text(output.read_text(encoding="utf-8") + " ", encoding="utf-8")

            with self.assertRaisesRegex(DatasetFormatError, "checksum mismatch"):
                tuple(
                    load_adapted_jsonl(
                        output,
                        manifest_path=root / "adapted.jsonl.manifest.json",
                    )
                )

    def test_adaptation_rejects_a_source_that_changes_during_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.jsonl"
            source.write_text("initial source\n", encoding="utf-8")
            output = root / "adapted.jsonl"
            record = adapt_qasrl_record(
                _synthetic_qasrl_record(),
                release="synthetic-1",
                split="dev",
            )

            def mutating_records():
                yield record
                source.write_text("changed source\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "changed while"):
                _write(
                    records=mutating_records(),
                    output=output,
                    manifest=None,
                    dataset="qa-srl",
                    release="synthetic-1",
                    split="dev",
                    adapter="qa-srl-jsonl",
                    adapter_version="0.1.0",
                    source=source,
                    overwrite=False,
                )

            self.assertFalse(output.exists())
            self.assertFalse((root / "adapted.jsonl.manifest.json").exists())

    def test_training_adaptation_applies_the_document_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contaminated_raw = _synthetic_qasrl_record()
            clean_raw = copy.deepcopy(contaminated_raw)
            clean_raw["sentenceId"] = "Wiki1k:wikipedia:998:0:0"
            clean_raw["sentenceTokens"][2] = "credits"
            evaluation_raw = copy.deepcopy(contaminated_raw)
            evaluation_raw["sentenceId"] = "Wiki1k:wikipedia:997:0:0"
            source = root / "train.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(item) for item in (contaminated_raw, clean_raw)
                )
                + "\n",
                encoding="utf-8",
            )
            training_records = tuple(
                adapt_qasrl_record(
                    item,
                    release="synthetic-1",
                    split="train",
                    layer="expanded",
                )
                for item in (contaminated_raw, clean_raw)
            )
            evaluation = adapt_qasrl_record(
                evaluation_raw,
                release="synthetic-gold",
                split="dev",
            )
            report = build_training_quarantine(
                (DatasetPartition("qasrl-train", "train", training_records),),
                (DatasetPartition("qasrl-dev", "development", (evaluation,)),),
            )
            report = replace(
                report,
                source_artifacts={"qasrl_train": sha256_file(source)},
            )
            report_path = report.write(root / "quarantine.json")
            output = root / "adapted.jsonl"

            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "adapt-qasrl",
                        str(source),
                        str(output),
                        "--release",
                        "synthetic-1",
                        "--split",
                        "train",
                        "--layer",
                        "expanded",
                        "--quarantine-report",
                        str(report_path),
                    ]
                )

            records = load_adapted_jsonl(
                output,
                manifest_path=root / "adapted.jsonl.manifest.json",
            )
            manifest = json.loads(
                (root / "adapted.jsonl.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status, 0)
            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0].provenance.source_id,
                "Wiki1k:wikipedia:998:0:0",
            )
            self.assertEqual(
                manifest["exclusions"]["cross-role-document-quarantine-v1"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
