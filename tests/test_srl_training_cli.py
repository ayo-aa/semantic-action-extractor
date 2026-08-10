from contextlib import redirect_stderr
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from semantic_action_extractor.srl.dataset_io import write_prepared_dataset
from semantic_action_extractor.srl.evaluation import (
    evaluate_supplied_predicate_srl,
)
from semantic_action_extractor.srl.example import WordLevelSRLExample
from semantic_action_extractor.srl.experiment_config import load_training_config
from semantic_action_extractor.srl.run_metadata import RunMetadata
from semantic_action_extractor.srl.training_cli import (
    CONFIG_IDENTITIES_FILENAME,
    FAILURE_MARKER_FILENAME,
    PAIRED_RESULTS_FILENAME,
    PARTIAL_STATUS_FILENAME,
    RESULT_IDENTITY_FILENAME,
    main,
    require_clean_repository_revision,
    require_git_ignored_output_path,
)
from semantic_action_extractor.srl.training_engine import (
    EXPERIMENT_RESULT_VERSION,
    EpochSummary,
    EvaluationSummary,
    SRLRunResult,
)


MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40
GIT_REVISION = "c" * 40
LABELS = ("O", "B-ARG0", "B-V", "I-ARG0", "I-V")


class IncrementingClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        minute = self.calls
        self.calls += 1
        return f"2026-08-10T08:{minute:02d}:00Z"


class InjectedRunner:
    def __init__(self, *, fail_on_call=None):
        self.fail_on_call = fail_on_call
        self.calls = []

    def __call__(
        self,
        dataset,
        config,
        checkpoint,
        *,
        seed,
        context,
        utc_now,
    ):
        checkpoint = Path(checkpoint)
        self.calls.append(
            {
                "dataset": Path(dataset),
                "config": config,
                "checkpoint": checkpoint,
                "seed": seed,
                "context": context,
            }
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "synthetic-checkpoint").write_bytes(b"state")
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("SECRET CORPUS SENTENCE must never be emitted")

        evaluation = evaluate_supplied_predicate_srl(
            [("B-ARG0", "B-V")],
            [("B-ARG0", "B-V")],
            [1],
        )
        summary = EvaluationSummary(loss=0.25, evaluation=evaluation)
        selection_value = (
            evaluation.arguments.f1
            if config.checkpoint_selection_metric
            == "development_argument_f1"
            else summary.loss
        )
        epochs = tuple(
            EpochSummary(
                epoch=epoch,
                training_loss=0.5,
                development=summary,
                selection_value=selection_value,
            )
            for epoch in range(1, config.epochs + 1)
        )
        metadata = RunMetadata.create(
            git_revision=context.git_revision,
            config_digest=config.digest,
            dataset_fingerprint=config.prepared_data_fingerprint,
            labels=LABELS,
            package_versions={
                "python": "3.12.11",
                "torch": "fake-2.7",
                "transformers": "fake-4.53",
            },
            hardware={
                "accelerator": "fake-cpu",
                "machine": "fake-machine",
                "operating_system": "fake-os",
            },
            started_at=context.started_at,
            recorded_at=utc_now(),
            seed=seed,
            variant=config.variant,
            requested_device=config.device_request,
            resolved_device="cpu",
            counts={
                "epochs_completed": config.epochs,
                "optimizer_steps": 2,
                "training_active_tokens": 4,
            },
            drop_stats={"total_overlength": 0},
            initial_state_fingerprint=hashlib.sha256(
                f"initial:{seed}".encode()
            ).hexdigest(),
        )
        return SRLRunResult(
            result_version=EXPERIMENT_RESULT_VERSION,
            metadata=metadata,
            best_epoch=1,
            checkpoint_selection_metric=config.checkpoint_selection_metric,
            best_development=summary,
            test=summary,
            epochs=epochs,
        )


def _example(example_id, split, words):
    return WordLevelSRLExample(
        example_id=example_id,
        document_id=example_id,
        sentence_id=f"{example_id}:sentence",
        split=split,
        words=words,
        predicate_index=1,
        tags=("B-ARG0", "B-V"),
        predicate_roleset="invent.01",
    )


def _write_dataset(directory):
    return write_prepared_dataset(
        directory,
        (
            _example("train-one", "train", ("SecretTrain", "filed")),
            _example(
                "development-one",
                "development",
                ("SecretDevelopment", "reviewed"),
            ),
            _example("test-one", "test", ("SecretTest", "approved")),
        ),
    )


def _config_source(fingerprint, variant, **overrides):
    values = {
        "config_version": "1",
        "variant": f'"{variant}"',
        "model_id": '"google-bert/bert-base-uncased"',
        "model_revision": f'"{MODEL_REVISION}"',
        "tokenizer_id": '"google-bert/bert-base-uncased"',
        "tokenizer_revision": f'"{TOKENIZER_REVISION}"',
        "prepared_data_fingerprint": f'"{fingerprint}"',
        "max_length": "64",
        "batch_size": "2",
        "learning_rate": "1e-5",
        "epochs": "2",
        "weight_decay": "0.0",
        "warmup_ratio": "0.0",
        "gradient_clip_norm": "1.0",
        "paired_seeds": "[11, 17, 23]",
        "checkpoint_selection_metric": '"development_argument_f1"',
        "device_request": '"cpu"',
    }
    values.update(overrides)
    return "\n".join(f"{key} = {value}" for key, value in values.items()) + "\n"


def _write_configs(root, fingerprint, *, ablation_overrides=None):
    predicate_path = root / "predicate.toml"
    ablation_path = root / "ablation.toml"
    predicate_path.write_text(
        _config_source(fingerprint, "predicate_signal"),
        encoding="utf-8",
    )
    ablation_path.write_text(
        _config_source(
            fingerprint,
            "no_predicate_signal",
            **(ablation_overrides or {}),
        ),
        encoding="utf-8",
    )
    return predicate_path, ablation_path


def _arguments(predicate, ablation, dataset, output):
    return [
        "--predicate-config",
        str(predicate),
        "--ablation-config",
        str(ablation),
        "--dataset",
        str(dataset),
        "--output-root",
        str(output),
        "--git-revision",
        GIT_REVISION,
    ]


def _allow_repository_revision(repository, revision):
    del repository, revision


class TrainingCliTests(unittest.TestCase):
    def test_runs_default_engine_entry_and_publishes_canonical_results(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate_path, ablation_path = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "paired-study"
            partial = output.with_name(f"{output.name}.partial")
            runner = InjectedRunner()
            clock = IncrementingClock()
            checked_paths = []
            stdout = StringIO()
            stderr = StringIO()

            with patch(
                "semantic_action_extractor.srl.training_cli."
                "run_srl_experiment",
                new=runner,
            ):
                status = main(
                    _arguments(
                        predicate_path,
                        ablation_path,
                        dataset,
                        output,
                    ),
                    utc_now=clock,
                    output_path_policy=checked_paths.append,
                    repository_revision_policy=_allow_repository_revision,
                    stdout=stdout,
                    stderr=stderr,
                )

            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue(output.is_dir())
            self.assertFalse(partial.exists())
            self.assertEqual(checked_paths, [output.resolve(), partial.resolve()])
            self.assertEqual(len(runner.calls), 6)
            self.assertEqual(clock.calls, 12)
            self.assertEqual(
                tuple(
                    (call["config"].variant, call["seed"])
                    for call in runner.calls
                ),
                (
                    ("predicate_signal", 11),
                    ("no_predicate_signal", 11),
                    ("predicate_signal", 17),
                    ("no_predicate_signal", 17),
                    ("predicate_signal", 23),
                    ("no_predicate_signal", 23),
                ),
            )
            expected_checkpoint_paths = tuple(
                output
                / "checkpoints"
                / variant
                / f"seed-{seed:010d}"
                for seed in (11, 17, 23)
                for variant in ("predicate_signal", "no_predicate_signal")
            )
            self.assertTrue(
                all(path.is_dir() for path in expected_checkpoint_paths)
            )
            self.assertFalse((output / PARTIAL_STATUS_FILENAME).exists())
            self.assertFalse((output / FAILURE_MARKER_FILENAME).exists())

            predicate_config = load_training_config(predicate_path)
            ablation_config = load_training_config(ablation_path)
            self.assertEqual(
                (output / "configs" / "predicate_signal.json").read_bytes(),
                predicate_config.canonical_json_bytes(),
            )
            self.assertEqual(
                (
                    output / "configs" / "no_predicate_signal.json"
                ).read_bytes(),
                ablation_config.canonical_json_bytes(),
            )
            identities = json.loads(
                (output / CONFIG_IDENTITIES_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                identities["predicate_signal"]["digest"],
                predicate_config.digest,
            )

            result_path = output / PAIRED_RESULTS_FILENAME
            result_bytes = result_path.read_bytes()
            self.assertEqual(
                result_bytes,
                json.dumps(
                    json.loads(result_bytes),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode(),
            )
            result_identity = json.loads(
                (output / RESULT_IDENTITY_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(result_identity["path"], PAIRED_RESULTS_FILENAME)
            self.assertEqual(
                result_identity["digest"],
                hashlib.sha256(result_bytes).hexdigest(),
            )
            emitted = json.loads(stdout.getvalue())
            self.assertEqual(emitted["status"], "complete")
            self.assertEqual(emitted["completed_runs"], 6)
            self.assertEqual(emitted["output_root"], str(output.resolve()))
            self.assertEqual(
                emitted["paired_results_path"], str(result_path.resolve())
            )
            self.assertEqual(
                emitted["paired_results_digest"],
                hashlib.sha256(result_bytes).hexdigest(),
            )
            self.assertEqual(
                emitted["paired_result_identity_path"],
                str((output / RESULT_IDENTITY_FILENAME).resolve()),
            )
            self.assertEqual(
                emitted["predicate_config_digest"], predicate_config.digest
            )
            for secret in ("SecretTrain", "SecretDevelopment", "SecretTest"):
                self.assertNotIn(secret, stdout.getvalue())
                self.assertNotIn(secret, result_bytes.decode("utf-8"))

    def test_rejects_mismatched_configs_before_output_or_runner(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root,
                manifest.dataset_fingerprint,
                ablation_overrides={"batch_size": "4"},
            )
            output = root / "runs" / "mismatch"
            runner = InjectedRunner()
            checked_paths = []
            stderr = StringIO()

            status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=checked_paths.append,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(status, 2)
            self.assertEqual(runner.calls, [])
            self.assertEqual(checked_paths, [])
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name("mismatch.partial").exists())
            rejection = json.loads(stderr.getvalue())
            self.assertEqual(rejection["status"], "rejected")
            self.assertEqual(rejection["phase"], "config_validation")

    def test_rejects_existing_or_unapproved_output_before_ml(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "existing"
            output.mkdir(parents=True)
            runner = InjectedRunner()
            stderr = StringIO()

            status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(status, 2)
            self.assertEqual(runner.calls, [])
            self.assertEqual(
                json.loads(stderr.getvalue())["phase"], "output_preflight"
            )

            output = root / "runs" / "not-approved"
            checked = []

            def reject_path(path):
                checked.append(path)
                raise ValueError("not ignored")

            stderr = StringIO()
            status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=reject_path,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=stderr,
            )
            self.assertEqual(status, 2)
            self.assertEqual(len(checked), 1)
            self.assertFalse(output.exists())

    def test_rejects_repository_provenance_before_output_or_runner(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "bad-revision"
            runner = InjectedRunner()
            checked_paths = []
            stderr = StringIO()

            def reject_revision(repository, revision):
                del repository, revision
                raise ValueError("dirty repository")

            status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=checked_paths.append,
                repository_revision_policy=reject_revision,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(status, 2)
            self.assertEqual(runner.calls, [])
            self.assertEqual(checked_paths, [])
            self.assertFalse(output.exists())
            rejection = json.loads(stderr.getvalue())
            self.assertEqual(rejection["phase"], "repository_preflight")

    def test_rejects_dataset_mismatch_before_creating_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            _write_dataset(dataset)
            predicate, ablation = _write_configs(root, "d" * 64)
            output = root / "runs" / "bad-dataset"
            runner = InjectedRunner()
            stderr = StringIO()

            status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(status, 2)
            self.assertEqual(runner.calls, [])
            self.assertFalse(output.exists())
            rejection = json.loads(stderr.getvalue())
            self.assertEqual(rejection["phase"], "dataset_preflight")

    def test_failure_keeps_sanitized_partial_marker(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "failed-study"
            partial = output.with_name("failed-study.partial")
            runner = InjectedRunner(fail_on_call=2)
            stdout = StringIO()
            stderr = StringIO()

            status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(output.exists())
            self.assertTrue(partial.is_dir())
            self.assertFalse((partial / PAIRED_RESULTS_FILENAME).exists())
            marker_path = partial / FAILURE_MARKER_FILENAME
            marker_bytes = marker_path.read_bytes()
            marker = json.loads(marker_bytes)
            self.assertEqual(marker["status"], "failed")
            self.assertEqual(marker["phase"], "paired_training")
            self.assertEqual(marker["completed_runs"], 1)
            self.assertEqual(marker["active_variant"], "no_predicate_signal")
            self.assertEqual(marker["active_seed"], 11)
            self.assertEqual(
                marker_bytes,
                json.dumps(
                    marker,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode(),
            )
            self.assertNotIn(b"SECRET CORPUS SENTENCE", marker_bytes)
            self.assertNotIn("SECRET CORPUS SENTENCE", stderr.getvalue())
            failure_output = json.loads(stderr.getvalue())
            self.assertEqual(failure_output["status"], "failed")
            self.assertEqual(failure_output["completed_runs"], 1)

    def test_requires_exact_git_revision(self):
        error_output = StringIO()
        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--predicate-config",
                        "predicate.toml",
                        "--ablation-config",
                        "ablation.toml",
                        "--dataset",
                        "prepared",
                        "--output-root",
                        "runs/study",
                        "--git-revision",
                        "main",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("40-character lowercase", error_output.getvalue())

    def test_default_output_policy_accepts_ignored_runs_only(self):
        repository = Path(__file__).resolve().parents[1]
        require_git_ignored_output_path(
            repository / "runs" / "invented-training-cli-test"
        )
        with self.assertRaisesRegex(ValueError, "Git ignore"):
            require_git_ignored_output_path(
                repository / "invented-unignored-training-cli-test"
            )

    def test_repository_policy_pins_head_source_and_clean_tracked_state(self):
        with TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            repository.mkdir()
            source = (
                repository
                / "src"
                / "semantic_action_extractor"
                / "srl"
                / "training_cli.py"
            )
            source.parent.mkdir(parents=True)
            source.write_text("# synthetic tracked CLI\n", encoding="utf-8")
            (repository / ".gitignore").write_text(
                "runs/\n", encoding="utf-8"
            )

            def git(*arguments):
                return subprocess.run(
                    ["git", "-C", str(repository), *arguments],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init")
            git("config", "user.name", "Synthetic Test")
            git("config", "user.email", "synthetic@example.invalid")
            git("add", ".gitignore", "src")
            git("commit", "-m", "Synthetic clean revision")
            revision = git("rev-parse", "HEAD")

            require_clean_repository_revision(
                repository,
                revision,
                source_path=source,
            )
            ignored = repository / "runs" / "ignored-state"
            ignored.mkdir(parents=True)
            (ignored / "checkpoint.pt").write_bytes(b"ignored")
            require_clean_repository_revision(
                repository,
                revision,
                source_path=source,
            )

            source.write_text("# modified tracked CLI\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be clean"):
                require_clean_repository_revision(
                    repository,
                    revision,
                    source_path=source,
                )
            source.write_text("# synthetic tracked CLI\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                require_clean_repository_revision(
                    repository,
                    "d" * 40,
                    source_path=source,
                )


if __name__ == "__main__":
    unittest.main()
