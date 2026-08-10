from contextlib import redirect_stderr
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import semantic_action_extractor.srl.training_cli as training_cli
from semantic_action_extractor.srl.checkpoint_bundle import (
    CheckpointLabelConfig,
    save_checkpoint_bundle,
)
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
    def __init__(
        self,
        *,
        fail_on_call=None,
        fail_after_checkpoint_on_call=None,
        metadata_overrides=None,
    ):
        self.fail_on_call = fail_on_call
        self.fail_after_checkpoint_on_call = fail_after_checkpoint_on_call
        self.metadata_overrides = metadata_overrides
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
        metadata_values = {
            "labels": LABELS,
            "package_versions": {
                "python": "3.12.11",
                "torch": "fake-2.7",
                "transformers": "fake-4.53",
            },
            "hardware": {
                "accelerator": "fake-cpu",
                "machine": "fake-machine",
                "operating_system": "fake-os",
            },
            "resolved_device": "cpu",
            "counts": {
                "epochs_completed": config.epochs,
                "optimizer_steps": 2,
                "training_active_tokens": 4,
            },
            "drop_stats": {"total_overlength": 0},
            "initial_state_fingerprint": hashlib.sha256(
                f"initial:{seed}".encode()
            ).hexdigest(),
        }
        if self.metadata_overrides is not None:
            metadata_values.update(
                self.metadata_overrides(config, seed, len(self.calls))
            )
        metadata = RunMetadata.create(
            git_revision=context.git_revision,
            config_digest=config.digest,
            dataset_fingerprint=config.prepared_data_fingerprint,
            started_at=context.started_at,
            recorded_at=utc_now(),
            seed=seed,
            variant=config.variant,
            requested_device=config.device_request,
            **metadata_values,
        )
        result = SRLRunResult(
            result_version=EXPERIMENT_RESULT_VERSION,
            metadata=metadata,
            best_epoch=1,
            checkpoint_selection_metric=config.checkpoint_selection_metric,
            best_development=summary,
            test=summary,
            epochs=epochs,
        )
        save_checkpoint_bundle(
            checkpoint,
            metadata=metadata,
            config=config,
            label_config=CheckpointLabelConfig.from_labels(metadata.labels),
            state_dict=f"state:{config.variant}:{seed}".encode("utf-8"),
            save_state_dict=lambda state, path: path.write_bytes(state),
        )
        if len(self.calls) == self.fail_after_checkpoint_on_call:
            raise RuntimeError("SECRET CORPUS SENTENCE must never be emitted")
        return result


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


def _write_configs(
    root,
    fingerprint,
    *,
    predicate_overrides=None,
    ablation_overrides=None,
):
    predicate_path = root / "predicate.toml"
    ablation_path = root / "ablation.toml"
    predicate_path.write_text(
        _config_source(
            fingerprint,
            "predicate_signal",
            **(predicate_overrides or {}),
        ),
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


def _arguments(
    predicate,
    ablation,
    dataset,
    output,
    *,
    git_revision=GIT_REVISION,
    resume=False,
):
    arguments = [
        "--predicate-config",
        str(predicate),
        "--ablation-config",
        str(ablation),
        "--dataset",
        str(dataset),
        "--output-root",
        str(output),
        "--git-revision",
        git_revision,
    ]
    if resume:
        arguments.append("--resume")
    return arguments


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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
            lock_path = output.with_name(f".{output.name}.lock")
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
            self.assertEqual(
                checked_paths,
                [output.resolve(), partial.resolve(), lock_path.resolve()],
            )
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
            final_journal = json.loads(
                (output / PARTIAL_STATUS_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(final_journal["status"], "complete")
            self.assertEqual(final_journal["completed_runs"], 6)
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

    def test_resume_reuses_only_verified_completed_prefix(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "resumed-study"
            partial = output.with_name("resumed-study.partial")

            interrupted_runner = InjectedRunner(fail_on_call=3)
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=interrupted_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)
            partial_status = json.loads(
                (partial / PARTIAL_STATUS_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(partial_status["completed_runs"], 2)

            no_flag_runner = InjectedRunner()
            no_flag_stderr = StringIO()
            no_flag_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=no_flag_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=no_flag_stderr,
            )
            self.assertEqual(no_flag_status, 2)
            self.assertEqual(no_flag_runner.calls, [])
            self.assertEqual(
                json.loads(no_flag_stderr.getvalue())["phase"],
                "output_preflight",
            )

            resumed_runner = InjectedRunner()
            stdout = StringIO()
            stderr = StringIO()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=resumed_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(resumed_status, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue(output.is_dir())
            self.assertFalse(partial.exists())
            self.assertEqual(
                tuple(
                    (call["config"].variant, call["seed"])
                    for call in resumed_runner.calls
                ),
                (
                    ("predicate_signal", 17),
                    ("no_predicate_signal", 17),
                    ("predicate_signal", 23),
                    ("no_predicate_signal", 23),
                ),
            )
            completion = json.loads(stdout.getvalue())
            self.assertEqual(completion["completed_runs"], 6)
            self.assertEqual(completion["reused_runs"], 2)

    def test_resume_reruns_valid_checkpoint_without_completed_result(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "checkpoint-only"
            partial = output.with_name("checkpoint-only.partial")
            interrupted_runner = InjectedRunner(
                fail_after_checkpoint_on_call=2
            )
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=interrupted_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)
            partial_status = json.loads(
                (partial / PARTIAL_STATUS_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(partial_status["completed_runs"], 1)
            self.assertTrue(
                (
                    partial
                    / "checkpoints"
                    / "no_predicate_signal"
                    / "seed-0000000011"
                ).is_dir()
            )

            resumed_runner = InjectedRunner()
            stdout = StringIO()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=resumed_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=stdout,
                stderr=StringIO(),
            )

            self.assertEqual(resumed_status, 0)
            self.assertEqual(len(resumed_runner.calls), 5)
            self.assertEqual(
                (
                    resumed_runner.calls[0]["config"].variant,
                    resumed_runner.calls[0]["seed"],
                ),
                ("no_predicate_signal", 11),
            )
            self.assertEqual(json.loads(stdout.getvalue())["reused_runs"], 1)

    def test_resume_recovers_verified_run_missing_only_status_commit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "stale-status"
            partial = output.with_name("stale-status.partial")
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=InjectedRunner(fail_on_call=3),
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)
            status_path = partial / PARTIAL_STATUS_FILENAME
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            status_payload["completed_runs"] = 1
            status_payload["runs"] = status_payload["runs"][:1]
            status_path.write_bytes(_canonical_json_bytes(status_payload))
            (partial / FAILURE_MARKER_FILENAME).unlink()

            resumed_runner = InjectedRunner()
            stdout = StringIO()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=resumed_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=stdout,
                stderr=StringIO(),
            )

            self.assertEqual(resumed_status, 0)
            self.assertEqual(len(resumed_runner.calls), 4)
            self.assertEqual(
                tuple(call["seed"] for call in resumed_runner.calls),
                (17, 17, 23, 23),
            )
            self.assertEqual(json.loads(stdout.getvalue())["reused_runs"], 2)

    def test_resume_recovers_result_when_status_write_failed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "status-write-failure"
            partial = output.with_name("status-write-failure.partial")
            real_write_status = training_cli._write_partial_status
            write_calls = 0

            def fail_first_run_status(*args, **kwargs):
                nonlocal write_calls
                write_calls += 1
                if write_calls == 2:
                    raise OSError("injected status write failure")
                return real_write_status(*args, **kwargs)

            with patch.object(
                training_cli,
                "_write_partial_status",
                new=fail_first_run_status,
            ):
                first_status = main(
                    _arguments(predicate, ablation, dataset, output),
                    experiment_runner=InjectedRunner(),
                    utc_now=IncrementingClock(),
                    output_path_policy=lambda path: None,
                    repository_revision_policy=_allow_repository_revision,
                    stdout=StringIO(),
                    stderr=StringIO(),
                )

            self.assertEqual(first_status, 1)
            status_payload = json.loads(
                (partial / PARTIAL_STATUS_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(status_payload["completed_runs"], 0)
            marker = json.loads(
                (partial / FAILURE_MARKER_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(marker["completed_runs"], 1)
            self.assertTrue(
                (
                    partial
                    / "run_results"
                    / "predicate_signal"
                    / "seed-0000000011.json"
                ).is_file()
            )

            resumed_runner = InjectedRunner()
            stdout = StringIO()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=resumed_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=stdout,
                stderr=StringIO(),
            )

            self.assertEqual(resumed_status, 0)
            self.assertEqual(len(resumed_runner.calls), 5)
            self.assertEqual(json.loads(stdout.getvalue())["reused_runs"], 1)

    def test_resume_rejects_runtime_identity_mixing(self):
        mismatches = {
            "package_versions": {
                "python": "3.12.11",
                "torch": "fake-2.8",
                "transformers": "fake-4.53",
            },
            "hardware": {
                "accelerator": "fake-other",
                "machine": "fake-machine",
                "operating_system": "fake-os",
            },
            "resolved_device": "mps",
            "labels": (
                "O",
                "B-ARG0",
                "B-ARG1",
                "B-V",
                "I-ARG0",
                "I-ARG1",
                "I-V",
            ),
            "counts": {
                "epochs_completed": 2,
                "optimizer_steps": 3,
                "training_active_tokens": 4,
            },
            "drop_stats": {"total_overlength": 1},
        }
        for field, replacement in mismatches.items():
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset = root / "prepared"
                manifest = _write_dataset(dataset)
                predicate, ablation = _write_configs(
                    root,
                    manifest.dataset_fingerprint,
                    predicate_overrides={"device_request": '"auto"'},
                    ablation_overrides={"device_request": '"auto"'},
                )
                output = root / "runs" / f"runtime-{field}"
                partial = output.with_name(f"runtime-{field}.partial")
                first_status = main(
                    _arguments(predicate, ablation, dataset, output),
                    experiment_runner=InjectedRunner(fail_on_call=3),
                    utc_now=IncrementingClock(),
                    output_path_policy=lambda path: None,
                    repository_revision_policy=_allow_repository_revision,
                    stdout=StringIO(),
                    stderr=StringIO(),
                )
                self.assertEqual(first_status, 1)

                resumed_runner = InjectedRunner(
                    metadata_overrides=(
                        lambda config, seed, call, field=field,
                        replacement=replacement: {field: replacement}
                    )
                )
                stderr = StringIO()
                resumed_status = main(
                    _arguments(
                        predicate,
                        ablation,
                        dataset,
                        output,
                        resume=True,
                    ),
                    experiment_runner=resumed_runner,
                    utc_now=IncrementingClock(),
                    output_path_policy=lambda path: None,
                    repository_revision_policy=_allow_repository_revision,
                    stdout=StringIO(),
                    stderr=stderr,
                )

                self.assertEqual(resumed_status, 1)
                self.assertEqual(len(resumed_runner.calls), 1)
                self.assertEqual(
                    json.loads(stderr.getvalue())["phase"],
                    "paired_training",
                )
                status_payload = json.loads(
                    (partial / PARTIAL_STATUS_FILENAME).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(status_payload["completed_runs"], 2)
                self.assertEqual(len(status_payload["runs"]), 2)
                self.assertFalse(output.exists())

    def test_invalid_ablation_pair_is_not_committed_and_can_be_rerun(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "invalid-pair"
            partial = output.with_name("invalid-pair.partial")

            def mismatch_initial_state(config, seed, call):
                del seed, call
                if config.variant == "no_predicate_signal":
                    return {
                        "initial_state_fingerprint": hashlib.sha256(
                            b"different initial state"
                        ).hexdigest()
                    }
                return {}

            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=InjectedRunner(
                    metadata_overrides=mismatch_initial_state
                ),
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )

            self.assertEqual(first_status, 1)
            status_payload = json.loads(
                (partial / PARTIAL_STATUS_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(status_payload["completed_runs"], 1)
            invalid_result = (
                partial
                / "run_results"
                / "no_predicate_signal"
                / "seed-0000000011.json"
            )
            invalid_checkpoint = (
                partial
                / "checkpoints"
                / "no_predicate_signal"
                / "seed-0000000011"
            )
            self.assertFalse(invalid_result.exists())
            self.assertTrue(invalid_checkpoint.is_dir())

            resumed_runner = InjectedRunner()
            stdout = StringIO()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=resumed_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=stdout,
                stderr=StringIO(),
            )

            self.assertEqual(resumed_status, 0)
            self.assertEqual(len(resumed_runner.calls), 5)
            self.assertEqual(
                resumed_runner.calls[0]["config"].variant,
                "no_predicate_signal",
            )
            self.assertEqual(json.loads(stdout.getvalue())["reused_runs"], 1)

    def test_resume_after_final_rename_failure_keeps_complete_journal(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "rename-failure"
            partial = output.with_name("rename-failure.partial")
            real_rename = Path.rename
            injected = False

            def fail_final_rename(source, target):
                nonlocal injected
                if (
                    not injected
                    and source.name == partial.name
                    and Path(target).name == output.name
                ):
                    injected = True
                    raise OSError("injected final rename failure")
                return real_rename(source, target)

            first_runner = InjectedRunner()
            with patch.object(type(partial), "rename", new=fail_final_rename):
                first_status = main(
                    _arguments(predicate, ablation, dataset, output),
                    experiment_runner=first_runner,
                    utc_now=IncrementingClock(),
                    output_path_policy=lambda path: None,
                    repository_revision_policy=_allow_repository_revision,
                    stdout=StringIO(),
                    stderr=StringIO(),
                )

            self.assertEqual(first_status, 1)
            self.assertEqual(len(first_runner.calls), 6)
            self.assertTrue(partial.is_dir())
            self.assertTrue((partial / PARTIAL_STATUS_FILENAME).is_file())
            terminal_journal = json.loads(
                (partial / PARTIAL_STATUS_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(terminal_journal["status"], "complete")
            self.assertEqual(terminal_journal["completed_runs"], 6)
            self.assertTrue((partial / PAIRED_RESULTS_FILENAME).is_file())
            self.assertFalse(output.exists())

            paired_path = partial / PAIRED_RESULTS_FILENAME
            paired_bytes = paired_path.read_bytes()
            paired_path.write_bytes(_canonical_json_bytes({"tampered": True}))
            rejected_runner = InjectedRunner()
            rejected_stderr = StringIO()
            rejected_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=rejected_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=rejected_stderr,
            )
            self.assertEqual(rejected_status, 2)
            self.assertEqual(rejected_runner.calls, [])
            self.assertEqual(
                json.loads(rejected_stderr.getvalue())["phase"],
                "resume_preflight",
            )
            paired_path.write_bytes(paired_bytes)

            resumed_runner = InjectedRunner()
            stdout = StringIO()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=resumed_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=stdout,
                stderr=StringIO(),
            )

            self.assertEqual(resumed_status, 0)
            self.assertEqual(resumed_runner.calls, [])
            self.assertEqual(json.loads(stdout.getvalue())["reused_runs"], 6)
            self.assertTrue(output.is_dir())
            self.assertFalse(partial.exists())
            final_journal = json.loads(
                (output / PARTIAL_STATUS_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(final_journal["status"], "complete")

    def test_resume_cleans_exact_atomic_writer_residues(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "atomic-residue"
            partial = output.with_name("atomic-residue.partial")
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=InjectedRunner(fail_on_call=2),
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)

            journal_residue = (
                partial
                / f".{PARTIAL_STATUS_FILENAME}.abcdefgh.write-partial"
            )
            journal_residue.write_bytes(b"interrupted journal bytes")
            next_result_directory = (
                partial / "run_results" / "no_predicate_signal"
            )
            next_result_directory.mkdir(parents=True, exist_ok=True)
            result_residue = (
                next_result_directory
                / ".seed-0000000011.json.1234_ab_.write-partial"
            )
            result_residue.write_bytes(b"interrupted result bytes")

            resumed_runner = InjectedRunner()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=resumed_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )

            self.assertEqual(resumed_status, 0)
            self.assertEqual(len(resumed_runner.calls), 5)
            self.assertFalse(journal_residue.exists())
            self.assertFalse(result_residue.exists())
            self.assertTrue(output.is_dir())

    def test_resume_rejects_atomic_writer_residue_lookalike(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "residue-lookalike"
            partial = output.with_name("residue-lookalike.partial")
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=InjectedRunner(fail_on_call=2),
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)
            lookalike = (
                partial
                / f".{PARTIAL_STATUS_FILENAME}.ABCDEFGH.write-partial"
            )
            lookalike.write_bytes(b"not a writer-owned tempfile name")

            runner = InjectedRunner()
            stderr = StringIO()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(resumed_status, 2)
            self.assertEqual(runner.calls, [])
            self.assertEqual(
                json.loads(stderr.getvalue())["phase"], "resume_preflight"
            )
            self.assertTrue(lookalike.is_file())

    def test_resume_cleans_only_next_checkpoint_staging_residue(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "checkpoint-stage-residue"
            partial = output.with_name("checkpoint-stage-residue.partial")
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=InjectedRunner(fail_on_call=2),
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)

            wrong_stage = (
                partial
                / "checkpoints"
                / "predicate_signal"
                / ".seed-0000000017.abcdefgh.run-partial"
            )
            wrong_stage.mkdir(parents=True)
            (wrong_stage / "garbage.bin").write_bytes(b"incomplete")
            rejected_runner = InjectedRunner()
            rejected_stderr = StringIO()
            rejected_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=rejected_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=rejected_stderr,
            )
            self.assertEqual(rejected_status, 2)
            self.assertEqual(rejected_runner.calls, [])
            shutil.rmtree(wrong_stage)

            next_stage = (
                partial
                / "checkpoints"
                / "no_predicate_signal"
                / ".seed-0000000011.a1b2c3d4.run-partial"
            )
            staged_checkpoint = next_stage / "checkpoint"
            staged_checkpoint.mkdir(parents=True)
            (staged_checkpoint / "state_dict.pt").write_bytes(b"partial")

            resumed_runner = InjectedRunner()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=resumed_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )

            self.assertEqual(resumed_status, 0)
            self.assertEqual(len(resumed_runner.calls), 5)
            self.assertFalse(next_stage.exists())
            self.assertTrue(output.is_dir())

    def test_resume_finishes_partially_deleted_checkpoint_tombstone(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "checkpoint-tombstone"
            partial = output.with_name("checkpoint-tombstone.partial")
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=InjectedRunner(
                    fail_after_checkpoint_on_call=2
                ),
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)
            checkpoint = (
                partial
                / "checkpoints"
                / "no_predicate_signal"
                / "seed-0000000011"
            )
            tombstone = checkpoint.with_name(
                ".seed-0000000011.discard-tombstone"
            )
            checkpoint.rename(tombstone)
            (tombstone / "state_dict.pt").unlink()

            resumed_runner = InjectedRunner()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=resumed_runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )

            self.assertEqual(resumed_status, 0)
            self.assertEqual(len(resumed_runner.calls), 5)
            self.assertFalse(tombstone.exists())
            self.assertTrue(output.is_dir())

    def test_resume_rejects_unbounded_checkpoint_tombstones(self):
        cases = ("lookalike", "symlink", "wrong-plan", "unknown")
        for case in cases:
            with self.subTest(case=case), TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset = root / "prepared"
                manifest = _write_dataset(dataset)
                predicate, ablation = _write_configs(
                    root, manifest.dataset_fingerprint
                )
                output = root / "runs" / f"tombstone-{case}"
                partial = output.with_name(f"tombstone-{case}.partial")
                first_status = main(
                    _arguments(predicate, ablation, dataset, output),
                    experiment_runner=InjectedRunner(
                        fail_after_checkpoint_on_call=2
                    ),
                    utc_now=IncrementingClock(),
                    output_path_policy=lambda path: None,
                    repository_revision_policy=_allow_repository_revision,
                    stdout=StringIO(),
                    stderr=StringIO(),
                )
                self.assertEqual(first_status, 1)
                checkpoint = (
                    partial
                    / "checkpoints"
                    / "no_predicate_signal"
                    / "seed-0000000011"
                )
                if case == "lookalike":
                    artifact = checkpoint.with_name(
                        ".seed-0000000011.discard-tombstone-extra"
                    )
                    artifact.mkdir()
                elif case == "symlink":
                    saved_checkpoint = root / "saved-checkpoint"
                    checkpoint.rename(saved_checkpoint)
                    artifact = checkpoint.with_name(
                        ".seed-0000000011.discard-tombstone"
                    )
                    artifact.symlink_to(saved_checkpoint, target_is_directory=True)
                elif case == "wrong-plan":
                    artifact = (
                        partial
                        / "checkpoints"
                        / "predicate_signal"
                        / ".seed-0000000017.discard-tombstone"
                    )
                    artifact.mkdir()
                else:
                    artifact = partial / "unknown-tombstone-artifact.bin"
                    artifact.write_bytes(b"unknown")

                runner = InjectedRunner()
                stderr = StringIO()
                resumed_status = main(
                    _arguments(
                        predicate,
                        ablation,
                        dataset,
                        output,
                        resume=True,
                    ),
                    experiment_runner=runner,
                    utc_now=IncrementingClock(),
                    output_path_policy=lambda path: None,
                    repository_revision_policy=_allow_repository_revision,
                    stdout=StringIO(),
                    stderr=stderr,
                )

                self.assertEqual(resumed_status, 2)
                self.assertEqual(runner.calls, [])
                self.assertEqual(
                    json.loads(stderr.getvalue())["phase"],
                    "resume_preflight",
                )
                self.assertTrue(artifact.exists() or artifact.is_symlink())

    def test_output_lock_rejects_concurrent_writer_without_blocking(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "locked-study"
            resolved_output = output.resolve(strict=False)
            lock_path = training_cli._output_lock_path_for(resolved_output)
            held_lock = training_cli._acquire_output_lock(lock_path)
            runner = InjectedRunner()
            stderr = StringIO()
            try:
                status = main(
                    _arguments(predicate, ablation, dataset, output),
                    experiment_runner=runner,
                    utc_now=IncrementingClock(),
                    output_path_policy=lambda path: None,
                    repository_revision_policy=_allow_repository_revision,
                    stdout=StringIO(),
                    stderr=stderr,
                )
            finally:
                held_lock.close()

            self.assertEqual(status, 2)
            self.assertEqual(runner.calls, [])
            self.assertEqual(
                json.loads(stderr.getvalue())["phase"], "output_lock"
            )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name("locked-study.partial").exists())

            final_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(final_status, 0)
            self.assertEqual(len(runner.calls), 6)

    def test_resume_rejects_tampered_completed_run_result(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "tampered-result"
            partial = output.with_name("tampered-result.partial")
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=InjectedRunner(fail_on_call=2),
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)
            status_payload = json.loads(
                (partial / PARTIAL_STATUS_FILENAME).read_text(encoding="utf-8")
            )
            result_path = partial / status_payload["runs"][0]["result_path"]
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            result_payload["test"]["loss"] = 0.125
            result_path.write_bytes(_canonical_json_bytes(result_payload))
            marker_before = (partial / FAILURE_MARKER_FILENAME).read_bytes()

            runner = InjectedRunner()
            stderr = StringIO()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(resumed_status, 2)
            self.assertEqual(runner.calls, [])
            self.assertEqual(
                json.loads(stderr.getvalue())["phase"], "resume_preflight"
            )
            self.assertEqual(
                (partial / FAILURE_MARKER_FILENAME).read_bytes(), marker_before
            )
            self.assertFalse(output.exists())

    def test_resume_rejects_tampered_checkpoint_state(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "tampered-checkpoint"
            partial = output.with_name("tampered-checkpoint.partial")
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=InjectedRunner(fail_on_call=2),
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)
            status_payload = json.loads(
                (partial / PARTIAL_STATUS_FILENAME).read_text(encoding="utf-8")
            )
            checkpoint = partial / status_payload["runs"][0]["checkpoint_path"]
            (checkpoint / "state_dict.pt").write_bytes(b"altered state")

            runner = InjectedRunner()
            stderr = StringIO()
            resumed_status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(resumed_status, 2)
            self.assertEqual(runner.calls, [])
            self.assertEqual(
                json.loads(stderr.getvalue())["phase"], "resume_preflight"
            )
            self.assertFalse(output.exists())

    def test_resume_requires_identical_provenance_and_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            predicate, ablation = _write_configs(
                root, manifest.dataset_fingerprint
            )
            output = root / "runs" / "provenance"
            partial = output.with_name("provenance.partial")
            first_status = main(
                _arguments(predicate, ablation, dataset, output),
                experiment_runner=InjectedRunner(fail_on_call=2),
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(first_status, 1)

            dataset_copy = root / "prepared-copy"
            shutil.copytree(dataset, dataset_copy)
            other_output = root / "runs" / "other-output"
            shutil.copytree(
                partial,
                other_output.with_name("other-output.partial"),
            )
            runner = InjectedRunner()

            attempts = (
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    git_revision="d" * 40,
                    resume=True,
                ),
                _arguments(
                    predicate,
                    ablation,
                    dataset_copy,
                    output,
                    resume=True,
                ),
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    other_output,
                    resume=True,
                ),
            )
            for arguments in attempts:
                with self.subTest(arguments=arguments):
                    stderr = StringIO()
                    status = main(
                        arguments,
                        experiment_runner=runner,
                        utc_now=IncrementingClock(),
                        output_path_policy=lambda path: None,
                        repository_revision_policy=_allow_repository_revision,
                        stdout=StringIO(),
                        stderr=stderr,
                    )
                    self.assertEqual(status, 2)
                    self.assertEqual(
                        json.loads(stderr.getvalue())["phase"],
                        "resume_preflight",
                    )

            _write_configs(
                root,
                manifest.dataset_fingerprint,
                ablation_overrides={"batch_size": "4"},
            )
            predicate.write_text(
                _config_source(
                    manifest.dataset_fingerprint,
                    "predicate_signal",
                    batch_size="4",
                ),
                encoding="utf-8",
            )
            stderr = StringIO()
            status = main(
                _arguments(
                    predicate,
                    ablation,
                    dataset,
                    output,
                    resume=True,
                ),
                experiment_runner=runner,
                utc_now=IncrementingClock(),
                output_path_policy=lambda path: None,
                repository_revision_policy=_allow_repository_revision,
                stdout=StringIO(),
                stderr=stderr,
            )
            self.assertEqual(status, 2)
            self.assertEqual(
                json.loads(stderr.getvalue())["phase"], "resume_preflight"
            )
            self.assertEqual(runner.calls, [])
            self.assertTrue(partial.is_dir())
            self.assertFalse(output.exists())

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
