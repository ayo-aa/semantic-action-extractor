"""Command line for a complete paired supplied-predicate SRL experiment."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, BinaryIO, TextIO

from .checkpoint_bundle import (
    CheckpointLabelConfig,
    CheckpointMetadata,
    LABEL_CONFIG_FILENAME,
    METADATA_FILENAME,
    STATE_DICT_FILENAME,
    validate_checkpoint_bundle,
)
from .dataset_io import read_prepared_dataset
from .evaluation import (
    PredicateDiagnostics,
    RoleSpanMetrics,
    SpanMetrics,
    SuppliedPredicateEvaluation,
    TokenAccuracy,
)
from .experiment_config import TrainingConfig, load_training_config
from .run_metadata import RunMetadata
from .training_engine import (
    EXPERIMENT_RESULT_VERSION,
    EpochSummary,
    EvaluationSummary,
    PairedExperimentResult,
    PairedSeedResult,
    RunContext,
    SRLRunResult,
    run_paired_srl_experiments,
    run_srl_experiment,
    validate_paired_configs,
)


CONFIG_IDENTITY_VERSION = 1
RESULT_IDENTITY_VERSION = 1
PARTIAL_STATUS_VERSION = 2
FAILURE_MARKER_VERSION = 1

PAIRED_RESULTS_FILENAME = "paired_results.json"
CONFIG_IDENTITIES_FILENAME = "config_identities.json"
RESULT_IDENTITY_FILENAME = "paired_result_identity.json"
PARTIAL_STATUS_FILENAME = "partial_status.json"
FAILURE_MARKER_FILENAME = "failure.json"
CONFIG_DIRECTORY = "configs"
CHECKPOINT_DIRECTORY = "checkpoints"
RUN_RESULTS_DIRECTORY = "run_results"
PREDICATE_CONFIG_FILENAME = "predicate_signal.json"
ABLATION_CONFIG_FILENAME = "no_predicate_signal.json"

_PARTIAL_STATUS_KEYS = frozenset(
    {
        "partial_status_version",
        "status",
        "completed_runs",
        "expected_runs",
        "output_root",
        "repository",
        "dataset_directory",
        "dataset_fingerprint",
        "git_revision",
        "predicate_config_digest",
        "no_predicate_signal_config_digest",
        "runs",
    }
)
_COMPLETED_RUN_KEYS = frozenset(
    {
        "variant",
        "seed",
        "result_path",
        "result_sha256",
        "checkpoint_path",
        "checkpoint_state_sha256",
    }
)

_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")

ExperimentRunner = Callable[..., SRLRunResult]
UtcClock = Callable[[], str]
OutputPathPolicy = Callable[[Path], None]
RepositoryRevisionPolicy = Callable[[Path, str], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-action-train-srl",
        description=(
            "Run the three-seed predicate-signal SRL experiment from an "
            "already-prepared local dataset."
        ),
    )
    parser.add_argument(
        "--predicate-config",
        type=Path,
        required=True,
        help="Strict TOML config for the predicate_signal variant.",
    )
    parser.add_argument(
        "--ablation-config",
        type=Path,
        required=True,
        help="Strict TOML config for the no_predicate_signal variant.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Existing prepared SRL dataset directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New Git-ignored directory that will receive the complete run.",
    )
    parser.add_argument(
        "--git-revision",
        type=_git_revision_argument,
        required=True,
        help="Exact 40-character lowercase repository commit.",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help=(
            "Clean Git checkout containing the running code; defaults to "
            "the current directory."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume the exact matching partial run after validating every "
            "completed result and checkpoint bundle"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    experiment_runner: ExperimentRunner | None = None,
    utc_now: UtcClock | None = None,
    output_path_policy: OutputPathPolicy | None = None,
    repository_revision_policy: RepositoryRevisionPolicy | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Validate, stage, execute, and atomically publish one paired study."""

    args = build_parser().parse_args(argv)
    output_stream = stdout if stdout is not None else sys.stdout
    error_stream = stderr if stderr is not None else sys.stderr
    runner = (
        experiment_runner
        if experiment_runner is not None
        else run_srl_experiment
    )
    clock = utc_now if utc_now is not None else _canonical_utc_now
    path_policy = (
        output_path_policy
        if output_path_policy is not None
        else require_git_ignored_output_path
    )
    revision_policy = (
        repository_revision_policy
        if repository_revision_policy is not None
        else require_clean_repository_revision
    )

    phase = "config_validation"
    try:
        predicate_config = load_training_config(args.predicate_config)
        ablation_config = load_training_config(args.ablation_config)
        validate_paired_configs(predicate_config, ablation_config)
    except Exception as error:
        _emit_rejection(error_stream, phase=phase, error=error)
        return 2

    phase = "repository_preflight"
    try:
        if args.repository.is_symlink():
            raise ValueError("repository cannot be a symbolic link")
        repository = args.repository.resolve(strict=True)
        if not repository.is_dir():
            raise ValueError("repository must be a directory")
        revision_policy(repository, args.git_revision)
    except Exception as error:
        _emit_rejection(error_stream, phase=phase, error=error)
        return 2

    phase = "output_preflight"
    try:
        if args.output_root.is_symlink():
            raise FileExistsError("output root cannot be a symbolic link")
        output_root = args.output_root.resolve(strict=False)
        partial_root = _partial_root_for(output_root)
        lock_path = _output_lock_path_for(output_root)
        _require_new_directory_target(output_root, field="output root")
        if args.resume:
            _require_existing_directory(partial_root, field="partial output root")
        else:
            _require_new_directory_target(partial_root, field="partial output root")
        path_policy(output_root)
        path_policy(partial_root)
        _validate_output_lock_path(lock_path)
        path_policy(lock_path)
    except Exception as error:
        _emit_rejection(error_stream, phase=phase, error=error)
        return 2

    phase = "dataset_preflight"
    try:
        dataset_directory = args.dataset.resolve(strict=True)
        if not dataset_directory.is_dir():
            raise ValueError("prepared dataset must be a directory")
        prepared = read_prepared_dataset(dataset_directory)
        if (
            prepared.manifest.dataset_fingerprint
            != predicate_config.prepared_data_fingerprint
        ):
            raise ValueError(
                "prepared dataset fingerprint does not match paired configs"
            )
    except Exception as error:
        _emit_rejection(error_stream, phase=phase, error=error)
        return 2

    phase = "output_lock"
    output_lock: BinaryIO | None = None
    try:
        output_lock = _acquire_output_lock(lock_path)
        _require_new_directory_target(output_root, field="output root")
        if args.resume:
            _require_existing_directory(
                partial_root,
                field="partial output root",
            )
        else:
            _require_new_directory_target(
                partial_root,
                field="partial output root",
            )
    except Exception as error:
        if output_lock is not None:
            output_lock.close()
        _emit_rejection(error_stream, phase=phase, error=error)
        return 2

    run_plan = _paired_run_plan(predicate_config, ablation_config)
    cached_runs: dict[tuple[str, int], SRLRunResult] = {}
    completed_entries: list[dict[str, object]] = []
    discard_checkpoints: tuple[Path, ...] = ()
    discard_atomic_write_files: tuple[Path, ...] = ()
    discard_checkpoint_staging: tuple[Path, ...] = ()
    discard_checkpoint_tombstones: tuple[Path, ...] = ()
    recovered_run = False
    config_paths = _config_relative_paths()
    if args.resume:
        phase = "resume_preflight"
        try:
            resume_state = _load_resume_state(
                partial_root,
                output_root=output_root,
                repository=repository,
                dataset_directory=dataset_directory,
                dataset_fingerprint=prepared.manifest.dataset_fingerprint,
                git_revision=args.git_revision,
                predicate_config=predicate_config,
                ablation_config=ablation_config,
                run_plan=run_plan,
            )
            cached_runs = dict(resume_state.results)
            completed_entries = [dict(entry) for entry in resume_state.entries]
            discard_checkpoints = resume_state.discard_checkpoints
            discard_atomic_write_files = (
                resume_state.discard_atomic_write_files
            )
            discard_checkpoint_staging = (
                resume_state.discard_checkpoint_staging
            )
            discard_checkpoint_tombstones = (
                resume_state.discard_checkpoint_tombstones
            )
            recovered_run = resume_state.recovered_run
        except Exception as error:
            output_lock.close()
            _emit_rejection(error_stream, phase=phase, error=error)
            return 2

    phase = "output_staging"
    completed_runs = len(completed_entries)
    reused_runs = completed_runs
    completed_results = dict(cached_runs)
    runtime_reference = (
        next(iter(cached_runs.values())).metadata if cached_runs else None
    )
    active_variant: str | None = None
    active_seed: int | None = None
    try:
        if args.resume:
            for residue in discard_atomic_write_files:
                _remove_atomic_write_residue(residue)
            for staging_directory in discard_checkpoint_staging:
                _remove_checkpoint_staging_residue(staging_directory)
            for tombstone in discard_checkpoint_tombstones:
                _remove_checkpoint_tombstone(tombstone)
            (partial_root / FAILURE_MARKER_FILENAME).unlink(missing_ok=True)
            for checkpoint in discard_checkpoints:
                _discard_validated_checkpoint(checkpoint)
            if recovered_run:
                _write_partial_status(
                    partial_root,
                    output_root=output_root,
                    repository=repository,
                    dataset_directory=dataset_directory,
                    dataset_fingerprint=(
                        prepared.manifest.dataset_fingerprint
                    ),
                    git_revision=args.git_revision,
                    predicate_config=predicate_config,
                    ablation_config=ablation_config,
                    completed_entries=completed_entries,
                )
        else:
            partial_root.mkdir(parents=True, exist_ok=False)
            config_paths = _write_config_identities(
                partial_root,
                predicate_config=predicate_config,
                ablation_config=ablation_config,
            )
            _write_partial_status(
                partial_root,
                output_root=output_root,
                repository=repository,
                dataset_directory=dataset_directory,
                dataset_fingerprint=prepared.manifest.dataset_fingerprint,
                git_revision=args.git_revision,
                predicate_config=predicate_config,
                ablation_config=ablation_config,
                completed_entries=completed_entries,
            )

        phase = "paired_training"

        def run_one(config: TrainingConfig, seed: int) -> SRLRunResult:
            nonlocal active_seed, active_variant, completed_runs
            nonlocal runtime_reference
            cached = cached_runs.get((config.variant, seed))
            if cached is not None:
                return cached
            active_variant = config.variant
            active_seed = seed
            checkpoint = _checkpoint_path(partial_root, config.variant, seed)
            started_at = clock()
            context = RunContext(
                git_revision=args.git_revision,
                started_at=started_at,
            )
            result = runner(
                dataset_directory,
                config,
                checkpoint,
                seed=seed,
                context=context,
                utc_now=clock,
            )
            state_digest = _validate_completed_run(
                result,
                config=config,
                seed=seed,
                git_revision=args.git_revision,
                checkpoint=checkpoint,
            )
            if runtime_reference is not None:
                _validate_runtime_compatibility(
                    runtime_reference,
                    result.metadata,
                )
            if config.variant == "no_predicate_signal":
                predicate_result = completed_results.get(
                    ("predicate_signal", seed)
                )
                if predicate_result is None:
                    raise RuntimeError(
                        "ablation run has no completed predicate-signal pair"
                    )
                _validate_seed_pair(predicate_result, result, seed=seed)
            entry = _persist_completed_run(
                partial_root,
                result=result,
                config=config,
                seed=seed,
                checkpoint_state_digest=state_digest,
            )
            completed_entries.append(entry)
            completed_runs += 1
            completed_results[(config.variant, seed)] = result
            if runtime_reference is None:
                runtime_reference = result.metadata
            _write_partial_status(
                partial_root,
                output_root=output_root,
                repository=repository,
                dataset_directory=dataset_directory,
                dataset_fingerprint=prepared.manifest.dataset_fingerprint,
                git_revision=args.git_revision,
                predicate_config=predicate_config,
                ablation_config=ablation_config,
                completed_entries=completed_entries,
            )
            return result

        paired_result = run_paired_srl_experiments(
            predicate_config,
            ablation_config,
            run_one=run_one,
        )

        phase = "result_publication"
        result_path = partial_root / PAIRED_RESULTS_FILENAME
        result_bytes = paired_result.canonical_json_bytes()
        _require_matching_existing_file(result_path, result_bytes)
        _atomic_write_bytes(result_path, result_bytes)
        result_digest = hashlib.sha256(result_bytes).hexdigest()
        result_identity = {
            "result_identity_version": RESULT_IDENTITY_VERSION,
            "path": PAIRED_RESULTS_FILENAME,
            "digest": result_digest,
        }
        _require_matching_existing_json(
            partial_root / RESULT_IDENTITY_FILENAME,
            result_identity,
        )
        _atomic_write_json(
            partial_root / RESULT_IDENTITY_FILENAME,
            result_identity,
        )
        if completed_runs != len(run_plan):
            raise RuntimeError("paired run completion count does not reconcile")
        _write_partial_status(
            partial_root,
            output_root=output_root,
            repository=repository,
            dataset_directory=dataset_directory,
            dataset_fingerprint=prepared.manifest.dataset_fingerprint,
            git_revision=args.git_revision,
            predicate_config=predicate_config,
            ablation_config=ablation_config,
            completed_entries=completed_entries,
            status="complete",
        )
        (partial_root / FAILURE_MARKER_FILENAME).unlink(missing_ok=True)
        _require_new_directory_target(output_root, field="output root")
        partial_root.rename(output_root)
    except Exception as error:
        _write_failure_marker_safely(
            partial_root,
            phase=phase,
            error=error,
            completed_runs=completed_runs,
            active_variant=active_variant,
            active_seed=active_seed,
            predicate_config=predicate_config,
            ablation_config=ablation_config,
        )
        _emit_json(
            error_stream,
            {
                "status": "failed",
                "phase": phase,
                "error_type": type(error).__name__,
                "partial_output_root": str(partial_root),
                "completed_runs": completed_runs,
            },
        )
        return 1
    finally:
        output_lock.close()

    final_config_paths = {
        variant: str(output_root / relative_path)
        for variant, relative_path in config_paths.items()
    }
    _emit_json(
        output_stream,
        {
            "status": "complete",
            "output_root": str(output_root),
            "paired_results_path": str(
                output_root / PAIRED_RESULTS_FILENAME
            ),
            "paired_results_digest": result_digest,
            "paired_result_identity_path": str(
                output_root / RESULT_IDENTITY_FILENAME
            ),
            "config_identities_path": str(
                output_root / CONFIG_IDENTITIES_FILENAME
            ),
            "config_paths": final_config_paths,
            "predicate_config_digest": predicate_config.digest,
            "no_predicate_signal_config_digest": ablation_config.digest,
            "completed_runs": completed_runs,
            "reused_runs": reused_runs,
            "git_revision": args.git_revision,
            "repository": str(repository),
        },
    )
    return 0


@dataclass(frozen=True, slots=True)
class _ResumeState:
    results: tuple[tuple[tuple[str, int], SRLRunResult], ...]
    entries: tuple[dict[str, object], ...]
    discard_checkpoints: tuple[Path, ...]
    discard_atomic_write_files: tuple[Path, ...]
    discard_checkpoint_staging: tuple[Path, ...]
    discard_checkpoint_tombstones: tuple[Path, ...]
    recovered_run: bool


def _paired_run_plan(
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
) -> tuple[tuple[TrainingConfig, int], ...]:
    return tuple(
        (config, seed)
        for seed in predicate_config.paired_seeds
        for config in (predicate_config, ablation_config)
    )


def _config_relative_paths() -> dict[str, Path]:
    return {
        "predicate_signal": Path(CONFIG_DIRECTORY) / PREDICATE_CONFIG_FILENAME,
        "no_predicate_signal": Path(CONFIG_DIRECTORY) / ABLATION_CONFIG_FILENAME,
    }


def _checkpoint_relative_path(variant: str, seed: int) -> Path:
    return Path(CHECKPOINT_DIRECTORY) / variant / f"seed-{seed:010d}"


def _checkpoint_path(partial_root: Path, variant: str, seed: int) -> Path:
    return partial_root / _checkpoint_relative_path(variant, seed)


def _run_result_relative_path(variant: str, seed: int) -> Path:
    return Path(RUN_RESULTS_DIRECTORY) / variant / f"seed-{seed:010d}.json"


def _partial_status_payload(
    *,
    output_root: Path,
    repository: Path,
    dataset_directory: Path,
    dataset_fingerprint: str,
    git_revision: str,
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
    completed_entries: Sequence[dict[str, object]],
    status: str = "partial",
) -> dict[str, object]:
    if status not in {"partial", "complete"}:
        raise ValueError("run journal status must be partial or complete")
    return {
        "partial_status_version": PARTIAL_STATUS_VERSION,
        "status": status,
        "completed_runs": len(completed_entries),
        "expected_runs": 6,
        "output_root": str(output_root),
        "repository": str(repository),
        "dataset_directory": str(dataset_directory),
        "dataset_fingerprint": dataset_fingerprint,
        "git_revision": git_revision,
        "predicate_config_digest": predicate_config.digest,
        "no_predicate_signal_config_digest": ablation_config.digest,
        "runs": [dict(entry) for entry in completed_entries],
    }


def _write_partial_status(
    partial_root: Path,
    *,
    output_root: Path,
    repository: Path,
    dataset_directory: Path,
    dataset_fingerprint: str,
    git_revision: str,
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
    completed_entries: Sequence[dict[str, object]],
    status: str = "partial",
) -> None:
    _atomic_write_json(
        partial_root / PARTIAL_STATUS_FILENAME,
        _partial_status_payload(
            output_root=output_root,
            repository=repository,
            dataset_directory=dataset_directory,
            dataset_fingerprint=dataset_fingerprint,
            git_revision=git_revision,
            predicate_config=predicate_config,
            ablation_config=ablation_config,
            completed_entries=completed_entries,
            status=status,
        ),
    )


def _persist_completed_run(
    partial_root: Path,
    *,
    result: SRLRunResult,
    config: TrainingConfig,
    seed: int,
    checkpoint_state_digest: str,
) -> dict[str, object]:
    _require_sha256(
        checkpoint_state_digest,
        field="checkpoint_state_digest",
    )
    relative_result = _run_result_relative_path(config.variant, seed)
    result_path = partial_root / relative_result
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("completed run result already exists")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_bytes = result.canonical_json_bytes()
    _atomic_write_bytes(result_path, result_bytes)
    persisted = _parse_run_result(result_path.read_bytes())
    if persisted != result:
        raise RuntimeError("persisted completed run result changed after writing")
    return _completed_run_entry(
        config=config,
        seed=seed,
        result_bytes=result_bytes,
        checkpoint_state_digest=checkpoint_state_digest,
    )


def _completed_run_entry(
    *,
    config: TrainingConfig,
    seed: int,
    result_bytes: bytes,
    checkpoint_state_digest: str,
) -> dict[str, object]:
    return {
        "variant": config.variant,
        "seed": seed,
        "result_path": _run_result_relative_path(
            config.variant, seed
        ).as_posix(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "checkpoint_path": _checkpoint_relative_path(
            config.variant, seed
        ).as_posix(),
        "checkpoint_state_sha256": checkpoint_state_digest,
    }


def _load_resume_state(
    partial_root: Path,
    *,
    output_root: Path,
    repository: Path,
    dataset_directory: Path,
    dataset_fingerprint: str,
    git_revision: str,
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
    run_plan: tuple[tuple[TrainingConfig, int], ...],
) -> _ResumeState:
    status = _read_canonical_json_object(
        partial_root / PARTIAL_STATUS_FILENAME,
        context="partial status",
    )
    _require_exact_keys(status, _PARTIAL_STATUS_KEYS, context="partial status")
    journal_status = status["status"]
    if journal_status not in {"partial", "complete"}:
        raise ValueError("run journal status must be partial or complete")
    expected_values = {
        "partial_status_version": PARTIAL_STATUS_VERSION,
        "expected_runs": len(run_plan),
        "output_root": str(output_root),
        "repository": str(repository),
        "dataset_directory": str(dataset_directory),
        "dataset_fingerprint": dataset_fingerprint,
        "git_revision": git_revision,
        "predicate_config_digest": predicate_config.digest,
        "no_predicate_signal_config_digest": ablation_config.digest,
    }
    mismatches = sorted(
        key for key, value in expected_values.items() if status[key] != value
    )
    if mismatches:
        raise ValueError(
            "partial run provenance mismatch: " + ", ".join(mismatches)
        )
    if type(status["completed_runs"]) is not int:
        raise TypeError("partial completed_runs must be an integer")
    raw_entries = status["runs"]
    if not isinstance(raw_entries, list):
        raise TypeError("partial runs must be a JSON array")
    if status["completed_runs"] != len(raw_entries):
        raise ValueError("partial completed run count does not reconcile")
    if len(raw_entries) > len(run_plan):
        raise ValueError("partial run contains too many completed runs")
    if journal_status == "complete" and len(raw_entries) != len(run_plan):
        raise ValueError("complete run journal must contain every planned run")
    recorded_completed_runs = len(raw_entries)

    _validate_config_identities(
        partial_root,
        predicate_config=predicate_config,
        ablation_config=ablation_config,
    )

    results: list[tuple[tuple[str, int], SRLRunResult]] = []
    result_by_key: dict[tuple[str, int], SRLRunResult] = {}
    runtime_reference: RunMetadata | None = None
    entries: list[dict[str, object]] = []
    expected_files = {
        Path(PARTIAL_STATUS_FILENAME),
        Path(CONFIG_IDENTITIES_FILENAME),
        *_config_relative_paths().values(),
    }
    for index, raw_entry in enumerate(raw_entries):
        entry = _require_json_object(raw_entry, context="completed run entry")
        _require_exact_keys(
            entry, _COMPLETED_RUN_KEYS, context="completed run entry"
        )
        if not isinstance(entry["variant"], str):
            raise TypeError("completed run variant must be a string")
        if type(entry["seed"]) is not int:
            raise TypeError("completed run seed must be an integer")
        for field in ("result_path", "checkpoint_path"):
            if not isinstance(entry[field], str):
                raise TypeError(f"completed run {field} must be a string")
        config, seed = run_plan[index]
        expected_result = _run_result_relative_path(config.variant, seed)
        expected_checkpoint = _checkpoint_relative_path(config.variant, seed)
        expected_entry_values = {
            "variant": config.variant,
            "seed": seed,
            "result_path": expected_result.as_posix(),
            "checkpoint_path": expected_checkpoint.as_posix(),
        }
        entry_mismatches = sorted(
            key
            for key, value in expected_entry_values.items()
            if entry[key] != value
        )
        if entry_mismatches:
            raise ValueError(
                "completed run plan mismatch: " + ", ".join(entry_mismatches)
            )
        _require_sha256(entry["result_sha256"], field="result_sha256")
        _require_sha256(
            entry["checkpoint_state_sha256"],
            field="checkpoint_state_sha256",
        )
        result_path = partial_root / expected_result
        result_bytes = _read_regular_file(result_path, context="completed run result")
        if hashlib.sha256(result_bytes).hexdigest() != entry["result_sha256"]:
            raise ValueError("completed run result SHA-256 mismatch")
        result = _parse_run_result(result_bytes)
        state_digest = _validate_completed_run(
            result,
            config=config,
            seed=seed,
            git_revision=git_revision,
            checkpoint=partial_root / expected_checkpoint,
        )
        if state_digest != entry["checkpoint_state_sha256"]:
            raise ValueError("completed checkpoint state digest mismatch")
        if runtime_reference is not None:
            _validate_runtime_compatibility(runtime_reference, result.metadata)
        if config.variant == "no_predicate_signal":
            predicate_result = result_by_key.get(("predicate_signal", seed))
            if predicate_result is None:
                raise ValueError(
                    "completed ablation has no predicate-signal pair"
                )
            _validate_seed_pair(predicate_result, result, seed=seed)
        if runtime_reference is None:
            runtime_reference = result.metadata
        results.append(((config.variant, seed), result))
        result_by_key[(config.variant, seed)] = result
        entries.append(dict(entry))
        expected_files.add(expected_result)
        for filename in (
            METADATA_FILENAME,
            LABEL_CONFIG_FILENAME,
            STATE_DICT_FILENAME,
        ):
            expected_files.add(expected_checkpoint / filename)

    next_uncommitted_checkpoint: Path | None = None
    next_uncommitted_result: Path | None = None
    if recorded_completed_runs < len(run_plan):
        recorded_config, recorded_seed = run_plan[recorded_completed_runs]
        next_uncommitted_checkpoint = _checkpoint_path(
            partial_root,
            recorded_config.variant,
            recorded_seed,
        )
        next_uncommitted_result = (
            partial_root
            / _run_result_relative_path(
                recorded_config.variant,
                recorded_seed,
            )
        )

    discard_checkpoints: list[Path] = []
    recovered_run = False
    if len(entries) < len(run_plan):
        next_config, next_seed = run_plan[len(entries)]
        next_result_relative = _run_result_relative_path(
            next_config.variant, next_seed
        )
        next_checkpoint_relative = _checkpoint_relative_path(
            next_config.variant, next_seed
        )
        next_result_path = partial_root / next_result_relative
        next_checkpoint_path = partial_root / next_checkpoint_relative
        result_exists = next_result_path.exists() or next_result_path.is_symlink()
        checkpoint_exists = (
            next_checkpoint_path.exists()
            or next_checkpoint_path.is_symlink()
        )
        if result_exists and not checkpoint_exists:
            raise ValueError(
                "uncommitted run result has no matching checkpoint bundle"
            )
        if checkpoint_exists:
            if result_exists:
                result_bytes = _read_regular_file(
                    next_result_path,
                    context="uncommitted completed run result",
                )
                result = _parse_run_result(result_bytes)
                state_digest = _validate_completed_run(
                    result,
                    config=next_config,
                    seed=next_seed,
                    git_revision=git_revision,
                    checkpoint=next_checkpoint_path,
                )
                if runtime_reference is not None:
                    _validate_runtime_compatibility(
                        runtime_reference,
                        result.metadata,
                    )
                if next_config.variant == "no_predicate_signal":
                    predicate_result = result_by_key.get(
                        ("predicate_signal", next_seed)
                    )
                    if predicate_result is None:
                        raise ValueError(
                            "recovered ablation has no predicate-signal pair"
                        )
                    _validate_seed_pair(
                        predicate_result,
                        result,
                        seed=next_seed,
                    )
                entry = _completed_run_entry(
                    config=next_config,
                    seed=next_seed,
                    result_bytes=result_bytes,
                    checkpoint_state_digest=state_digest,
                )
                results.append(((next_config.variant, next_seed), result))
                result_by_key[(next_config.variant, next_seed)] = result
                entries.append(entry)
                if runtime_reference is None:
                    runtime_reference = result.metadata
                expected_files.add(next_result_relative)
                recovered_run = True
            else:
                _validate_uncommitted_checkpoint(
                    next_checkpoint_path,
                    config=next_config,
                    seed=next_seed,
                    git_revision=git_revision,
                )
                discard_checkpoints.append(next_checkpoint_path)
            for filename in (
                METADATA_FILENAME,
                LABEL_CONFIG_FILENAME,
                STATE_DICT_FILENAME,
            ):
                expected_files.add(next_checkpoint_relative / filename)

    failure_path = partial_root / FAILURE_MARKER_FILENAME
    if failure_path.exists() or failure_path.is_symlink():
        _validate_failure_marker(
            failure_path,
            allowed_completed_runs=frozenset(
                {recorded_completed_runs, len(entries)}
            ),
            predicate_config=predicate_config,
            ablation_config=ablation_config,
            run_plan=run_plan,
        )
        expected_files.add(Path(FAILURE_MARKER_FILENAME))

    if len(entries) == len(run_plan):
        paired_result = _paired_result_from_completed_runs(
            predicate_config,
            ablation_config,
            result_by_key,
        )
        expected_result_bytes = paired_result.canonical_json_bytes()
        expected_result_identity = {
            "result_identity_version": RESULT_IDENTITY_VERSION,
            "path": PAIRED_RESULTS_FILENAME,
            "digest": hashlib.sha256(expected_result_bytes).hexdigest(),
        }
        publication_paths = {
            PAIRED_RESULTS_FILENAME: expected_result_bytes,
            RESULT_IDENTITY_FILENAME: _canonical_json_bytes(
                expected_result_identity
            ),
        }
        for filename, expected_payload in publication_paths.items():
            path = partial_root / filename
            if path.exists() or path.is_symlink():
                actual = _read_regular_file(
                    path,
                    context="partial publication artifact",
                )
                if actual != expected_payload:
                    raise ValueError(
                        "partial publication artifact does not match journal"
                    )
                expected_files.add(Path(filename))
            elif journal_status == "complete":
                raise ValueError(
                    "complete run journal is missing publication artifacts"
                )
    elif any(
        (partial_root / filename).exists()
        or (partial_root / filename).is_symlink()
        for filename in (PAIRED_RESULTS_FILENAME, RESULT_IDENTITY_FILENAME)
    ):
        raise ValueError("partial publication artifact exists before all runs complete")

    allowed_atomic_targets = {
        partial_root / PARTIAL_STATUS_FILENAME,
        partial_root / FAILURE_MARKER_FILENAME,
    }
    if next_uncommitted_result is not None:
        allowed_atomic_targets.add(next_uncommitted_result)
    if len(entries) == len(run_plan):
        allowed_atomic_targets.update(
            {
                partial_root / PAIRED_RESULTS_FILENAME,
                partial_root / RESULT_IDENTITY_FILENAME,
            }
        )
    (
        atomic_residues,
        checkpoint_staging_residues,
        checkpoint_tombstones,
    ) = (
        _validate_partial_file_inventory(
            partial_root,
            expected_files,
            allowed_atomic_targets=allowed_atomic_targets,
            next_checkpoint_target=next_uncommitted_checkpoint,
        )
    )
    if checkpoint_tombstones and (
        checkpoint_staging_residues
        or len(checkpoint_tombstones) != 1
        or next_uncommitted_checkpoint is None
        or next_uncommitted_checkpoint.exists()
        or next_uncommitted_checkpoint.is_symlink()
        or next_uncommitted_result is None
        or next_uncommitted_result.exists()
        or next_uncommitted_result.is_symlink()
    ):
        raise ValueError(
            "checkpoint tombstone conflicts with canonical run artifacts"
        )
    return _ResumeState(
        results=tuple(results),
        entries=tuple(entries),
        discard_checkpoints=tuple(discard_checkpoints),
        discard_atomic_write_files=atomic_residues,
        discard_checkpoint_staging=checkpoint_staging_residues,
        discard_checkpoint_tombstones=checkpoint_tombstones,
        recovered_run=recovered_run,
    )


def _validate_completed_run(
    result: SRLRunResult,
    *,
    config: TrainingConfig,
    seed: int,
    git_revision: str,
    checkpoint: Path,
) -> str:
    if not isinstance(result, SRLRunResult):
        raise TypeError("experiment runner must return SRLRunResult")
    _validate_run_metadata(
        result.metadata,
        config=config,
        seed=seed,
        git_revision=git_revision,
    )
    if result.checkpoint_selection_metric != config.checkpoint_selection_metric:
        raise ValueError("completed run checkpoint-selection metric mismatch")
    if len(result.epochs) != config.epochs:
        raise ValueError("completed run epoch count mismatch")
    selection_values = tuple(
        (
            epoch.development.evaluation.arguments.f1
            if config.checkpoint_selection_metric == "development_argument_f1"
            else epoch.development.loss
        )
        for epoch in result.epochs
    )
    if any(
        epoch.selection_value != selection_value
        for epoch, selection_value in zip(
            result.epochs, selection_values, strict=True
        )
    ):
        raise ValueError("completed run selection values do not reconcile")
    selected_value = (
        max(selection_values)
        if config.checkpoint_selection_metric == "development_argument_f1"
        else min(selection_values)
    )
    if result.best_epoch != selection_values.index(selected_value) + 1:
        raise ValueError("completed run best epoch does not reconcile")
    validated = validate_checkpoint_bundle(
        checkpoint,
        expected_metadata=result.metadata,
        expected_config=config,
        expected_label_config=CheckpointLabelConfig.from_labels(
            result.metadata.labels
        ),
    )
    return validated.state_dict_sha256


def _validate_uncommitted_checkpoint(
    checkpoint: Path,
    *,
    config: TrainingConfig,
    seed: int,
    git_revision: str,
) -> None:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("uncommitted checkpoint must be a real directory")
    metadata_payload = _read_regular_file(
        checkpoint / METADATA_FILENAME,
        context="uncommitted checkpoint metadata",
    )
    checkpoint_metadata = CheckpointMetadata.from_canonical_json_bytes(
        metadata_payload
    )
    metadata = checkpoint_metadata.run_metadata
    _validate_run_metadata(
        metadata,
        config=config,
        seed=seed,
        git_revision=git_revision,
    )
    validate_checkpoint_bundle(
        checkpoint,
        expected_metadata=metadata,
        expected_config=config,
        expected_label_config=CheckpointLabelConfig.from_labels(
            metadata.labels
        ),
    )


def _validate_run_metadata(
    metadata: RunMetadata,
    *,
    config: TrainingConfig,
    seed: int,
    git_revision: str,
) -> None:
    metadata.assert_config_compatible(config)
    if metadata.seed != seed:
        raise ValueError("completed run seed does not match its plan")
    if metadata.git_revision != git_revision:
        raise ValueError("completed run Git revision mismatch")
    counts = dict(metadata.counts)
    if counts.get("epochs_completed") != config.epochs:
        raise ValueError("completed run metadata epoch count mismatch")


def _validate_runtime_compatibility(
    reference: RunMetadata,
    candidate: RunMetadata,
) -> None:
    if not isinstance(reference, RunMetadata) or not isinstance(
        candidate, RunMetadata
    ):
        raise TypeError("runtime compatibility requires run metadata")
    fields = (
        "package_versions",
        "hardware",
        "resolved_device",
        "labels",
        "counts",
        "drop_stats",
    )
    mismatches = [
        field
        for field in fields
        if getattr(reference, field) != getattr(candidate, field)
    ]
    if mismatches:
        raise ValueError(
            "run runtime identity mismatch: " + ", ".join(mismatches)
        )


def _validate_seed_pair(
    predicate_signal: SRLRunResult,
    no_predicate_signal: SRLRunResult,
    *,
    seed: int,
) -> None:
    PairedSeedResult(
        seed=seed,
        predicate_signal=predicate_signal,
        no_predicate_signal=no_predicate_signal,
        development_argument_f1_delta=(
            predicate_signal.best_development.evaluation.arguments.f1
            - no_predicate_signal.best_development.evaluation.arguments.f1
        ),
        test_argument_f1_delta=(
            predicate_signal.test.evaluation.arguments.f1
            - no_predicate_signal.test.evaluation.arguments.f1
        ),
    )


def _paired_result_from_completed_runs(
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
    results: dict[tuple[str, int], SRLRunResult],
) -> PairedExperimentResult:
    def completed_run(config: TrainingConfig, seed: int) -> SRLRunResult:
        try:
            return results[(config.variant, seed)]
        except KeyError as error:
            raise ValueError(
                "complete run journal is missing a planned result"
            ) from error

    return run_paired_srl_experiments(
        predicate_config,
        ablation_config,
        run_one=completed_run,
    )


def _validate_config_identities(
    partial_root: Path,
    *,
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
) -> None:
    paths = _config_relative_paths()
    expected_configs = {
        "predicate_signal": predicate_config,
        "no_predicate_signal": ablation_config,
    }
    for variant, config in expected_configs.items():
        actual = _read_regular_file(
            partial_root / paths[variant], context=f"{variant} stored config"
        )
        if actual != config.canonical_json_bytes():
            raise ValueError(f"stored {variant} configuration mismatch")
    expected_identity = {
        "config_identity_version": CONFIG_IDENTITY_VERSION,
        "predicate_signal": {
            "path": paths["predicate_signal"].as_posix(),
            "digest": predicate_config.digest,
        },
        "no_predicate_signal": {
            "path": paths["no_predicate_signal"].as_posix(),
            "digest": ablation_config.digest,
        },
    }
    identity = _read_canonical_json_object(
        partial_root / CONFIG_IDENTITIES_FILENAME,
        context="config identities",
    )
    if identity != expected_identity:
        raise ValueError("stored config identities mismatch")


def _validate_partial_file_inventory(
    partial_root: Path,
    expected_files: set[Path],
    *,
    allowed_atomic_targets: set[Path],
    next_checkpoint_target: Path | None,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    actual_files: set[Path] = set()
    actual_directories: set[Path] = set()
    atomic_residues: list[Path] = []
    checkpoint_staging_residues: list[Path] = []
    checkpoint_tombstones: list[Path] = []

    def visit(directory: Path) -> None:
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError("partial run cannot contain symbolic links")
            relative = path.relative_to(partial_root)
            if path.is_dir():
                if (
                    next_checkpoint_target is not None
                    and _is_checkpoint_tombstone(
                        path,
                        target=next_checkpoint_target,
                    )
                ):
                    _validate_checkpoint_tombstone_contents(path)
                    checkpoint_tombstones.append(path)
                    continue
                if (
                    next_checkpoint_target is not None
                    and _is_checkpoint_staging_residue(
                        path,
                        target=next_checkpoint_target,
                    )
                ):
                    checkpoint_staging_residues.append(path)
                    continue
                actual_directories.add(relative)
                visit(path)
            elif path.is_file():
                if any(
                    _is_atomic_write_residue(path, target=target)
                    for target in allowed_atomic_targets
                ):
                    atomic_residues.append(path)
                else:
                    actual_files.add(relative)
            else:
                raise ValueError("partial run contains a non-file artifact")

    visit(partial_root)
    if actual_files != expected_files:
        missing = sorted(str(path) for path in expected_files - actual_files)
        unknown = sorted(str(path) for path in actual_files - expected_files)
        details: list[str] = []
        if missing:
            details.append("missing files: " + ", ".join(missing))
        if unknown:
            details.append("unknown files: " + ", ".join(unknown))
        raise ValueError("invalid partial run layout; " + "; ".join(details))

    allowed_directories: set[Path] = set()
    directory_sources = {
        *(partial_root / path for path in expected_files),
        *allowed_atomic_targets,
    }
    if next_checkpoint_target is not None:
        directory_sources.add(next_checkpoint_target)
    for source in directory_sources:
        relative_parent = source.parent.relative_to(partial_root)
        while relative_parent != Path("."):
            allowed_directories.add(relative_parent)
            relative_parent = relative_parent.parent
    unknown_directories = sorted(
        str(path) for path in actual_directories - allowed_directories
    )
    if unknown_directories:
        raise ValueError(
            "invalid partial run layout; unknown directories: "
            + ", ".join(unknown_directories)
        )
    return (
        tuple(sorted(atomic_residues)),
        tuple(sorted(checkpoint_staging_residues)),
        tuple(sorted(checkpoint_tombstones)),
    )


def _is_atomic_write_residue(path: Path, *, target: Path) -> bool:
    if path.parent != target.parent:
        return False
    pattern = rf"\.{re.escape(target.name)}\.[a-z0-9_]{{8}}\.write-partial"
    return re.fullmatch(pattern, path.name) is not None


def _is_checkpoint_staging_residue(path: Path, *, target: Path) -> bool:
    if path.parent != target.parent:
        return False
    pattern = rf"\.{re.escape(target.name)}\.[a-z0-9_]{{8}}\.run-partial"
    return re.fullmatch(pattern, path.name) is not None


def _checkpoint_tombstone_for(target: Path) -> Path:
    return target.with_name(f".{target.name}.discard-tombstone")


def _is_checkpoint_tombstone(path: Path, *, target: Path) -> bool:
    return path == _checkpoint_tombstone_for(target)


def _validate_checkpoint_tombstone_contents(path: Path) -> None:
    allowed_members = {
        METADATA_FILENAME,
        LABEL_CONFIG_FILENAME,
        STATE_DICT_FILENAME,
    }
    for member in path.iterdir():
        if member.is_symlink() or not member.is_file():
            raise ValueError(
                "checkpoint tombstone can contain only regular bundle files"
            )
        if member.name not in allowed_members:
            raise ValueError("checkpoint tombstone contains an unknown artifact")


def _remove_atomic_write_residue(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("atomic-write residue must remain a regular file")
    path.unlink()


def _remove_checkpoint_staging_residue(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint staging residue must remain a real directory")
    shutil.rmtree(path)


def _discard_validated_checkpoint(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("discardable checkpoint must remain a real directory")
    tombstone = _checkpoint_tombstone_for(path)
    if tombstone.exists() or tombstone.is_symlink():
        raise FileExistsError("checkpoint discard tombstone already exists")
    path.rename(tombstone)
    _remove_checkpoint_tombstone(tombstone)


def _remove_checkpoint_tombstone(path: Path) -> None:
    pattern = r"\.seed-[0-9]{10}\.discard-tombstone"
    if re.fullmatch(pattern, path.name) is None:
        raise ValueError("invalid checkpoint tombstone name")
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint tombstone must remain a real directory")
    _validate_checkpoint_tombstone_contents(path)
    shutil.rmtree(path)


def _validate_failure_marker(
    path: Path,
    *,
    allowed_completed_runs: frozenset[int],
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
    run_plan: tuple[tuple[TrainingConfig, int], ...],
) -> None:
    marker = _read_canonical_json_object(path, context="failure marker")
    base_keys = {
        "failure_marker_version",
        "status",
        "phase",
        "error_type",
        "completed_runs",
        "expected_runs",
        "predicate_config_digest",
        "no_predicate_signal_config_digest",
    }
    active_keys = {"active_variant", "active_seed"}
    if frozenset(marker) not in {
        frozenset(base_keys),
        frozenset(base_keys | active_keys),
    }:
        raise ValueError("failure marker has unexpected or missing fields")
    expected = {
        "failure_marker_version": FAILURE_MARKER_VERSION,
        "status": "failed",
        "expected_runs": len(run_plan),
        "predicate_config_digest": predicate_config.digest,
        "no_predicate_signal_config_digest": ablation_config.digest,
    }
    if any(marker[key] != value for key, value in expected.items()):
        raise ValueError("failure marker does not match partial run provenance")
    if (
        type(marker["completed_runs"]) is not int
        or marker["completed_runs"] not in allowed_completed_runs
    ):
        raise ValueError("failure marker completed run count is inconsistent")
    for key in ("phase", "error_type"):
        if not isinstance(marker[key], str) or not marker[key]:
            raise ValueError(f"failure marker {key} must be nonempty")
    if active_keys.issubset(marker):
        active = (marker["active_variant"], marker["active_seed"])
        planned = {(config.variant, seed) for config, seed in run_plan}
        if active not in planned:
            raise ValueError("failure marker active run is outside the plan")


def _require_matching_existing_file(path: Path, expected: bytes) -> None:
    if path.exists() or path.is_symlink():
        actual = _read_regular_file(path, context="existing publication result")
        if actual != expected:
            raise ValueError("existing publication result does not match resumed runs")


def _require_matching_existing_json(
    path: Path, expected: dict[str, object]
) -> None:
    if path.exists() or path.is_symlink():
        actual = _read_canonical_json_object(
            path, context="existing publication identity"
        )
        if actual != expected:
            raise ValueError(
                "existing publication identity does not match resumed runs"
            )


def _require_existing_directory(path: Path, *, field: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"{field} does not exist as a real directory")


def require_git_ignored_output_path(path: Path) -> None:
    """Reject a path unless Git confirms it is ignored in this worktree."""

    if not isinstance(path, Path):
        raise TypeError("ignored output path must be a Path")
    anchor = path.parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    result = subprocess.run(
        [
            "git",
            "-C",
            str(anchor),
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 1:
        raise ValueError("output path must be covered by a Git ignore rule")
    if result.returncode != 0:
        raise RuntimeError("could not verify the Git ignore policy")


def require_clean_repository_revision(
    repository: Path,
    git_revision: str,
    *,
    source_path: Path | None = None,
) -> None:
    """Pin provenance to the clean checkout providing this CLI module."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a Path")
    if _GIT_REVISION_RE.fullmatch(git_revision) is None:
        raise ValueError("git revision must be a 40-character lowercase commit")
    repository = repository.resolve(strict=True)
    running_source = (
        source_path.resolve(strict=True)
        if source_path is not None
        else Path(__file__).resolve(strict=True)
    )
    expected_source = (
        repository
        / "src"
        / "semantic_action_extractor"
        / "srl"
        / "training_cli.py"
    ).resolve(strict=True)
    if running_source != expected_source:
        raise ValueError("running training CLI is not from the declared repository")

    top_level = Path(
        _run_git(repository, "rev-parse", "--show-toplevel").strip()
    ).resolve(strict=True)
    if top_level != repository:
        raise ValueError("repository must be the Git worktree root")
    head = _run_git(repository, "rev-parse", "--verify", "HEAD").strip()
    if head != git_revision:
        raise ValueError("supplied git revision does not match repository HEAD")
    tracked_status = _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if tracked_status:
        raise ValueError("repository index and tracked worktree must be clean")
    relative_source = expected_source.relative_to(repository).as_posix()
    _run_git(
        repository,
        "ls-files",
        "--error-unmatch",
        "--",
        relative_source,
    )


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Git repository provenance check failed")
    return result.stdout


def _write_config_identities(
    partial_root: Path,
    *,
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
) -> dict[str, Path]:
    config_directory = partial_root / CONFIG_DIRECTORY
    config_directory.mkdir()
    relative_paths = _config_relative_paths()
    _atomic_write_bytes(
        partial_root / relative_paths["predicate_signal"],
        predicate_config.canonical_json_bytes(),
    )
    _atomic_write_bytes(
        partial_root / relative_paths["no_predicate_signal"],
        ablation_config.canonical_json_bytes(),
    )
    _atomic_write_json(
        partial_root / CONFIG_IDENTITIES_FILENAME,
        {
            "config_identity_version": CONFIG_IDENTITY_VERSION,
            "predicate_signal": {
                "path": relative_paths["predicate_signal"].as_posix(),
                "digest": predicate_config.digest,
            },
            "no_predicate_signal": {
                "path": relative_paths["no_predicate_signal"].as_posix(),
                "digest": ablation_config.digest,
            },
        },
    )
    return relative_paths


def _write_failure_marker_safely(
    partial_root: Path,
    *,
    phase: str,
    error: Exception,
    completed_runs: int,
    active_variant: str | None,
    active_seed: int | None,
    predicate_config: TrainingConfig,
    ablation_config: TrainingConfig,
) -> None:
    if not partial_root.is_dir() or partial_root.is_symlink():
        return
    marker: dict[str, object] = {
        "failure_marker_version": FAILURE_MARKER_VERSION,
        "status": "failed",
        "phase": phase,
        "error_type": type(error).__name__,
        "completed_runs": completed_runs,
        "expected_runs": 6,
        "predicate_config_digest": predicate_config.digest,
        "no_predicate_signal_config_digest": ablation_config.digest,
    }
    if active_variant is not None:
        marker["active_variant"] = active_variant
    if active_seed is not None:
        marker["active_seed"] = active_seed
    try:
        _atomic_write_json(partial_root / FAILURE_MARKER_FILENAME, marker)
    except OSError:
        return


def _partial_root_for(output_root: Path) -> Path:
    if not output_root.name:
        raise ValueError("output root must have a directory name")
    return output_root.with_name(f"{output_root.name}.partial")


def _output_lock_path_for(output_root: Path) -> Path:
    if not output_root.name:
        raise ValueError("output root must have a directory name")
    return output_root.with_name(f".{output_root.name}.lock")


def _validate_output_lock_path(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("output lock cannot be a symbolic link")
    if path.exists() and not path.is_file():
        raise ValueError("output lock must be a regular file")


def _acquire_output_lock(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("output lock must be a single-link regular file")
        if metadata.st_uid != os.geteuid():
            raise PermissionError("output lock must be owned by the current user")
        handle = os.fdopen(descriptor, "a+b")
    except Exception:
        os.close(descriptor)
        raise
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            raise BlockingIOError(
                errno.EWOULDBLOCK,
                "output is locked by another process",
                str(path),
            ) from error
        raise
    return handle


def _require_new_directory_target(path: Path, *, field: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{field} already exists")


def _git_revision_argument(value: str) -> str:
    if _GIT_REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "must be a 40-character lowercase hexadecimal commit"
        )
    return value


def _parse_run_result(payload: bytes) -> SRLRunResult:
    value = _decode_json_object(payload, context="completed run result")
    _require_exact_keys(
        value,
        frozenset(
            {
                "result_version",
                "metadata",
                "best_epoch",
                "checkpoint_selection_metric",
                "best_development",
                "test",
                "epochs",
            }
        ),
        context="completed run result",
    )
    if value["result_version"] != EXPERIMENT_RESULT_VERSION:
        raise ValueError("unsupported completed run result version")
    raw_epochs = value["epochs"]
    if not isinstance(raw_epochs, list):
        raise TypeError("completed run epochs must be a JSON array")
    epochs: list[EpochSummary] = []
    for raw_epoch in raw_epochs:
        epoch = _require_json_object(raw_epoch, context="epoch summary")
        _require_exact_keys(
            epoch,
            frozenset(
                {
                    "epoch",
                    "training_loss",
                    "development",
                    "selection_value",
                }
            ),
            context="epoch summary",
        )
        epochs.append(
            EpochSummary(
                epoch=_nonnegative_int(epoch["epoch"], field="epoch", minimum=1),
                training_loss=_finite_float(
                    epoch["training_loss"], field="training_loss"
                ),
                development=_parse_evaluation_summary(epoch["development"]),
                selection_value=_finite_float(
                    epoch["selection_value"], field="selection_value"
                ),
            )
        )
    metric = value["checkpoint_selection_metric"]
    if not isinstance(metric, str):
        raise TypeError("checkpoint_selection_metric must be a string")
    result = SRLRunResult(
        result_version=value["result_version"],
        metadata=RunMetadata.from_dict(value["metadata"]),
        best_epoch=_nonnegative_int(
            value["best_epoch"], field="best_epoch", minimum=1
        ),
        checkpoint_selection_metric=metric,
        best_development=_parse_evaluation_summary(value["best_development"]),
        test=_parse_evaluation_summary(value["test"]),
        epochs=tuple(epochs),
    )
    if result.canonical_json_bytes() != payload:
        raise ValueError("completed run result JSON is not canonical")
    return result


def _parse_evaluation_summary(value: object) -> EvaluationSummary:
    summary = _require_json_object(value, context="evaluation summary")
    _require_exact_keys(
        summary,
        frozenset({"loss", "arguments", "per_role", "predicate", "token_accuracy"}),
        context="evaluation summary",
    )
    arguments = _parse_span_metrics(summary["arguments"])
    raw_roles = summary["per_role"]
    if not isinstance(raw_roles, list):
        raise TypeError("per_role must be a JSON array")
    roles = tuple(_parse_role_metrics(value) for value in raw_roles)
    role_labels = tuple(role.label for role in roles)
    if role_labels != tuple(sorted(set(role_labels))):
        raise ValueError("per_role labels must be unique and sorted")
    if (
        sum(role.true_positives for role in roles) != arguments.true_positives
        or sum(role.predicted for role in roles) != arguments.predicted
        or sum(role.gold for role in roles) != arguments.gold
    ):
        raise ValueError("per-role counts do not reconcile to argument counts")
    predicate = _parse_predicate_diagnostics(summary["predicate"])
    token_accuracy = _parse_token_accuracy(summary["token_accuracy"])
    return EvaluationSummary(
        loss=_finite_float(summary["loss"], field="evaluation loss"),
        evaluation=SuppliedPredicateEvaluation(
            arguments=arguments,
            per_role=roles,
            predicate=predicate,
            token_accuracy=token_accuracy,
        ),
    )


def _parse_span_metrics(value: object) -> SpanMetrics:
    metrics = _require_json_object(value, context="argument metrics")
    _require_exact_keys(
        metrics,
        frozenset(
            {
                "true_positives",
                "predicted",
                "gold",
                "precision",
                "recall",
                "f1",
                "repaired_prediction_tags",
            }
        ),
        context="argument metrics",
    )
    true_positives = _nonnegative_int(
        metrics["true_positives"], field="argument true_positives"
    )
    predicted = _nonnegative_int(metrics["predicted"], field="argument predicted")
    gold = _nonnegative_int(metrics["gold"], field="argument gold")
    precision, recall, f1 = _validated_ratios(
        true_positives=true_positives,
        predicted=predicted,
        gold=gold,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        field="argument",
    )
    return SpanMetrics(
        true_positives=true_positives,
        predicted=predicted,
        gold=gold,
        precision=precision,
        recall=recall,
        f1=f1,
        repaired_prediction_tags=_nonnegative_int(
            metrics["repaired_prediction_tags"],
            field="repaired_prediction_tags",
        ),
    )


def _parse_role_metrics(value: object) -> RoleSpanMetrics:
    metrics = _require_json_object(value, context="role metrics")
    _require_exact_keys(
        metrics,
        frozenset(
            {
                "label",
                "true_positives",
                "predicted",
                "gold",
                "precision",
                "recall",
                "f1",
            }
        ),
        context="role metrics",
    )
    label = metrics["label"]
    if not isinstance(label, str) or not label:
        raise ValueError("role label must be a nonempty string")
    true_positives = _nonnegative_int(
        metrics["true_positives"], field=f"{label} true_positives"
    )
    predicted = _nonnegative_int(metrics["predicted"], field=f"{label} predicted")
    gold = _nonnegative_int(metrics["gold"], field=f"{label} gold")
    precision, recall, f1 = _validated_ratios(
        true_positives=true_positives,
        predicted=predicted,
        gold=gold,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        field=label,
    )
    return RoleSpanMetrics(
        label=label,
        true_positives=true_positives,
        predicted=predicted,
        gold=gold,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _parse_predicate_diagnostics(value: object) -> PredicateDiagnostics:
    diagnostics = _require_json_object(value, context="predicate diagnostics")
    keys = frozenset(
        {
            "examples",
            "correct_anchors",
            "missing_anchors",
            "spurious_predicate_words",
            "argument_spans_overlapping_predicate",
        }
    )
    _require_exact_keys(diagnostics, keys, context="predicate diagnostics")
    parsed = {
        key: _nonnegative_int(diagnostics[key], field=key) for key in keys
    }
    if parsed["correct_anchors"] + parsed["missing_anchors"] != parsed["examples"]:
        raise ValueError("predicate anchor counts do not reconcile")
    return PredicateDiagnostics(**parsed)


def _parse_token_accuracy(value: object) -> TokenAccuracy:
    metrics = _require_json_object(value, context="token accuracy")
    _require_exact_keys(
        metrics,
        frozenset({"correct", "total", "accuracy"}),
        context="token accuracy",
    )
    correct = _nonnegative_int(metrics["correct"], field="token correct")
    total = _nonnegative_int(metrics["total"], field="token total")
    if correct > total:
        raise ValueError("token correct count exceeds total")
    accuracy = _finite_float(metrics["accuracy"], field="token accuracy")
    if accuracy != _safe_ratio(correct, total):
        raise ValueError("token accuracy does not reconcile")
    return TokenAccuracy(correct=correct, total=total, accuracy=accuracy)


def _validated_ratios(
    *,
    true_positives: int,
    predicted: int,
    gold: int,
    precision: object,
    recall: object,
    f1: object,
    field: str,
) -> tuple[float, float, float]:
    if true_positives > predicted or true_positives > gold:
        raise ValueError(f"{field} true positives exceed support")
    parsed_precision = _finite_float(precision, field=f"{field} precision")
    parsed_recall = _finite_float(recall, field=f"{field} recall")
    parsed_f1 = _finite_float(f1, field=f"{field} f1")
    expected_precision = _safe_ratio(true_positives, predicted)
    expected_recall = _safe_ratio(true_positives, gold)
    expected_f1 = _safe_ratio(
        2 * expected_precision * expected_recall,
        expected_precision + expected_recall,
    )
    if (parsed_precision, parsed_recall, parsed_f1) != (
        expected_precision,
        expected_recall,
        expected_f1,
    ):
        raise ValueError(f"{field} metrics do not reconcile")
    return parsed_precision, parsed_recall, parsed_f1


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _nonnegative_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{field} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _read_regular_file(path: Path, *, context: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular file")
    return path.read_bytes()


def _read_canonical_json_object(path: Path, *, context: str) -> dict[str, Any]:
    payload = _read_regular_file(path, context=context)
    value = _decode_json_object(payload, context=context)
    if _canonical_json_bytes(value) != payload:
        raise ValueError(f"{context} JSON is not canonical")
    return value


def _decode_json_object(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not valid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {context} JSON: {error.msg}") from error
    return _require_json_object(value, context=context)


def _require_json_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _require_exact_keys(
    value: dict[str, Any], expected: frozenset[str], *, context: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} has unexpected or missing fields")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _emit_rejection(stream: TextIO, *, phase: str, error: Exception) -> None:
    _emit_json(
        stream,
        {
            "status": "rejected",
            "phase": phase,
            "error_type": type(error).__name__,
        },
    )


def _emit_json(stream: TextIO, value: object) -> None:
    stream.write(_canonical_json_bytes(value).decode("utf-8"))
    stream.write("\n")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".write-partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _canonical_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
