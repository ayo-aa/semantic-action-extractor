"""Command line for a complete paired supplied-predicate SRL experiment."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import TextIO

from .dataset_io import read_prepared_dataset
from .experiment_config import TrainingConfig, load_training_config
from .training_engine import (
    RunContext,
    SRLRunResult,
    run_paired_srl_experiments,
    run_srl_experiment,
    validate_paired_configs,
)


CONFIG_IDENTITY_VERSION = 1
RESULT_IDENTITY_VERSION = 1
PARTIAL_STATUS_VERSION = 1
FAILURE_MARKER_VERSION = 1

PAIRED_RESULTS_FILENAME = "paired_results.json"
CONFIG_IDENTITIES_FILENAME = "config_identities.json"
RESULT_IDENTITY_FILENAME = "paired_result_identity.json"
PARTIAL_STATUS_FILENAME = "partial_status.json"
FAILURE_MARKER_FILENAME = "failure.json"
CONFIG_DIRECTORY = "configs"
CHECKPOINT_DIRECTORY = "checkpoints"
PREDICATE_CONFIG_FILENAME = "predicate_signal.json"
ABLATION_CONFIG_FILENAME = "no_predicate_signal.json"

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
        _require_new_directory_target(output_root, field="output root")
        _require_new_directory_target(partial_root, field="partial output root")
        path_policy(output_root)
        path_policy(partial_root)
    except Exception as error:
        _emit_rejection(error_stream, phase=phase, error=error)
        return 2

    phase = "dataset_preflight"
    try:
        prepared = read_prepared_dataset(args.dataset)
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

    phase = "output_staging"
    completed_runs = 0
    active_variant: str | None = None
    active_seed: int | None = None
    try:
        partial_root.mkdir(parents=True, exist_ok=False)
        config_paths = _write_config_identities(
            partial_root,
            predicate_config=predicate_config,
            ablation_config=ablation_config,
        )
        _atomic_write_json(
            partial_root / PARTIAL_STATUS_FILENAME,
            {
                "partial_status_version": PARTIAL_STATUS_VERSION,
                "status": "partial",
                "completed_runs": 0,
                "expected_runs": 6,
                "predicate_config_digest": predicate_config.digest,
                "no_predicate_signal_config_digest": ablation_config.digest,
            },
        )

        phase = "paired_training"

        def run_one(config: TrainingConfig, seed: int) -> SRLRunResult:
            nonlocal active_seed, active_variant, completed_runs
            active_variant = config.variant
            active_seed = seed
            checkpoint = (
                partial_root
                / CHECKPOINT_DIRECTORY
                / config.variant
                / f"seed-{seed:010d}"
            )
            started_at = clock()
            context = RunContext(
                git_revision=args.git_revision,
                started_at=started_at,
            )
            result = runner(
                args.dataset,
                config,
                checkpoint,
                seed=seed,
                context=context,
                utc_now=clock,
            )
            completed_runs += 1
            _atomic_write_json(
                partial_root / PARTIAL_STATUS_FILENAME,
                {
                    "partial_status_version": PARTIAL_STATUS_VERSION,
                    "status": "partial",
                    "completed_runs": completed_runs,
                    "expected_runs": 6,
                    "predicate_config_digest": predicate_config.digest,
                    "no_predicate_signal_config_digest": (
                        ablation_config.digest
                    ),
                },
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
        _atomic_write_bytes(result_path, result_bytes)
        result_digest = hashlib.sha256(result_bytes).hexdigest()
        _atomic_write_json(
            partial_root / RESULT_IDENTITY_FILENAME,
            {
                "result_identity_version": RESULT_IDENTITY_VERSION,
                "path": PAIRED_RESULTS_FILENAME,
                "digest": result_digest,
            },
        )
        (partial_root / PARTIAL_STATUS_FILENAME).unlink()
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
            "git_revision": args.git_revision,
            "repository": str(repository),
        },
    )
    return 0


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
    relative_paths = {
        "predicate_signal": Path(CONFIG_DIRECTORY) / PREDICATE_CONFIG_FILENAME,
        "no_predicate_signal": Path(CONFIG_DIRECTORY) / ABLATION_CONFIG_FILENAME,
    }
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


def _require_new_directory_target(path: Path, *, field: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{field} already exists")


def _git_revision_argument(value: str) -> str:
    if _GIT_REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "must be a 40-character lowercase hexadecimal commit"
        )
    return value


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
    stream.write(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    stream.write("\n")


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
    )


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
