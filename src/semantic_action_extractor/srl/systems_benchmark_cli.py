"""Run the fixed systems benchmark against one validated SRL checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol, TextIO

from .batching import PaddedSRLBatch, collate_srl_batch, prepare_srl_split
from .checkpoint_bundle import (
    LABEL_CONFIG_FILENAME,
    METADATA_FILENAME,
    STATE_DICT_FILENAME,
    CheckpointLabelConfig,
    CheckpointMetadata,
    LoadedCheckpointBundle,
    ValidatedCheckpointBundle,
    load_checkpoint_bundle,
    validate_checkpoint_bundle,
)
from .dataset_io import PreparedSRLDataset, read_prepared_dataset
from .experiment_config import TrainingConfig, load_training_config
from .label_vocabulary import (
    SRLLabelVocabulary,
    build_training_label_vocabulary,
)
from .systems_benchmark import (
    BenchmarkIdentity,
    BenchmarkProtocol,
    SystemsBenchmarkResult,
    SystemsBenchmarkRunner,
    run_systems_benchmark,
)
from .training_engine import TorchTrainingRuntime


FIXED_BENCHMARK_PROTOCOL = BenchmarkProtocol(
    single_example_batch_size=1,
    batched_batch_size=8,
    warmup_iterations=10,
    measured_iterations=100,
)

CUDA_MEMORY_MEASUREMENT_METHOD = "cuda_max_memory_allocated"
INJECTED_MEMORY_MEASUREMENT_METHOD = "injected_runner_peak"
PROCESS_MEMORY_MEASUREMENT_METHOD = "process_peak_rss"
_BUNDLE_FILENAMES = (
    METADATA_FILENAME,
    LABEL_CONFIG_FILENAME,
    STATE_DICT_FILENAME,
)
_SAFE_ATOM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:/()-]{0,95}")


class UnsupportedPeakMemoryError(RuntimeError):
    """Raised when the selected device lacks resettable peak counters."""


class CheckpointBenchmarkRuntime(Protocol):
    """Concrete-model boundary used after all dependency-free validation."""

    def resolve_device(self, request: str) -> str: ...

    def memory_measurement_method(self, *, device: str) -> str: ...

    def load_tokenizer(self, config: TrainingConfig) -> Any: ...

    def build_model(
        self,
        config: TrainingConfig,
        num_labels: int,
        device: str,
    ) -> Any: ...

    def load_state_dict(self, path: Path, *, device: str) -> Any: ...

    def restore_state_dict(self, model: Any, state_dict: Any) -> None: ...

    def create_runner(
        self,
        model: Any,
        batches: FixedBenchmarkBatches,
        *,
        device: str,
        memory_measurement_method: str,
    ) -> SystemsBenchmarkRunner: ...

    def package_versions(self) -> Mapping[str, str]: ...

    def hardware(self, *, device: str) -> Mapping[str, str]: ...


RuntimeFactory = Callable[[], CheckpointBenchmarkRuntime]
OutputPathPolicy = Callable[[Path], None]


@dataclass(frozen=True, slots=True)
class FixedBenchmarkBatches:
    """Two internal batches selected without exposing their source records."""

    single_example: PaddedSRLBatch
    batched: PaddedSRLBatch

    def __post_init__(self) -> None:
        if not isinstance(self.single_example, PaddedSRLBatch):
            raise TypeError("single_example must be a PaddedSRLBatch")
        if not isinstance(self.batched, PaddedSRLBatch):
            raise TypeError("batched must be a PaddedSRLBatch")
        if len(self.single_example.example_ids) != 1:
            raise ValueError("single benchmark batch must contain one example")
        if len(self.batched.example_ids) < 2:
            raise ValueError("batched benchmark batch must contain two examples")

    def for_size(self, batch_size: int) -> PaddedSRLBatch:
        if type(batch_size) is not int:
            raise TypeError("benchmark batch size must be an integer")
        if batch_size == len(self.single_example.example_ids):
            return self.single_example
        if batch_size == len(self.batched.example_ids):
            return self.batched
        raise ValueError("benchmark runner received an unsupported batch size")


class TorchSystemsBenchmarkRunner:
    """Synchronized inference with explicit device memory semantics."""

    def __init__(
        self,
        *,
        torch_module: Any,
        model: Any,
        batches: FixedBenchmarkBatches,
        device: str,
        memory_measurement_method: str,
        timer: Callable[[], float] = time.perf_counter,
        resource_module: Any | None = None,
        platform_system: Callable[[], str] = platform.system,
    ) -> None:
        if not callable(timer):
            raise TypeError("benchmark timer must be callable")
        if not callable(platform_system):
            raise TypeError("platform_system must be callable")
        if memory_measurement_method not in {
            CUDA_MEMORY_MEASUREMENT_METHOD,
            PROCESS_MEMORY_MEASUREMENT_METHOD,
        }:
            raise ValueError("unsupported real memory measurement method")
        self._torch = torch_module
        self._model = model
        self._device_name = device
        self._device = torch_module.device(device)
        self._memory_measurement_method = memory_measurement_method
        self._timer = timer
        self._resource_module = resource_module
        self._platform_system = platform_system
        self._inputs = {
            1: self._tensorize(batches.single_example),
            len(batches.batched.example_ids): self._tensorize(batches.batched),
        }
        eval_method = getattr(model, "eval", None)
        if not callable(eval_method):
            raise TypeError("benchmark model must provide callable eval")
        eval_method()
        self._require_device_api()

    def warmup(self, batch_size: int) -> None:
        self._synchronize()
        self._infer(batch_size)
        self._synchronize()

    def reset_peak_memory(self) -> None:
        self._synchronize()
        if (
            self._memory_measurement_method
            == CUDA_MEMORY_MEASUREMENT_METHOD
        ):
            self._torch.cuda.reset_peak_memory_stats(self._device)
        # Process peak RSS is a lifetime high-water mark and cannot be reset.

    def measure_seconds(self, batch_size: int) -> float:
        self._synchronize()
        started = self._timer()
        self._infer(batch_size)
        self._synchronize()
        finished = self._timer()
        return finished - started

    def peak_memory_bytes(self) -> int:
        self._synchronize()
        if (
            self._memory_measurement_method
            == CUDA_MEMORY_MEASUREMENT_METHOD
        ):
            return int(self._torch.cuda.max_memory_allocated(self._device))
        resource_module = self._get_resource_module()
        return _process_peak_rss_bytes(
            resource_module,
            operating_system=self._platform_system(),
        )

    def _infer(self, batch_size: int) -> None:
        try:
            inputs = self._inputs[batch_size]
        except KeyError as error:
            raise ValueError(
                "benchmark runner received an unsupported batch size"
            ) from error
        inference_mode = getattr(self._torch, "inference_mode", None)
        context_factory = (
            inference_mode
            if callable(inference_mode)
            else getattr(self._torch, "no_grad", None)
        )
        if not callable(context_factory):
            raise TypeError("torch must provide inference_mode or no_grad")
        with context_factory():
            self._model(**inputs)

    def _tensorize(self, batch: PaddedSRLBatch) -> dict[str, Any]:
        values = batch.to_model_inputs()
        return {
            name: self._torch.tensor(
                values[name],
                dtype=self._torch.long,
                device=self._device,
            )
            for name in ("input_ids", "attention_mask", "token_type_ids")
        }

    def _synchronize(self) -> None:
        device_kind = self._device_name.partition(":")[0]
        if device_kind == "cuda":
            self._torch.cuda.synchronize(self._device)
        elif device_kind == "mps":
            self._torch.mps.synchronize()

    def _require_device_api(self) -> None:
        device_kind = self._device_name.partition(":")[0]
        if device_kind not in {"cpu", "mps", "cuda"}:
            raise ValueError("unsupported benchmark device")
        if device_kind == "cuda":
            for name in (
                "synchronize",
                "reset_peak_memory_stats",
                "max_memory_allocated",
            ):
                if callable(getattr(self._torch.cuda, name, None)):
                    continue
                raise UnsupportedPeakMemoryError(
                    "CUDA runtime lacks resettable peak-memory counters"
                )
        if device_kind == "mps" and not callable(
            getattr(self._torch.mps, "synchronize", None)
        ):
            raise UnsupportedPeakMemoryError(
                "MPS runtime lacks device synchronization"
            )
        if (
            self._memory_measurement_method
            == CUDA_MEMORY_MEASUREMENT_METHOD
            and device_kind != "cuda"
        ):
            raise ValueError("CUDA memory method requires a CUDA device")
        if (
            self._memory_measurement_method
            == PROCESS_MEMORY_MEASUREMENT_METHOD
        ):
            _require_process_peak_rss_support(
                self._get_resource_module(),
                operating_system=self._platform_system(),
            )

    def _get_resource_module(self) -> Any:
        if self._resource_module is not None:
            return self._resource_module
        try:
            return importlib.import_module("resource")
        except ImportError as error:
            raise UnsupportedPeakMemoryError(
                "process peak RSS is unavailable on this platform"
            ) from error


class TorchCheckpointBenchmarkRuntime:
    """Pinned tokenizer/model loading plus a synchronized Torch runner."""

    def __init__(
        self,
        *,
        torch_module: Any | None = None,
        transformers_module: Any | None = None,
        timer: Callable[[], float] = time.perf_counter,
        resource_module: Any | None = None,
        platform_system: Callable[[], str] = platform.system,
    ) -> None:
        self._training = TorchTrainingRuntime(
            torch_module=torch_module,
            transformers_module=transformers_module,
        )
        self._timer = timer
        self._resource_module = resource_module
        self._platform_system = platform_system

    @property
    def torch(self) -> Any:
        return self._training.torch

    def resolve_device(self, request: str) -> str:
        return self._training.resolve_device(request)

    def memory_measurement_method(self, *, device: str) -> str:
        device_kind = device.partition(":")[0]
        if device_kind == "cpu":
            _require_process_peak_rss_support(
                self._get_resource_module(),
                operating_system=self._platform_system(),
            )
            return PROCESS_MEMORY_MEASUREMENT_METHOD
        if device_kind == "mps":
            if not callable(getattr(self.torch.mps, "synchronize", None)):
                raise UnsupportedPeakMemoryError(
                    "MPS runtime lacks device synchronization"
                )
            _require_process_peak_rss_support(
                self._get_resource_module(),
                operating_system=self._platform_system(),
            )
            return PROCESS_MEMORY_MEASUREMENT_METHOD
        if device_kind != "cuda":
            raise ValueError("unsupported benchmark device")
        for name in (
            "synchronize",
            "reset_peak_memory_stats",
            "max_memory_allocated",
        ):
            if not callable(getattr(self.torch.cuda, name, None)):
                raise UnsupportedPeakMemoryError(
                    "CUDA runtime lacks resettable peak-memory counters"
                )
        return CUDA_MEMORY_MEASUREMENT_METHOD

    def load_tokenizer(self, config: TrainingConfig) -> Any:
        return self._training.load_tokenizer(config)

    def build_model(
        self,
        config: TrainingConfig,
        num_labels: int,
        device: str,
    ) -> Any:
        return self._training.build_model(config, num_labels, device)

    def load_state_dict(self, path: Path, *, device: str) -> Any:
        return self._training.load_state_dict(path, device=device)

    def restore_state_dict(self, model: Any, state_dict: Any) -> None:
        self._training.restore_state_dict(model, state_dict)

    def create_runner(
        self,
        model: Any,
        batches: FixedBenchmarkBatches,
        *,
        device: str,
        memory_measurement_method: str,
    ) -> SystemsBenchmarkRunner:
        return TorchSystemsBenchmarkRunner(
            torch_module=self.torch,
            model=model,
            batches=batches,
            device=device,
            memory_measurement_method=memory_measurement_method,
            timer=self._timer,
            resource_module=self._resource_module,
            platform_system=self._platform_system,
        )

    def package_versions(self) -> Mapping[str, str]:
        return self._training.package_versions()

    def hardware(self, *, device: str) -> Mapping[str, str]:
        if device.startswith("cuda"):
            accelerator = self.torch.cuda.get_device_name(
                self.torch.device(device)
            )
        elif device == "mps":
            accelerator = "Apple-MPS"
        else:
            accelerator = platform.processor().strip() or "generic-cpu"
        operating_system = f"{platform.system()}-{platform.release()}"
        return {
            "accelerator": _bounded_hardware_atom(accelerator),
            "machine": _bounded_hardware_atom(
                platform.machine() or "unknown-machine"
            ),
            "operating_system": _bounded_hardware_atom(operating_system),
        }

    def _get_resource_module(self) -> Any:
        if self._resource_module is not None:
            return self._resource_module
        try:
            return importlib.import_module("resource")
        except ImportError as error:
            raise UnsupportedPeakMemoryError(
                "process peak RSS is unavailable on this platform"
            ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-action-benchmark-srl",
        description=(
            "Benchmark one validated supplied-predicate SRL checkpoint using "
            "a fixed aggregate-only protocol."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Strict TOML configuration used to train the checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Existing prepared SRL dataset directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Existing checkpoint bundle directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New Git-ignored JSON file for the canonical aggregate result.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: RuntimeFactory | None = None,
    protocol: BenchmarkProtocol = FIXED_BENCHMARK_PROTOCOL,
    output_path_policy: OutputPathPolicy | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Validate all local inputs before loading optional ML dependencies."""

    args = build_parser().parse_args(argv)
    output_stream = stdout if stdout is not None else sys.stdout
    error_stream = stderr if stderr is not None else sys.stderr
    factory = (
        runtime_factory
        if runtime_factory is not None
        else TorchCheckpointBenchmarkRuntime
    )
    path_policy = (
        output_path_policy
        if output_path_policy is not None
        else require_git_ignored_output_path
    )

    phase = "config_validation"
    try:
        if not isinstance(protocol, BenchmarkProtocol):
            raise TypeError("protocol must be a BenchmarkProtocol")
        config = load_training_config(args.config)
    except Exception as error:
        _emit_failure(error_stream, status="rejected", phase=phase, error=error)
        return 2

    phase = "dataset_validation"
    try:
        dataset = read_prepared_dataset(args.dataset)
        if (
            dataset.manifest.dataset_fingerprint
            != config.prepared_data_fingerprint
        ):
            raise ValueError(
                "prepared dataset fingerprint does not match the config"
            )
        vocabulary = _training_vocabulary(dataset)
    except Exception as error:
        _emit_failure(error_stream, status="rejected", phase=phase, error=error)
        return 2

    phase = "output_validation"
    try:
        output = _resolve_new_output(args.output)
        path_policy(output)
    except Exception as error:
        _emit_failure(error_stream, status="rejected", phase=phase, error=error)
        return 2

    phase = "checkpoint_validation"
    try:
        validated = inspect_checkpoint_bundle(
            args.checkpoint,
            config=config,
            vocabulary=vocabulary,
        )
        checkpoint_size = checkpoint_bundle_size_bytes(validated)
    except Exception as error:
        _emit_failure(error_stream, status="rejected", phase=phase, error=error)
        return 2

    phase = "runtime_initialization"
    try:
        if not callable(factory):
            raise TypeError("runtime_factory must be callable")
        runtime = factory()
        _require_runtime(runtime)
        device = runtime.resolve_device(config.device_request)
        memory_method = runtime.memory_measurement_method(device=device)
        tokenizer = runtime.load_tokenizer(config)
        batches = build_fixed_benchmark_batches(
            dataset,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            config=config,
            protocol=protocol,
        )
        model = runtime.build_model(config, len(vocabulary), device)
        loaded = load_validated_checkpoint_state(
            validated,
            config=config,
            runtime=runtime,
            device=device,
        )
        runtime.restore_state_dict(model, loaded.state_dict)
        runner = runtime.create_runner(
            model,
            batches,
            device=device,
            memory_measurement_method=memory_method,
        )
        result = run_systems_benchmark(
            identity=BenchmarkIdentity(
                dataset_fingerprint=dataset.manifest.dataset_fingerprint,
                config_digest=config.digest,
                checkpoint_digest=validated.state_dict_sha256,
            ),
            variant=config.variant,
            device=device,
            protocol=protocol,
            checkpoint_size_bytes=checkpoint_size,
            hardware=runtime.hardware(device=device),
            package_versions=runtime.package_versions(),
            runner=runner,
            memory_measurement_method=memory_method,
        )
    except Exception as error:
        _emit_failure(error_stream, status="failed", phase=phase, error=error)
        return 1

    phase = "result_publication"
    try:
        _atomic_publish_new(output, result.canonical_json_bytes())
    except Exception as error:
        _emit_failure(error_stream, status="failed", phase=phase, error=error)
        return 1

    _emit_json(
        output_stream,
        {
            "status": "complete",
            "benchmark_digest": result.digest,
            "checkpoint_digest": validated.state_dict_sha256,
            "config_digest": config.digest,
            "dataset_fingerprint": dataset.manifest.dataset_fingerprint,
            "variant": config.variant,
            "device": device,
            "memory_measurement_method": memory_method,
        },
    )
    return 0


def inspect_checkpoint_bundle(
    directory: str | Path,
    *,
    config: TrainingConfig,
    vocabulary: SRLLabelVocabulary,
) -> ValidatedCheckpointBundle:
    """Parse canonical metadata, then verify layout, labels, and state hash."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if not isinstance(vocabulary, SRLLabelVocabulary):
        raise TypeError("vocabulary must be an SRLLabelVocabulary")
    source = Path(directory)
    parsed_metadata = CheckpointMetadata.from_canonical_json_bytes(
        (source / METADATA_FILENAME).read_bytes()
    )
    expected_labels = CheckpointLabelConfig.from_labels(vocabulary.labels)
    return validate_checkpoint_bundle(
        source,
        expected_metadata=parsed_metadata.run_metadata,
        expected_config=config,
        expected_label_config=expected_labels,
    )


def load_validated_checkpoint_state(
    validated: ValidatedCheckpointBundle,
    *,
    config: TrainingConfig,
    runtime: CheckpointBenchmarkRuntime,
    device: str,
) -> LoadedCheckpointBundle:
    """Revalidate immediately before the callback is allowed to load state."""

    if not isinstance(validated, ValidatedCheckpointBundle):
        raise TypeError("validated must be a ValidatedCheckpointBundle")
    return load_checkpoint_bundle(
        validated.directory,
        expected_metadata=validated.metadata,
        expected_config=config,
        expected_label_config=validated.label_config,
        load_state_dict=lambda path: runtime.load_state_dict(
            path, device=device
        ),
    )


def checkpoint_bundle_size_bytes(
    validated: ValidatedCheckpointBundle,
) -> int:
    """Return the byte sum of the three integrity-validated bundle members."""

    if not isinstance(validated, ValidatedCheckpointBundle):
        raise TypeError("validated must be a ValidatedCheckpointBundle")
    return sum(
        (validated.directory / filename).stat().st_size
        for filename in _BUNDLE_FILENAMES
    )


def build_fixed_benchmark_batches(
    dataset: PreparedSRLDataset,
    *,
    vocabulary: SRLLabelVocabulary,
    tokenizer: Any,
    config: TrainingConfig,
    protocol: BenchmarkProtocol,
) -> FixedBenchmarkBatches:
    """Select the first fixed-size retained test prefix in canonical order."""

    if not isinstance(dataset, PreparedSRLDataset):
        raise TypeError("dataset must be a PreparedSRLDataset")
    if not isinstance(vocabulary, SRLLabelVocabulary):
        raise TypeError("vocabulary must be an SRLLabelVocabulary")
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if not isinstance(protocol, BenchmarkProtocol):
        raise TypeError("protocol must be a BenchmarkProtocol")
    prepared = prepare_srl_split(
        tokenizer,
        dataset.examples,
        vocabulary,
        "test",
        max_length=config.max_length,
    )
    required = protocol.batched_batch_size
    if prepared.retained_count < required:
        raise ValueError(
            "retained test split is smaller than the fixed benchmark batch"
        )
    selected = prepared.examples[:required]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("fast tokenizer must define a non-negative pad token ID")
    return FixedBenchmarkBatches(
        single_example=collate_srl_batch(
            selected[:1],
            pad_token_id=pad_token_id,
            predicate_signal=config.predicate_signal,
        ),
        batched=collate_srl_batch(
            selected,
            pad_token_id=pad_token_id,
            predicate_signal=config.predicate_signal,
        ),
    )


def require_git_ignored_output_path(path: Path) -> None:
    """Reject a result path unless Git confirms that it is ignored."""

    if not isinstance(path, Path):
        raise TypeError("ignored output path must be a Path")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(path.parent),
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


def _training_vocabulary(dataset: PreparedSRLDataset) -> SRLLabelVocabulary:
    training = tuple(
        example for example in dataset.examples if example.split == "train"
    )
    return build_training_label_vocabulary(training)


def _require_runtime(runtime: object) -> None:
    for method_name in (
        "resolve_device",
        "memory_measurement_method",
        "load_tokenizer",
        "build_model",
        "load_state_dict",
        "restore_state_dict",
        "create_runner",
        "package_versions",
        "hardware",
    ):
        if not callable(getattr(runtime, method_name, None)):
            raise TypeError(f"runtime must provide callable {method_name}")


def _resolve_new_output(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("output must be a Path")
    if path.is_symlink() or path.exists():
        raise FileExistsError("output already exists")
    if path.parent.is_symlink():
        raise ValueError("output parent cannot be a symbolic link")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("output parent must be a real existing directory")
    output = parent / path.name
    if not output.name or output.name in {".", ".."}:
        raise ValueError("output must have a file name")
    return output


def _atomic_publish_new(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("result payload must be bytes")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".write-partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if published and path.exists():
            path.unlink()
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.unlink()


def _bounded_hardware_atom(value: object) -> str:
    text = str(value).strip()
    normalized = re.sub(r"[^A-Za-z0-9._+:/()-]+", "-", text)
    normalized = normalized.strip("-")[:96]
    if _SAFE_ATOM_RE.fullmatch(normalized) is None:
        return "unknown-hardware"
    return normalized


def _require_process_peak_rss_support(
    resource_module: Any,
    *,
    operating_system: str,
) -> None:
    if operating_system not in {"Darwin", "Linux"}:
        raise UnsupportedPeakMemoryError(
            "process peak RSS units are unsupported on this platform"
        )
    if not callable(getattr(resource_module, "getrusage", None)):
        raise UnsupportedPeakMemoryError(
            "resource module lacks process peak RSS support"
        )
    if getattr(resource_module, "RUSAGE_SELF", None) is None:
        raise UnsupportedPeakMemoryError(
            "resource module lacks RUSAGE_SELF"
        )


def _process_peak_rss_bytes(
    resource_module: Any,
    *,
    operating_system: str,
) -> int:
    _require_process_peak_rss_support(
        resource_module,
        operating_system=operating_system,
    )
    usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
    value = getattr(usage, "ru_maxrss", None)
    if type(value) not in {int, float}:
        raise UnsupportedPeakMemoryError(
            "process peak RSS value is unavailable"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise UnsupportedPeakMemoryError("process peak RSS value is invalid")
    if operating_system == "Linux":
        normalized *= 1024.0
    if normalized > 2**63 - 1:
        raise UnsupportedPeakMemoryError("process peak RSS value is too large")
    return int(normalized)


def _emit_failure(
    stream: TextIO,
    *,
    status: str,
    phase: str,
    error: Exception,
) -> None:
    _emit_json(
        stream,
        {
            "status": status,
            "phase": phase,
            "error_type": type(error).__name__,
        },
    )


def _emit_json(stream: TextIO, value: object) -> None:
    stream.write(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
