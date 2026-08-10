"""Privacy-safe aggregate systems benchmarking for trained SRL checkpoints.

This module deliberately has no Torch, Transformers, dataset, or checkpoint
loader dependency.  A caller supplies a small runner that owns model loading,
batch construction, device synchronization, and timing.  The orchestration
surface passes only batch sizes and emits only canonical aggregate statistics;
it never accepts or serializes example text, example identifiers, paths, or raw
latency samples.

A concrete ML runner can be added once the trained-checkpoint reload boundary
is finalized.  Until then, deterministic injected runners exercise the complete
aggregation and provenance contract without corpus or model access.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Protocol


SYSTEMS_BENCHMARK_VERSION = 1

MEMORY_MEASUREMENT_METHODS = frozenset(
    {
        "cuda_max_memory_allocated",
        "injected_runner_peak",
        "process_peak_rss",
    }
)

_VARIANTS = frozenset({"predicate_signal", "no_predicate_signal"})
_DEVICE_RE = re.compile(r"(?:cpu|mps|cuda(?::[0-9]+)?)")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_HARDWARE_VALUE_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._+:/()-]{0,95}"
)
_SAFE_VERSION_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")

# Arbitrary keys could invite hostnames, paths, user names, corpus identifiers,
# or free-form notes into a public aggregate.  Keep the serialized environment
# schema intentionally finite.
_HARDWARE_KEYS = frozenset(
    {"accelerator", "machine", "operating_system", "processor"}
)
_PACKAGE_KEYS = frozenset(
    {
        "numpy",
        "python",
        "safetensors",
        "semantic_action_extractor",
        "tokenizers",
        "torch",
        "transformers",
    }
)

_RESULT_KEYS = frozenset(
    {
        "benchmark_version",
        "identity",
        "variant",
        "device",
        "protocol",
        "single_example",
        "batched",
        "resources",
        "environment",
    }
)
_IDENTITY_KEYS = frozenset(
    {"dataset_fingerprint", "config_digest", "checkpoint_digest"}
)
_PROTOCOL_KEYS = frozenset(
    {
        "single_example_batch_size",
        "batched_batch_size",
        "warmup_iterations",
        "measured_iterations",
    }
)
_LATENCY_KEYS = frozenset({"p50_latency_ms", "p95_latency_ms"})
_BATCHED_KEYS = frozenset(
    {
        "p50_latency_ms",
        "p95_latency_ms",
        "throughput_examples_per_second",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "peak_memory_bytes",
        "checkpoint_size_bytes",
        "memory_measurement_method",
    }
)
_ENVIRONMENT_KEYS = frozenset({"hardware", "package_versions"})


class SystemsBenchmarkRunner(Protocol):
    """Injected runtime boundary with no examples in its public surface.

    ``warmup`` performs one unmeasured inference for ``batch_size``.
    ``measure_seconds`` performs one synchronized inference and returns elapsed
    wall time in seconds.  ``reset_peak_memory`` is called after all warmups and
    before measured iterations.  ``peak_memory_bytes`` returns the measured
    peak after both single and batched runs.
    """

    def warmup(self, batch_size: int) -> None: ...

    def reset_peak_memory(self) -> None: ...

    def measure_seconds(self, batch_size: int) -> float: ...

    def peak_memory_bytes(self) -> int: ...


class SystemsBenchmarkRuntimeError(RuntimeError):
    """Raised without propagating caller-owned runner messages."""


def _require_exact_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    integer = _require_exact_int(value, field=field)
    if integer < 0:
        raise ValueError(f"{field} must be non-negative")
    return integer


def _require_metric_float(
    value: object,
    *,
    field: str,
    strictly_positive: bool = False,
) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if result < 0.0 or math.copysign(1.0, result) < 0.0:
        raise ValueError(f"{field} must be non-negative")
    if strictly_positive and result == 0.0:
        raise ValueError(f"{field} must be greater than zero")
    return result


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_memory_measurement_method(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("memory_measurement_method must be a string")
    if value not in MEMORY_MEASUREMENT_METHODS:
        allowed = ", ".join(sorted(MEMORY_MEASUREMENT_METHODS))
        raise ValueError(
            "memory_measurement_method must be one of: " + allowed
        )
    return value


def _require_exact_keys(
    value: object,
    *,
    required: frozenset[str],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    observed = set(value)
    if observed != required:
        missing = sorted(required - observed)
        unknown = sorted(observed - required)
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError(f"invalid {field} keys; " + "; ".join(details))
    return value


def _freeze_safe_mapping(
    value: Mapping[str, str],
    *,
    field: str,
    allowed_keys: frozenset[str],
    value_pattern: re.Pattern[str],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    if not value:
        raise ValueError(f"{field} cannot be empty")
    pairs: list[tuple[str, str]] = []
    for key, item in value.items():
        if not isinstance(key, str) or key not in allowed_keys:
            raise ValueError(f"{field} contains an unsupported key: {key!r}")
        if not isinstance(item, str):
            raise TypeError(f"{field}.{key} must be a string")
        if value_pattern.fullmatch(item) is None:
            raise ValueError(
                f"{field}.{key} must be a bounded metadata atom, not free text"
            )
        pairs.append((key, item))
    return tuple(sorted(pairs))


def _validate_frozen_safe_mapping(
    value: object,
    *,
    field: str,
    allowed_keys: frozenset[str],
    value_pattern: re.Pattern[str],
) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field} must be a nonempty immutable tuple")
    keys: list[str] = []
    for pair in value:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{field} entries must be two-item tuples")
        key, item = pair
        if not isinstance(key, str) or key not in allowed_keys:
            raise ValueError(f"{field} contains an unsupported key: {key!r}")
        if not isinstance(item, str) or value_pattern.fullmatch(item) is None:
            raise ValueError(
                f"{field}.{key} must be a bounded metadata atom, not free text"
            )
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field} keys must be unique")
    if tuple(keys) != tuple(sorted(keys)):
        raise ValueError(f"{field} keys must be in canonical order")


def nearest_rank_percentile(
    values: Sequence[float], percentile: int
) -> float:
    """Return a deterministic nearest-rank percentile.

    Values must be finite and non-negative.  The percentile is an integer from
    1 through 100.  For ``n`` sorted observations, the selected zero-based index
    is ``ceil(percentile * n / 100) - 1``.  No interpolation or platform
    statistics library is involved.
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("percentile values must be a sequence")
    if not values:
        raise ValueError("percentile values cannot be empty")
    requested = _require_exact_int(percentile, field="percentile")
    if not 1 <= requested <= 100:
        raise ValueError("percentile must be between 1 and 100")
    normalized = sorted(
        _require_metric_float(value, field="percentile value")
        for value in values
    )
    rank = math.ceil(requested * len(normalized) / 100)
    return normalized[rank - 1]


@dataclass(frozen=True, slots=True)
class BenchmarkIdentity:
    """Opaque content digests identifying benchmark inputs."""

    dataset_fingerprint: str
    config_digest: str
    checkpoint_digest: str

    def __post_init__(self) -> None:
        _require_digest(
            self.dataset_fingerprint, field="dataset_fingerprint"
        )
        _require_digest(self.config_digest, field="config_digest")
        _require_digest(self.checkpoint_digest, field="checkpoint_digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_digest": self.config_digest,
            "checkpoint_digest": self.checkpoint_digest,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    """Frozen iteration counts and batch sizes for one benchmark."""

    single_example_batch_size: int
    batched_batch_size: int
    warmup_iterations: int
    measured_iterations: int

    def __post_init__(self) -> None:
        single = _require_exact_int(
            self.single_example_batch_size,
            field="single_example_batch_size",
        )
        if single != 1:
            raise ValueError("single_example_batch_size must equal one")
        batched = _require_exact_int(
            self.batched_batch_size, field="batched_batch_size"
        )
        if batched < 2:
            raise ValueError("batched_batch_size must be at least two")
        _require_nonnegative_int(
            self.warmup_iterations, field="warmup_iterations"
        )
        measured = _require_exact_int(
            self.measured_iterations, field="measured_iterations"
        )
        if measured <= 0:
            raise ValueError("measured_iterations must be greater than zero")

    def to_dict(self) -> dict[str, int]:
        return {
            "single_example_batch_size": self.single_example_batch_size,
            "batched_batch_size": self.batched_batch_size,
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
        }


@dataclass(frozen=True, slots=True)
class LatencyPercentiles:
    """Nearest-rank latency aggregates expressed in milliseconds."""

    p50_latency_ms: float
    p95_latency_ms: float

    def __post_init__(self) -> None:
        p50 = _require_metric_float(
            self.p50_latency_ms, field="p50_latency_ms"
        )
        p95 = _require_metric_float(
            self.p95_latency_ms, field="p95_latency_ms"
        )
        if p95 < p50:
            raise ValueError("p95_latency_ms cannot be below p50_latency_ms")

    @classmethod
    def from_seconds(
        cls, duration_seconds: Sequence[float]
    ) -> LatencyPercentiles:
        """Aggregate raw injected durations without retaining the samples."""

        p50_ms = nearest_rank_percentile(duration_seconds, 50) * 1000.0
        p95_ms = nearest_rank_percentile(duration_seconds, 95) * 1000.0
        _require_metric_float(p50_ms, field="p50_latency_ms")
        _require_metric_float(p95_ms, field="p95_latency_ms")
        return cls(p50_latency_ms=p50_ms, p95_latency_ms=p95_ms)

    def to_dict(self) -> dict[str, float]:
        return {
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class SystemsBenchmarkResult:
    """Canonical aggregate-only systems evidence for one checkpoint."""

    benchmark_version: int
    identity: BenchmarkIdentity
    variant: str
    device: str
    protocol: BenchmarkProtocol
    single_example: LatencyPercentiles
    batched: LatencyPercentiles
    batched_throughput_examples_per_second: float
    peak_memory_bytes: int
    checkpoint_size_bytes: int
    hardware: tuple[tuple[str, str], ...]
    package_versions: tuple[tuple[str, str], ...]
    memory_measurement_method: str = "injected_runner_peak"

    def __post_init__(self) -> None:
        version = _require_exact_int(
            self.benchmark_version, field="benchmark_version"
        )
        if version != SYSTEMS_BENCHMARK_VERSION:
            raise ValueError(
                f"benchmark_version must be {SYSTEMS_BENCHMARK_VERSION}"
            )
        if not isinstance(self.identity, BenchmarkIdentity):
            raise TypeError("identity must be a BenchmarkIdentity")
        if self.variant not in _VARIANTS:
            raise ValueError(
                "variant must be predicate_signal or no_predicate_signal"
            )
        if (
            not isinstance(self.device, str)
            or _DEVICE_RE.fullmatch(self.device) is None
        ):
            raise ValueError("device must be cpu, mps, cuda, or cuda:<index>")
        if not isinstance(self.protocol, BenchmarkProtocol):
            raise TypeError("protocol must be a BenchmarkProtocol")
        if not isinstance(self.single_example, LatencyPercentiles):
            raise TypeError("single_example must be LatencyPercentiles")
        if not isinstance(self.batched, LatencyPercentiles):
            raise TypeError("batched must be LatencyPercentiles")
        _require_metric_float(
            self.batched_throughput_examples_per_second,
            field="batched_throughput_examples_per_second",
            strictly_positive=True,
        )
        _require_nonnegative_int(
            self.peak_memory_bytes, field="peak_memory_bytes"
        )
        _require_nonnegative_int(
            self.checkpoint_size_bytes, field="checkpoint_size_bytes"
        )
        _require_memory_measurement_method(self.memory_measurement_method)
        _validate_frozen_safe_mapping(
            self.hardware,
            field="hardware",
            allowed_keys=_HARDWARE_KEYS,
            value_pattern=_SAFE_HARDWARE_VALUE_RE,
        )
        _validate_frozen_safe_mapping(
            self.package_versions,
            field="package_versions",
            allowed_keys=_PACKAGE_KEYS,
            value_pattern=_SAFE_VERSION_VALUE_RE,
        )

    @classmethod
    def create(
        cls,
        *,
        identity: BenchmarkIdentity,
        variant: str,
        device: str,
        protocol: BenchmarkProtocol,
        single_example: LatencyPercentiles,
        batched: LatencyPercentiles,
        batched_throughput_examples_per_second: float,
        peak_memory_bytes: int,
        checkpoint_size_bytes: int,
        hardware: Mapping[str, str],
        package_versions: Mapping[str, str],
        memory_measurement_method: str = "injected_runner_peak",
    ) -> SystemsBenchmarkResult:
        """Freeze safe environment mappings in canonical key order."""

        return cls(
            benchmark_version=SYSTEMS_BENCHMARK_VERSION,
            identity=identity,
            variant=variant,
            device=device,
            protocol=protocol,
            single_example=single_example,
            batched=batched,
            batched_throughput_examples_per_second=(
                batched_throughput_examples_per_second
            ),
            peak_memory_bytes=peak_memory_bytes,
            checkpoint_size_bytes=checkpoint_size_bytes,
            hardware=_freeze_safe_mapping(
                hardware,
                field="hardware",
                allowed_keys=_HARDWARE_KEYS,
                value_pattern=_SAFE_HARDWARE_VALUE_RE,
            ),
            package_versions=_freeze_safe_mapping(
                package_versions,
                field="package_versions",
                allowed_keys=_PACKAGE_KEYS,
                value_pattern=_SAFE_VERSION_VALUE_RE,
            ),
            memory_measurement_method=memory_measurement_method,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_version": self.benchmark_version,
            "identity": self.identity.to_dict(),
            "variant": self.variant,
            "device": self.device,
            "protocol": self.protocol.to_dict(),
            "single_example": self.single_example.to_dict(),
            "batched": {
                **self.batched.to_dict(),
                "throughput_examples_per_second": (
                    self.batched_throughput_examples_per_second
                ),
            },
            "resources": {
                "peak_memory_bytes": self.peak_memory_bytes,
                "checkpoint_size_bytes": self.checkpoint_size_bytes,
                "memory_measurement_method": (
                    self.memory_measurement_method
                ),
            },
            "environment": {
                "hardware": dict(self.hardware),
                "package_versions": dict(self.package_versions),
            },
        }

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    @classmethod
    def from_canonical_json_bytes(
        cls, payload: bytes
    ) -> SystemsBenchmarkResult:
        """Load only exact-schema, duplicate-free canonical JSON."""

        if not isinstance(payload, bytes):
            raise TypeError("benchmark payload must be bytes")
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("benchmark JSON must be ASCII") from error
        try:
            value = json.loads(text, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid benchmark JSON: {error.msg}") from error
        root = _require_exact_keys(
            value, required=_RESULT_KEYS, field="benchmark result"
        )
        identity_value = _require_exact_keys(
            root["identity"], required=_IDENTITY_KEYS, field="identity"
        )
        protocol_value = _require_exact_keys(
            root["protocol"], required=_PROTOCOL_KEYS, field="protocol"
        )
        single_value = _require_exact_keys(
            root["single_example"],
            required=_LATENCY_KEYS,
            field="single_example",
        )
        batched_value = _require_exact_keys(
            root["batched"], required=_BATCHED_KEYS, field="batched"
        )
        resources_value = _require_exact_keys(
            root["resources"], required=_RESOURCE_KEYS, field="resources"
        )
        environment_value = _require_exact_keys(
            root["environment"],
            required=_ENVIRONMENT_KEYS,
            field="environment",
        )
        result = cls.create(
            identity=BenchmarkIdentity(**identity_value),
            variant=root["variant"],
            device=root["device"],
            protocol=BenchmarkProtocol(**protocol_value),
            single_example=LatencyPercentiles(**single_value),
            batched=LatencyPercentiles(
                p50_latency_ms=batched_value["p50_latency_ms"],
                p95_latency_ms=batched_value["p95_latency_ms"],
            ),
            batched_throughput_examples_per_second=batched_value[
                "throughput_examples_per_second"
            ],
            peak_memory_bytes=resources_value["peak_memory_bytes"],
            checkpoint_size_bytes=resources_value["checkpoint_size_bytes"],
            hardware=environment_value["hardware"],
            package_versions=environment_value["package_versions"],
            memory_measurement_method=resources_value[
                "memory_measurement_method"
            ],
        )
        if result.benchmark_version != root["benchmark_version"]:
            raise ValueError("benchmark_version does not match the schema")
        if result.canonical_json_bytes() != payload:
            raise ValueError("benchmark JSON is not in canonical form")
        return result


def run_systems_benchmark(
    *,
    identity: BenchmarkIdentity,
    variant: str,
    device: str,
    protocol: BenchmarkProtocol,
    checkpoint_size_bytes: int,
    hardware: Mapping[str, str],
    package_versions: Mapping[str, str],
    runner: SystemsBenchmarkRunner,
    memory_measurement_method: str = "injected_runner_peak",
) -> SystemsBenchmarkResult:
    """Run warmups and measured iterations through an injected runner.

    The output retains only p50/p95 aggregates, total batched throughput, peak
    memory, checkpoint size, bounded environment atoms, and opaque identities.
    Per-iteration timings are discarded before this function returns.
    """

    # Validate every public metadata field before caller-owned model code runs.
    _require_nonnegative_int(
        checkpoint_size_bytes, field="checkpoint_size_bytes"
    )
    _validate_orchestration_inputs(
        identity=identity,
        variant=variant,
        device=device,
        protocol=protocol,
        hardware=hardware,
        package_versions=package_versions,
        runner=runner,
        memory_measurement_method=memory_measurement_method,
    )

    for batch_size in (
        protocol.single_example_batch_size,
        protocol.batched_batch_size,
    ):
        for _ in range(protocol.warmup_iterations):
            _call_runner(
                runner.warmup,
                phase="warmup",
                batch_size=batch_size,
            )

    _call_runner(runner.reset_peak_memory, phase="peak-memory reset")
    single_seconds = tuple(
        _require_metric_float(
            _call_runner(
                runner.measure_seconds,
                phase="single-example measurement",
                batch_size=protocol.single_example_batch_size,
            ),
            field="single-example elapsed seconds",
        )
        for _ in range(protocol.measured_iterations)
    )
    batched_seconds = tuple(
        _require_metric_float(
            _call_runner(
                runner.measure_seconds,
                phase="batched measurement",
                batch_size=protocol.batched_batch_size,
            ),
            field="batched elapsed seconds",
        )
        for _ in range(protocol.measured_iterations)
    )
    total_batched_seconds = math.fsum(batched_seconds)
    _require_metric_float(
        total_batched_seconds,
        field="total batched elapsed seconds",
        strictly_positive=True,
    )
    throughput = (
        protocol.batched_batch_size
        * protocol.measured_iterations
        / total_batched_seconds
    )
    _require_metric_float(
        throughput,
        field="batched_throughput_examples_per_second",
        strictly_positive=True,
    )
    peak_memory = _require_nonnegative_int(
        _call_runner(runner.peak_memory_bytes, phase="peak-memory read"),
        field="peak_memory_bytes",
    )

    return SystemsBenchmarkResult.create(
        identity=identity,
        variant=variant,
        device=device,
        protocol=protocol,
        single_example=LatencyPercentiles.from_seconds(single_seconds),
        batched=LatencyPercentiles.from_seconds(batched_seconds),
        batched_throughput_examples_per_second=throughput,
        peak_memory_bytes=peak_memory,
        checkpoint_size_bytes=checkpoint_size_bytes,
        hardware=hardware,
        package_versions=package_versions,
        memory_measurement_method=memory_measurement_method,
    )


def _validate_orchestration_inputs(
    *,
    identity: BenchmarkIdentity,
    variant: str,
    device: str,
    protocol: BenchmarkProtocol,
    hardware: Mapping[str, str],
    package_versions: Mapping[str, str],
    runner: SystemsBenchmarkRunner,
    memory_measurement_method: str,
) -> None:
    if not isinstance(identity, BenchmarkIdentity):
        raise TypeError("identity must be a BenchmarkIdentity")
    if not isinstance(protocol, BenchmarkProtocol):
        raise TypeError("protocol must be a BenchmarkProtocol")
    if variant not in _VARIANTS:
        raise ValueError("variant must be predicate_signal or no_predicate_signal")
    if not isinstance(device, str) or _DEVICE_RE.fullmatch(device) is None:
        raise ValueError("device must be cpu, mps, cuda, or cuda:<index>")
    _require_memory_measurement_method(memory_measurement_method)
    _freeze_safe_mapping(
        hardware,
        field="hardware",
        allowed_keys=_HARDWARE_KEYS,
        value_pattern=_SAFE_HARDWARE_VALUE_RE,
    )
    _freeze_safe_mapping(
        package_versions,
        field="package_versions",
        allowed_keys=_PACKAGE_KEYS,
        value_pattern=_SAFE_VERSION_VALUE_RE,
    )
    for method_name in (
        "warmup",
        "reset_peak_memory",
        "measure_seconds",
        "peak_memory_bytes",
    ):
        if not callable(getattr(runner, method_name, None)):
            raise TypeError(f"runner must provide callable {method_name}")


def _call_runner(
    method: Callable[..., object], *, phase: str, **kwargs: int
) -> object:
    """Call injected code without exposing its exception text or payload."""

    try:
        return method(**kwargs)
    except Exception:
        raise SystemsBenchmarkRuntimeError(
            f"benchmark runner failed during {phase}"
        ) from None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
