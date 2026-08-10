from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
import unittest

from semantic_action_extractor.srl.systems_benchmark import (
    SYSTEMS_BENCHMARK_VERSION,
    BenchmarkIdentity,
    BenchmarkProtocol,
    LatencyPercentiles,
    SystemsBenchmarkRuntimeError,
    SystemsBenchmarkResult,
    nearest_rank_percentile,
    run_systems_benchmark,
)


DATASET_FINGERPRINT = "a" * 64
CONFIG_DIGEST = "b" * 64
CHECKPOINT_DIGEST = "c" * 64


class SyntheticRunner:
    """Timer-only fake whose private marker must never reach output."""

    def __init__(
        self,
        *,
        single=(0.010, 0.020, 0.030, 0.040),
        batched=(0.040, 0.050, 0.060, 0.070),
        peak_memory=12_345_678,
    ):
        self._durations = {1: iter(single), 8: iter(batched)}
        self._peak_memory = peak_memory
        self.calls = []
        self.private_corpus_marker = "SECRET CORPUS TEXT AND EXAMPLE ID"

    def warmup(self, batch_size):
        self.calls.append(("warmup", batch_size))

    def reset_peak_memory(self):
        self.calls.append(("reset_peak_memory",))

    def measure_seconds(self, batch_size):
        self.calls.append(("measure_seconds", batch_size))
        return next(self._durations[batch_size])

    def peak_memory_bytes(self):
        self.calls.append(("peak_memory_bytes",))
        return self._peak_memory


def _identity(**overrides):
    values = {
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "config_digest": CONFIG_DIGEST,
        "checkpoint_digest": CHECKPOINT_DIGEST,
    }
    values.update(overrides)
    return BenchmarkIdentity(**values)


def _protocol(**overrides):
    values = {
        "single_example_batch_size": 1,
        "batched_batch_size": 8,
        "warmup_iterations": 2,
        "measured_iterations": 4,
    }
    values.update(overrides)
    return BenchmarkProtocol(**values)


def _run(runner=None, **overrides):
    values = {
        "identity": _identity(),
        "variant": "predicate_signal",
        "device": "mps",
        "protocol": _protocol(),
        "checkpoint_size_bytes": 438_765_432,
        "hardware": {
            "operating_system": "macOS-15.6",
            "machine": "arm64",
            "accelerator": "Apple-M4",
        },
        "package_versions": {
            "transformers": "4.53.0",
            "python": "3.12.11",
            "torch": "2.7.1",
        },
        "runner": runner if runner is not None else SyntheticRunner(),
    }
    values.update(overrides)
    return run_systems_benchmark(**values)


class PercentileTests(unittest.TestCase):
    def test_uses_documented_nearest_rank_without_interpolation(self):
        values = (0.040, 0.010, 0.030, 0.020)

        self.assertEqual(nearest_rank_percentile(values, 50), 0.020)
        self.assertEqual(nearest_rank_percentile(values, 95), 0.040)
        self.assertEqual(nearest_rank_percentile((0.125,), 50), 0.125)

        summary = LatencyPercentiles.from_seconds(values)
        self.assertEqual(summary.p50_latency_ms, 20.0)
        self.assertEqual(summary.p95_latency_ms, 40.0)

    def test_rejects_invalid_percentile_inputs(self):
        for values, percentile, message in (
            ((), 50, "cannot be empty"),
            ((0.1,), 0, "between 1 and 100"),
            ((0.1,), 101, "between 1 and 100"),
            ((0.1,), 50.0, "must be an integer"),
            ((float("nan"),), 50, "must be finite"),
            ((float("inf"),), 50, "must be finite"),
            ((-0.1,), 50, "must be non-negative"),
            ((-0.0,), 50, "must be non-negative"),
            ((True,), 50, "must be a number"),
        ):
            with self.subTest(values=values, percentile=percentile):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    nearest_rank_percentile(values, percentile)

        with self.assertRaisesRegex(TypeError, "must be a sequence"):
            nearest_rank_percentile("0.1", 50)


class BenchmarkContractTests(unittest.TestCase):
    def test_runs_fixed_warmups_and_measurements_and_computes_aggregates(self):
        runner = SyntheticRunner()

        result = _run(runner)

        self.assertEqual(
            runner.calls,
            [
                ("warmup", 1),
                ("warmup", 1),
                ("warmup", 8),
                ("warmup", 8),
                ("reset_peak_memory",),
                ("measure_seconds", 1),
                ("measure_seconds", 1),
                ("measure_seconds", 1),
                ("measure_seconds", 1),
                ("measure_seconds", 8),
                ("measure_seconds", 8),
                ("measure_seconds", 8),
                ("measure_seconds", 8),
                ("peak_memory_bytes",),
            ],
        )
        self.assertEqual(result.single_example.p50_latency_ms, 20.0)
        self.assertEqual(result.single_example.p95_latency_ms, 40.0)
        self.assertEqual(result.batched.p50_latency_ms, 50.0)
        self.assertEqual(result.batched.p95_latency_ms, 70.0)
        self.assertAlmostEqual(
            result.batched_throughput_examples_per_second,
            32 / math.fsum((0.040, 0.050, 0.060, 0.070)),
        )
        self.assertEqual(result.peak_memory_bytes, 12_345_678)
        self.assertEqual(result.checkpoint_size_bytes, 438_765_432)
        self.assertEqual(result.benchmark_version, SYSTEMS_BENCHMARK_VERSION)

    def test_serializes_canonical_aggregate_only_result_and_round_trips(self):
        runner = SyntheticRunner()
        result = _run(runner)

        payload = result.canonical_json_bytes()
        decoded = json.loads(payload)
        restored = SystemsBenchmarkResult.from_canonical_json_bytes(payload)

        self.assertEqual(restored, result)
        self.assertEqual(restored.canonical_json_bytes(), payload)
        self.assertEqual(
            result.digest, hashlib.sha256(payload).hexdigest()
        )
        self.assertNotIn(b" ", payload)
        self.assertNotIn(b"SECRET CORPUS", payload)
        self.assertNotIn(b"example_id", payload)
        self.assertNotIn(b"raw_duration", payload)
        self.assertNotIn(b"path", payload)
        self.assertEqual(
            set(decoded),
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
            },
        )
        self.assertEqual(
            set(decoded["identity"]),
            {
                "dataset_fingerprint",
                "config_digest",
                "checkpoint_digest",
            },
        )
        self.assertEqual(
            decoded["environment"]["hardware"],
            {
                "accelerator": "Apple-M4",
                "machine": "arm64",
                "operating_system": "macOS-15.6",
            },
        )
        self.assertNotIn("samples", decoded["single_example"])
        self.assertNotIn("samples", decoded["batched"])

    def test_environment_mappings_are_canonical_and_immutable(self):
        result = _run()

        self.assertEqual(
            tuple(key for key, _ in result.hardware),
            ("accelerator", "machine", "operating_system"),
        )
        self.assertEqual(
            tuple(key for key, _ in result.package_versions),
            ("python", "torch", "transformers"),
        )
        with self.assertRaises(FrozenInstanceError):
            result.device = "cpu"

    def test_validates_identity_protocol_variant_and_device(self):
        for field in (
            "dataset_fingerprint",
            "config_digest",
            "checkpoint_digest",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    _identity(**{field: "not-a-digest"})

        protocol_cases = (
            ({"single_example_batch_size": 2}, "must equal one"),
            ({"batched_batch_size": 1}, "at least two"),
            ({"warmup_iterations": -1}, "non-negative"),
            ({"measured_iterations": 0}, "greater than zero"),
            ({"measured_iterations": True}, "must be an integer"),
        )
        for overrides, message in protocol_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    _protocol(**overrides)

        runner = SyntheticRunner()
        with self.assertRaisesRegex(ValueError, "variant must"):
            _run(runner, variant="arbitrary")
        self.assertEqual(runner.calls, [])
        with self.assertRaisesRegex(ValueError, "device must"):
            _run(SyntheticRunner(), device="gpu")

    def test_rejects_free_text_and_unapproved_environment_fields_before_run(self):
        cases = (
            (
                {"hardware": {"machine": "secret corpus sentence"}},
                "not free text",
            ),
            (
                {"hardware": {"hostname": "private-host"}},
                "unsupported key",
            ),
            (
                {"package_versions": {"torch": "2.7.1\nsecret"}},
                "not free text",
            ),
            (
                {"package_versions": {"private_package": "1.0.0"}},
                "unsupported key",
            ),
            ({"hardware": {}}, "cannot be empty"),
            ({"package_versions": {}}, "cannot be empty"),
        )
        for overrides, message in cases:
            runner = SyntheticRunner()
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    _run(runner, **overrides)
                self.assertEqual(runner.calls, [])

    def test_rejects_nonfinite_negative_and_invalid_timing_results(self):
        invalid = (
            (float("nan"), "must be finite"),
            (float("inf"), "must be finite"),
            (-0.01, "must be non-negative"),
            (-0.0, "must be non-negative"),
            (True, "must be a number"),
        )
        for duration, message in invalid:
            runner = SyntheticRunner(single=(duration,) * 4)
            with self.subTest(duration=duration):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    _run(runner)

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            _run(SyntheticRunner(batched=(0.0,) * 4))

    def test_rejects_negative_resource_metrics_and_bad_runner(self):
        runner = SyntheticRunner()
        with self.assertRaisesRegex(ValueError, "checkpoint_size_bytes"):
            _run(runner, checkpoint_size_bytes=-1)
        self.assertEqual(runner.calls, [])

        with self.assertRaisesRegex(ValueError, "peak_memory_bytes"):
            _run(SyntheticRunner(peak_memory=-1))

        with self.assertRaisesRegex(TypeError, "callable warmup"):
            _run(object())

    def test_sanitizes_injected_runner_exception_messages(self):
        class FailingRunner(SyntheticRunner):
            def warmup(self, batch_size):
                del batch_size
                raise RuntimeError("PRIVATE-RUNNER-PAYLOAD")

        with self.assertRaises(SystemsBenchmarkRuntimeError) as raised:
            _run(FailingRunner())

        self.assertEqual(
            str(raised.exception),
            "benchmark runner failed during warmup",
        )
        self.assertNotIn("PRIVATE-RUNNER-PAYLOAD", str(raised.exception))

    def test_result_rejects_inconsistent_or_nonfinite_aggregates(self):
        result = _run()

        with self.assertRaisesRegex(ValueError, "cannot be below"):
            replace(
                result,
                single_example=LatencyPercentiles(
                    p50_latency_ms=20.0,
                    p95_latency_ms=10.0,
                ),
            )
        for throughput in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(throughput=throughput):
                with self.assertRaisesRegex(ValueError, "throughput"):
                    replace(
                        result,
                        batched_throughput_examples_per_second=throughput,
                    )
        with self.assertRaisesRegex(ValueError, "peak_memory_bytes"):
            replace(result, peak_memory_bytes=-1)
        with self.assertRaisesRegex(ValueError, "checkpoint_size_bytes"):
            replace(result, checkpoint_size_bytes=-1)

    def test_canonical_loader_rejects_unknown_duplicate_and_noncanonical_json(self):
        payload = _run().canonical_json_bytes()
        decoded = json.loads(payload)

        decoded["notes"] = "not allowed"
        unknown = json.dumps(
            decoded, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        with self.assertRaisesRegex(ValueError, "unknown keys: notes"):
            SystemsBenchmarkResult.from_canonical_json_bytes(unknown)

        pretty = json.dumps(json.loads(payload), indent=2).encode("ascii")
        with self.assertRaisesRegex(ValueError, "not in canonical form"):
            SystemsBenchmarkResult.from_canonical_json_bytes(pretty)

        duplicate = payload.replace(
            b'"variant":"predicate_signal"',
            b'"variant":"predicate_signal","variant":"predicate_signal"',
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            SystemsBenchmarkResult.from_canonical_json_bytes(duplicate)

        with self.assertRaisesRegex(TypeError, "must be bytes"):
            SystemsBenchmarkResult.from_canonical_json_bytes(payload.decode())


if __name__ == "__main__":
    unittest.main()
