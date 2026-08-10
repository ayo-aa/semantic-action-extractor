from contextlib import nullcontext
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from semantic_action_extractor.srl.batching import PaddedSRLBatch
from semantic_action_extractor.srl.checkpoint_bundle import (
    LABEL_CONFIG_FILENAME,
    METADATA_FILENAME,
    STATE_DICT_FILENAME,
    CheckpointLabelConfig,
    save_checkpoint_bundle,
)
from semantic_action_extractor.srl.dataset_io import (
    read_prepared_dataset,
    write_prepared_dataset,
)
from semantic_action_extractor.srl.example import WordLevelSRLExample
from semantic_action_extractor.srl.experiment_config import (
    load_training_config,
)
from semantic_action_extractor.srl.label_vocabulary import (
    build_training_label_vocabulary,
)
from semantic_action_extractor.srl.run_metadata import RunMetadata
from semantic_action_extractor.srl.systems_benchmark import (
    BenchmarkProtocol,
    SystemsBenchmarkResult,
)
from semantic_action_extractor.srl.systems_benchmark_cli import (
    CUDA_MEMORY_MEASUREMENT_METHOD,
    INJECTED_MEMORY_MEASUREMENT_METHOD,
    PROCESS_MEMORY_MEASUREMENT_METHOD,
    FixedBenchmarkBatches,
    TorchSystemsBenchmarkRunner,
    UnsupportedPeakMemoryError,
    main,
)


MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40
GIT_REVISION = "c" * 40
INITIAL_STATE = "d" * 64
PRIVATE_MARKER = "SECRET-CORPUS-TEXT-AND-EXAMPLE-ID"

TEST_PROTOCOL = BenchmarkProtocol(
    single_example_batch_size=1,
    batched_batch_size=4,
    warmup_iterations=1,
    measured_iterations=3,
)


class FakeEncoding(dict):
    def __init__(self, word_ids, **values):
        super().__init__(values)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class FakeFastTokenizer:
    is_fast = True
    pad_token_id = 0

    def __call__(self, words, **kwargs):
        del kwargs
        input_ids = [101]
        word_ids = [None]
        next_id = 10
        for word_index, word in enumerate(words):
            pieces = 8 if word == "oversegmented" else 1
            for _ in range(pieces):
                input_ids.append(next_id)
                word_ids.append(word_index)
                next_id += 1
        input_ids.append(102)
        word_ids.append(None)
        return FakeEncoding(
            word_ids,
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
        )


class FakeBenchmarkRunner:
    def __init__(self, batched_size):
        self.batched_size = batched_size
        self.calls = []
        self.private_marker = PRIVATE_MARKER

    def warmup(self, batch_size):
        self.calls.append(("warmup", batch_size))

    def reset_peak_memory(self):
        self.calls.append(("reset_peak_memory",))

    def measure_seconds(self, batch_size):
        self.calls.append(("measure_seconds", batch_size))
        return 0.01 if batch_size == 1 else 0.04

    def peak_memory_bytes(self):
        self.calls.append(("peak_memory_bytes",))
        return 987_654


class FakeRuntime:
    def __init__(self, *, fail_state_load=False):
        self.calls = []
        self.tokenizer = FakeFastTokenizer()
        self.runner = None
        self.batches = None
        self.fail_state_load = fail_state_load
        self.private_marker = PRIVATE_MARKER

    def resolve_device(self, request):
        self.calls.append(("resolve_device", request))
        return "cpu"

    def memory_measurement_method(self, *, device):
        self.calls.append(("memory_measurement_method", device))
        return INJECTED_MEMORY_MEASUREMENT_METHOD

    def load_tokenizer(self, config):
        self.calls.append(("load_tokenizer", config.variant))
        return self.tokenizer

    def build_model(self, config, num_labels, device):
        self.calls.append(
            ("build_model", config.variant, num_labels, device)
        )
        return SimpleNamespace(state=None)

    def load_state_dict(self, path, *, device):
        self.calls.append(("load_state_dict", path.name, device))
        if self.fail_state_load:
            raise RuntimeError(PRIVATE_MARKER)
        return path.read_bytes()

    def restore_state_dict(self, model, state_dict):
        self.calls.append(("restore_state_dict",))
        model.state = state_dict

    def create_runner(
        self,
        model,
        batches,
        *,
        device,
        memory_measurement_method,
    ):
        self.calls.append(
            ("create_runner", device, memory_measurement_method)
        )
        self.batches = batches
        self.runner = FakeBenchmarkRunner(len(batches.batched.example_ids))
        return self.runner

    def package_versions(self):
        self.calls.append(("package_versions",))
        return {
            "python": "3.12.11",
            "torch": "fake-2.13",
            "transformers": "fake-4.53",
        }

    def hardware(self, *, device):
        self.calls.append(("hardware", device))
        return {
            "accelerator": "fake-cpu",
            "machine": "fake-machine",
            "operating_system": "fake-os",
        }


def _example(example_id, split, words):
    return WordLevelSRLExample(
        example_id=example_id,
        document_id=f"document-{example_id}",
        sentence_id=f"sentence-{example_id}",
        split=split,
        words=words,
        predicate_index=1,
        tags=("B-ARG0", "B-V", "B-ARG1"),
        predicate_roleset="review.01",
    )


def _write_dataset(path, *, retained_test_count=4):
    examples = [
        _example("train-one", "train", ("Ari", "reviewed", "files")),
        _example(
            "development-one",
            "development",
            ("Bea", "checked", "forms"),
        ),
        _example(
            f"test-00-overlength-{PRIVATE_MARKER}",
            "test",
            ("oversegmented", "reviewed", "records"),
        ),
    ]
    for index in range(retained_test_count):
        examples.append(
            _example(
                f"test-{index + 1:02d}-{PRIVATE_MARKER}",
                "test",
                (f"Person{index}", "reviewed", f"record{index}"),
            )
        )
    return write_prepared_dataset(path, examples)


def _write_config(path, fingerprint, *, variant="predicate_signal"):
    path.write_text(
        "\n".join(
            (
                "config_version = 1",
                f'variant = "{variant}"',
                'model_id = "fake/bert"',
                f'model_revision = "{MODEL_REVISION}"',
                'tokenizer_id = "fake/tokenizer"',
                f'tokenizer_revision = "{TOKENIZER_REVISION}"',
                f'prepared_data_fingerprint = "{fingerprint}"',
                "max_length = 8",
                "batch_size = 2",
                "learning_rate = 0.00005",
                "epochs = 2",
                "weight_decay = 0.01",
                "warmup_ratio = 0.1",
                "gradient_clip_norm = 1.0",
                "paired_seeds = [7, 11, 13]",
                'checkpoint_selection_metric = "development_argument_f1"',
                'device_request = "cpu"',
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_checkpoint(path, config, dataset_directory):
    dataset = read_prepared_dataset(dataset_directory)
    training = tuple(
        example for example in dataset.examples if example.split == "train"
    )
    vocabulary = build_training_label_vocabulary(training)
    metadata = RunMetadata.create(
        git_revision=GIT_REVISION,
        config_digest=config.digest,
        dataset_fingerprint=config.prepared_data_fingerprint,
        labels=vocabulary.labels,
        package_versions={"python": "3.12.11", "torch": "fake-2.13"},
        hardware={"accelerator": "fake-cpu", "machine": "fake-machine"},
        started_at="2026-08-10T08:00:00Z",
        recorded_at="2026-08-10T09:00:00Z",
        seed=7,
        variant=config.variant,
        requested_device=config.device_request,
        resolved_device="cpu",
        counts={"test_evaluations": 1, "train_retained": 1},
        drop_stats={},
        initial_state_fingerprint=INITIAL_STATE,
    )
    label_config = CheckpointLabelConfig.from_labels(vocabulary.labels)

    def save_state(state_dict, destination):
        destination.write_bytes(state_dict)

    save_checkpoint_bundle(
        path,
        metadata=metadata,
        config=config,
        label_config=label_config,
        state_dict=b"PRIVATE-CHECKPOINT-STATE",
        save_state_dict=save_state,
    )


def _case(root, *, variant="predicate_signal", retained_test_count=4):
    dataset = root / "dataset"
    manifest = _write_dataset(
        dataset,
        retained_test_count=retained_test_count,
    )
    config_path = root / "config.toml"
    _write_config(config_path, manifest.dataset_fingerprint, variant=variant)
    config = load_training_config(config_path)
    checkpoint = root / "checkpoint"
    _write_checkpoint(checkpoint, config, dataset)
    output_directory = root / "runs"
    output_directory.mkdir()
    return SimpleNamespace(
        dataset=dataset,
        manifest=manifest,
        config_path=config_path,
        config=config,
        checkpoint=checkpoint,
        output=output_directory / "systems.json",
    )


def _argv(case):
    return [
        "--config",
        str(case.config_path),
        "--dataset",
        str(case.dataset),
        "--checkpoint",
        str(case.checkpoint),
        "--output",
        str(case.output),
    ]


def _run_main(case, runtime):
    stdout = StringIO()
    stderr = StringIO()
    code = main(
        _argv(case),
        runtime_factory=lambda: runtime,
        protocol=TEST_PROTOCOL,
        output_path_policy=lambda path: None,
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


class CheckpointBenchmarkCliTests(unittest.TestCase):
    def test_validates_loads_benchmarks_and_publishes_only_aggregates(self):
        with TemporaryDirectory() as temporary:
            case = _case(Path(temporary))
            runtime = FakeRuntime()
            expected_size = sum(
                (case.checkpoint / name).stat().st_size
                for name in (
                    METADATA_FILENAME,
                    LABEL_CONFIG_FILENAME,
                    STATE_DICT_FILENAME,
                )
            )
            expected_state_digest = hashlib.sha256(
                (case.checkpoint / STATE_DICT_FILENAME).read_bytes()
            ).hexdigest()

            code, stdout, stderr = _run_main(case, runtime)

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            result = SystemsBenchmarkResult.from_canonical_json_bytes(
                case.output.read_bytes()
            )
            self.assertEqual(result.variant, "predicate_signal")
            self.assertEqual(
                result.identity.dataset_fingerprint,
                case.manifest.dataset_fingerprint,
            )
            self.assertEqual(result.identity.config_digest, case.config.digest)
            self.assertEqual(
                result.identity.checkpoint_digest,
                expected_state_digest,
            )
            self.assertEqual(result.checkpoint_size_bytes, expected_size)
            self.assertEqual(
                result.memory_measurement_method,
                INJECTED_MEMORY_MEASUREMENT_METHOD,
            )
            self.assertEqual(result.peak_memory_bytes, 987_654)
            self.assertEqual(
                tuple(sum(row) for row in runtime.batches.batched.token_type_ids),
                (1, 1, 1, 1),
            )
            emitted = case.output.read_bytes() + stdout.encode("ascii")
            self.assertNotIn(PRIVATE_MARKER.encode("ascii"), emitted)
            self.assertNotIn(str(case.dataset).encode("utf-8"), emitted)
            self.assertNotIn(b"example_id", emitted)
            self.assertNotIn(b"raw", emitted)
            self.assertEqual(json.loads(stdout)["status"], "complete")

    def test_no_signal_variant_zeroes_the_internal_predicate_indicator(self):
        with TemporaryDirectory() as temporary:
            case = _case(Path(temporary), variant="no_predicate_signal")
            runtime = FakeRuntime()

            code, _, stderr = _run_main(case, runtime)

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertEqual(
                tuple(sum(row) for row in runtime.batches.batched.token_type_ids),
                (0, 0, 0, 0),
            )
            result = SystemsBenchmarkResult.from_canonical_json_bytes(
                case.output.read_bytes()
            )
            self.assertEqual(result.variant, "no_predicate_signal")

    def test_selection_is_fixed_skips_overlength_and_emits_no_ids(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = _case(root)
            first_runtime = FakeRuntime()
            first_code, _, _ = _run_main(case, first_runtime)
            first_ids = first_runtime.batches.batched.example_ids

            case.output = root / "runs" / "second.json"
            second_runtime = FakeRuntime()
            second_code, _, _ = _run_main(case, second_runtime)

            self.assertEqual((first_code, second_code), (0, 0))
            self.assertEqual(
                second_runtime.batches.batched.example_ids,
                first_ids,
            )
            self.assertEqual(len(first_ids), TEST_PROTOCOL.batched_batch_size)
            self.assertTrue(all("overlength" not in item for item in first_ids))
            self.assertNotIn(
                PRIVATE_MARKER.encode("ascii"),
                case.output.read_bytes(),
            )

    def test_fingerprint_mismatch_stops_before_runtime_or_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = _case(root)
            _write_config(case.config_path, "f" * 64)
            called = []
            stderr = StringIO()

            code = main(
                _argv(case),
                runtime_factory=lambda: called.append(True),
                protocol=TEST_PROTOCOL,
                output_path_policy=lambda path: None,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(code, 2)
            self.assertEqual(called, [])
            self.assertFalse(case.output.exists())
            self.assertEqual(
                json.loads(stderr.getvalue())["phase"],
                "dataset_validation",
            )

    def test_corrupt_bundle_is_rejected_before_runtime_state_loading(self):
        with TemporaryDirectory() as temporary:
            case = _case(Path(temporary))
            (case.checkpoint / STATE_DICT_FILENAME).write_bytes(b"tampered")
            called = []
            stderr = StringIO()

            code = main(
                _argv(case),
                runtime_factory=lambda: called.append(True),
                protocol=TEST_PROTOCOL,
                output_path_policy=lambda path: None,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(code, 2)
            self.assertEqual(called, [])
            self.assertFalse(case.output.exists())
            self.assertEqual(
                json.loads(stderr.getvalue())["phase"],
                "checkpoint_validation",
            )

    def test_existing_or_unapproved_output_is_rejected_without_runtime(self):
        with TemporaryDirectory() as temporary:
            case = _case(Path(temporary))
            case.output.write_text("occupied", encoding="utf-8")
            called = []
            stderr = StringIO()

            code = main(
                _argv(case),
                runtime_factory=lambda: called.append(True),
                protocol=TEST_PROTOCOL,
                output_path_policy=lambda path: None,
                stdout=StringIO(),
                stderr=stderr,
            )

            self.assertEqual(code, 2)
            self.assertEqual(called, [])
            self.assertEqual(case.output.read_text(encoding="utf-8"), "occupied")
            self.assertEqual(
                json.loads(stderr.getvalue())["phase"],
                "output_validation",
            )

            case.output.unlink()
            code = main(
                _argv(case),
                runtime_factory=lambda: called.append(True),
                protocol=TEST_PROTOCOL,
                output_path_policy=lambda path: (_ for _ in ()).throw(
                    ValueError("not ignored")
                ),
                stdout=StringIO(),
                stderr=StringIO(),
            )
            self.assertEqual(code, 2)
            self.assertEqual(called, [])

    def test_state_loader_failure_is_sanitized_and_publishes_nothing(self):
        with TemporaryDirectory() as temporary:
            case = _case(Path(temporary))
            runtime = FakeRuntime(fail_state_load=True)

            code, stdout, stderr = _run_main(case, runtime)

            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertFalse(case.output.exists())
            self.assertNotIn(PRIVATE_MARKER, stderr)
            failure = json.loads(stderr)
            self.assertEqual(failure["phase"], "runtime_initialization")
            self.assertEqual(failure["error_type"], "RuntimeError")

    def test_insufficient_retained_test_data_fails_before_model_or_state(self):
        with TemporaryDirectory() as temporary:
            case = _case(Path(temporary), retained_test_count=3)
            runtime = FakeRuntime()

            code, _, stderr = _run_main(case, runtime)

            self.assertEqual(code, 1)
            call_names = tuple(call[0] for call in runtime.calls)
            self.assertIn("load_tokenizer", call_names)
            self.assertNotIn("build_model", call_names)
            self.assertNotIn("load_state_dict", call_names)
            self.assertFalse(case.output.exists())
            self.assertNotIn(PRIVATE_MARKER, stderr)


def _padded_batch(count):
    example_ids = tuple(f"internal-{index}" for index in range(count))
    return PaddedSRLBatch(
        example_ids=example_ids,
        word_counts=(1,) * count,
        word_ids=((None, 0, None),) * count,
        input_ids=((101, 10, 102),) * count,
        attention_mask=((1, 1, 1),) * count,
        token_type_ids=((0, 1, 0),) * count,
        labels=((-100, 0, -100),) * count,
    )


class FakeCuda:
    def __init__(self):
        self.calls = []

    def synchronize(self, device):
        self.calls.append(("synchronize", device))

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset", device))

    def max_memory_allocated(self, device):
        self.calls.append(("peak", device))
        return 456_789


class FakeMps:
    def __init__(self):
        self.calls = []

    def synchronize(self):
        self.calls.append(("synchronize",))


class FakeTorch:
    long = "long"

    def __init__(self):
        self.cuda = FakeCuda()
        self.mps = FakeMps()

    def device(self, name):
        return f"device:{name}"

    def tensor(self, values, *, dtype, device):
        return (tuple(values), dtype, device)

    def inference_mode(self):
        return nullcontext()


class FakeModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        self.calls.append(("eval",))

    def __call__(self, **inputs):
        self.calls.append(("infer", tuple(sorted(inputs))))
        return SimpleNamespace(logits="unused")


class TorchRunnerSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.batches = FixedBenchmarkBatches(
            single_example=_padded_batch(1),
            batched=_padded_batch(4),
        )

    def test_cuda_synchronizes_timing_and_uses_resettable_allocator_peak(self):
        torch = FakeTorch()
        model = FakeModel()
        timer = iter((10.0, 10.25))
        runner = TorchSystemsBenchmarkRunner(
            torch_module=torch,
            model=model,
            batches=self.batches,
            device="cuda",
            memory_measurement_method=CUDA_MEMORY_MEASUREMENT_METHOD,
            timer=lambda: next(timer),
        )

        runner.warmup(1)
        runner.reset_peak_memory()
        elapsed = runner.measure_seconds(4)
        peak = runner.peak_memory_bytes()

        self.assertEqual(elapsed, 0.25)
        self.assertEqual(peak, 456_789)
        self.assertEqual(
            tuple(call[0] for call in torch.cuda.calls),
            (
                "synchronize",
                "synchronize",
                "synchronize",
                "reset",
                "synchronize",
                "synchronize",
                "synchronize",
                "peak",
            ),
        )
        self.assertEqual(
            tuple(call[0] for call in model.calls),
            ("eval", "infer", "infer"),
        )

    def test_process_peak_rss_has_explicit_darwin_and_linux_units(self):
        resource_module = SimpleNamespace(
            RUSAGE_SELF="self",
            getrusage=lambda target: SimpleNamespace(ru_maxrss=123),
        )
        for system, expected in (("Darwin", 123), ("Linux", 125_952)):
            with self.subTest(system=system):
                runner = TorchSystemsBenchmarkRunner(
                    torch_module=FakeTorch(),
                    model=FakeModel(),
                    batches=self.batches,
                    device="cpu",
                    memory_measurement_method=(
                        PROCESS_MEMORY_MEASUREMENT_METHOD
                    ),
                    resource_module=resource_module,
                    platform_system=lambda: system,
                )
                runner.reset_peak_memory()
                self.assertEqual(runner.peak_memory_bytes(), expected)

    def test_mps_synchronizes_and_unknown_peak_units_fail_closed(self):
        resource_module = SimpleNamespace(
            RUSAGE_SELF="self",
            getrusage=lambda target: SimpleNamespace(ru_maxrss=123),
        )
        torch = FakeTorch()
        runner = TorchSystemsBenchmarkRunner(
            torch_module=torch,
            model=FakeModel(),
            batches=self.batches,
            device="mps",
            memory_measurement_method=PROCESS_MEMORY_MEASUREMENT_METHOD,
            resource_module=resource_module,
            platform_system=lambda: "Darwin",
        )

        runner.warmup(1)

        self.assertEqual(len(torch.mps.calls), 2)
        with self.assertRaises(UnsupportedPeakMemoryError):
            TorchSystemsBenchmarkRunner(
                torch_module=FakeTorch(),
                model=FakeModel(),
                batches=self.batches,
                device="cpu",
                memory_measurement_method=PROCESS_MEMORY_MEASUREMENT_METHOD,
                resource_module=resource_module,
                platform_system=lambda: "Windows",
            )


if __name__ == "__main__":
    unittest.main()
