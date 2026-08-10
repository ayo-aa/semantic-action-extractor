from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semantic_action_extractor.srl.batching import PaddedSRLBatch
from semantic_action_extractor.srl.dataset_io import write_prepared_dataset
from semantic_action_extractor.srl.evaluation import (
    evaluate_supplied_predicate_srl,
)
from semantic_action_extractor.srl.example import WordLevelSRLExample
from semantic_action_extractor.srl.experiment_config import TrainingConfig
from semantic_action_extractor.srl.training_engine import (
    BatchRuntimeOutput,
    EvaluationSummary,
    OptionalTrainingDependencyError,
    RunContext,
    TorchTrainingRuntime,
    deterministic_batch_indices,
    run_paired_srl_experiments,
    run_srl_experiment,
)


MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40
GIT_REVISION = "c" * 40
RECORDED_AT = "2026-08-10T09:00:00Z"
CONTEXT = RunContext(
    git_revision=GIT_REVISION,
    started_at="2026-08-10T08:00:00Z",
)


def _utc_now():
    return RECORDED_AT


class FakeEncoding(dict):
    def __init__(self, word_ids, **values):
        super().__init__(values)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class FakeFastTokenizer:
    is_fast = True
    pad_token_id = 0

    def __init__(self, piece_counts=None):
        self.piece_counts = piece_counts or {}
        self.calls = []

    def __call__(self, words, **kwargs):
        self.calls.append((tuple(words), dict(kwargs)))
        input_ids = [101]
        word_ids = [None]
        next_id = 10
        for word_index, word in enumerate(words):
            for _ in range(self.piece_counts.get(word, 1)):
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


class FakeRuntime:
    def __init__(self, *, nonfinite_train=False):
        self.nonfinite_train = nonfinite_train
        self.tokenizer = FakeFastTokenizer(
            {"oversegmented": 4, "segmented": 2}
        )
        self.loaded_configs = []
        self.seed_calls = []
        self.train_batches = []
        self.evaluation_splits = []
        self.restore_updates = []
        self.test_observed_updates = []
        self.scheduler_parameters = None

    def load_tokenizer(self, config):
        self.loaded_configs.append(config)
        return self.tokenizer

    def resolve_device(self, request):
        if request not in {"auto", "cpu"}:
            raise RuntimeError("fake runtime only supports CPU")
        return "cpu"

    def seed_everything(self, seed):
        self.seed_calls.append(seed)
        self.current_seed = seed

    def build_model(self, config, num_labels, device):
        return SimpleNamespace(
            updates=0,
            seed=self.current_seed,
            variant=config.variant,
            selection_metric=config.checkpoint_selection_metric,
            num_labels=num_labels,
            device=device,
        )

    def initial_state_fingerprint(self, model):
        return hashlib.sha256(f"initial:{model.seed}".encode()).hexdigest()

    def create_optimizer(self, model, config):
        return SimpleNamespace(model=model, learning_rate=config.learning_rate)

    def create_scheduler(self, optimizer, *, warmup_steps, total_steps):
        self.scheduler_parameters = (warmup_steps, total_steps)
        return SimpleNamespace(optimizer=optimizer)

    def train_batch(
        self,
        model,
        batch,
        optimizer,
        scheduler,
        *,
        gradient_clip_norm,
        device,
    ):
        del optimizer, scheduler, gradient_clip_norm, device
        self.train_batches.append(
            (batch.example_ids, batch.token_type_ids, model.variant)
        )
        if self.nonfinite_train:
            return float("nan")
        model.updates += 1
        return float(
            sum(label != -100 for row in batch.labels for label in row)
        )

    def evaluate_batch(self, model, batch, *, device):
        del device
        split = (
            "development"
            if batch.example_ids[0].startswith("development-")
            else "test"
        )
        self.evaluation_splits.append(split)
        if split == "test":
            self.test_observed_updates.append(model.updates)
        predictions = []
        good_predictions = split == "test" or model.updates == 2
        for labels in batch.labels:
            if good_predictions:
                predictions.append(
                    tuple(0 if label == -100 else label for label in labels)
                )
            else:
                predictions.append((0,) * len(labels))
        loss = 0.5 if good_predictions else 2.0
        if (
            split == "development"
            and model.selection_metric == "development_loss"
        ):
            loss = 2.0 if good_predictions else 0.5
        return BatchRuntimeOutput(
            loss=loss,
            predicted_label_ids=tuple(predictions),
        )

    def capture_state_dict(self, model):
        return {"updates": model.updates}

    def restore_state_dict(self, model, state_dict):
        model.updates = state_dict["updates"]
        self.restore_updates.append(model.updates)

    def save_state_dict(self, state_dict, path):
        path.write_text(json.dumps(state_dict, sort_keys=True), encoding="utf-8")

    def load_state_dict(self, path, *, device):
        del device
        return json.loads(path.read_text(encoding="utf-8"))

    def package_versions(self):
        return {
            "python": "3.12.11",
            "torch": "fake-2.7",
            "transformers": "fake-4.53",
        }

    def hardware(self, *, device):
        return {
            "accelerator": "fake-cpu",
            "machine": "fake-machine",
            "operating_system": f"fake-os-{device}",
        }


def _example(example_id, split, words):
    return WordLevelSRLExample(
        example_id=example_id,
        document_id=example_id,
        sentence_id=f"{example_id}:sentence",
        split=split,
        words=words,
        predicate_index=1,
        tags=("B-ARG0", "B-V", "B-ARG1"),
        predicate_roleset="invent.01",
    )


def _write_dataset(directory):
    examples = (
        _example(
            "train-overlength",
            "train",
            ("oversegmented", "indexed", "records"),
        ),
        _example("train-short", "train", ("Nia", "indexed", "files")),
        _example(
            "train-subwords",
            "train",
            ("segmented", "sorted", "entries"),
        ),
        _example(
            "development-one",
            "development",
            ("Omar", "reviewed", "forms"),
        ),
        _example("test-one", "test", ("Pia", "approved", "notes")),
    )
    return write_prepared_dataset(directory, examples)


def _config(fingerprint, *, variant="predicate_signal", **changes):
    values = {
        "config_version": 1,
        "variant": variant,
        "model_id": "google-bert/bert-base-uncased",
        "model_revision": MODEL_REVISION,
        "tokenizer_id": "google-bert/bert-base-uncased",
        "tokenizer_revision": TOKENIZER_REVISION,
        "prepared_data_fingerprint": fingerprint,
        "max_length": 6,
        "batch_size": 1,
        "learning_rate": 0.00001,
        "epochs": 2,
        "weight_decay": 0.0,
        "warmup_ratio": 0.5,
        "gradient_clip_norm": 1.0,
        "paired_seeds": (11, 17, 23),
        "checkpoint_selection_metric": "development_argument_f1",
        "device_request": "cpu",
    }
    values.update(changes)
    return TrainingConfig(**values)


def _replace_result_scores(result, *, development_f1, test_f1):
    development_evaluation = replace(
        result.best_development.evaluation,
        arguments=replace(
            result.best_development.evaluation.arguments,
            f1=development_f1,
        ),
    )
    development = replace(
        result.best_development,
        evaluation=development_evaluation,
    )
    epochs = list(result.epochs)
    epochs[result.best_epoch - 1] = replace(
        epochs[result.best_epoch - 1],
        development=development,
        selection_value=development_f1,
    )
    test = replace(
        result.test,
        evaluation=replace(
            result.test.evaluation,
            arguments=replace(
                result.test.evaluation.arguments,
                f1=test_f1,
            ),
        ),
    )
    return replace(
        result,
        best_development=development,
        test=test,
        epochs=tuple(epochs),
    )


class TrainingEngineTests(unittest.TestCase):
    def test_runs_complete_train_select_reload_and_single_test_flow(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            manifest = _write_dataset(dataset)
            config = _config(manifest.dataset_fingerprint)
            runtime = FakeRuntime()
            checkpoint = root / "best-checkpoint"
            clock_observations = []

            def observed_clock():
                clock_observations.append(tuple(runtime.evaluation_splits))
                return RECORDED_AT

            result = run_srl_experiment(
                dataset,
                config,
                checkpoint,
                seed=11,
                context=CONTEXT,
                runtime=runtime,
                utc_now=observed_clock,
            )

            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(
                result.best_development.evaluation.arguments.f1, 1.0
            )
            self.assertEqual(result.test.evaluation.arguments.f1, 1.0)
            self.assertEqual(
                runtime.evaluation_splits,
                ["development", "development", "test"],
            )
            self.assertEqual(runtime.scheduler_parameters, (2, 4))
            self.assertEqual(runtime.seed_calls, [11])
            self.assertEqual(runtime.restore_updates, [2, 2])
            self.assertEqual(runtime.test_observed_updates, [2])
            self.assertEqual(
                clock_observations,
                [("development", "development", "test")],
            )
            self.assertEqual(result.metadata.recorded_at, RECORDED_AT)
            self.assertEqual(len(runtime.train_batches), 4)
            self.assertAlmostEqual(result.epochs[0].training_loss, 25 / 7)
            self.assertAlmostEqual(result.epochs[1].training_loss, 25 / 7)
            self.assertTrue(
                all(
                    sum(row) == 1
                    for _, token_types, _ in runtime.train_batches
                    for row in token_types
                )
            )
            self.assertTrue(
                all(
                    call[1]["truncation"] is False
                    for call in runtime.tokenizer.calls
                )
            )
            self.assertTrue(checkpoint.is_dir())
            self.assertEqual(
                {path.name for path in checkpoint.iterdir()},
                {"checkpoint_metadata.json", "labels.json", "state_dict.pt"},
            )
            self.assertEqual(
                json.loads(
                    (checkpoint / "state_dict.pt").read_text(encoding="utf-8")
                ),
                {"updates": 2},
            )

            counts = dict(result.metadata.counts)
            drops = dict(result.metadata.drop_stats)
            self.assertEqual(counts["optimizer_steps"], 4)
            self.assertEqual(counts["train_records"], 3)
            self.assertEqual(counts["train_retained"], 2)
            self.assertEqual(counts["train_active_tokens"], 7)
            self.assertEqual(counts["training_active_tokens"], 14)
            self.assertEqual(counts["test_evaluations"], 1)
            self.assertEqual(drops["train_overlength"], 1)
            self.assertEqual(drops["total_overlength"], 1)
            self.assertEqual(
                result.metadata.dataset_fingerprint,
                manifest.dataset_fingerprint,
            )

            encoded = result.canonical_json_bytes()
            self.assertEqual(
                encoded,
                json.dumps(
                    json.loads(encoded),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode(),
            )
            for corpus_word in (b"Nia", b"indexed", b"Omar", b"approved"):
                self.assertNotIn(corpus_word, encoded)
            self.assertEqual(
                result.digest, hashlib.sha256(encoded).hexdigest()
            )

    def test_checks_dataset_fingerprint_before_loading_dependencies(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_dataset(root / "prepared")
            config = _config("0" * 64)
            self.assertNotEqual(
                config.prepared_data_fingerprint,
                manifest.dataset_fingerprint,
            )
            with patch(
                "semantic_action_extractor.srl.training_engine."
                "TorchTrainingRuntime"
            ) as runtime_constructor:
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    run_srl_experiment(
                        root / "prepared",
                        config,
                        root / "checkpoint",
                        seed=11,
                        context=CONTEXT,
                        utc_now=_utc_now,
                    )

            runtime_constructor.assert_not_called()

    def test_can_select_checkpoint_by_development_loss(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_dataset(root / "prepared")
            config = _config(
                manifest.dataset_fingerprint,
                checkpoint_selection_metric="development_loss",
            )
            runtime = FakeRuntime()

            result = run_srl_experiment(
                root / "prepared",
                config,
                root / "loss-checkpoint",
                seed=11,
                context=CONTEXT,
                runtime=runtime,
                utc_now=_utc_now,
            )

            self.assertEqual(result.best_epoch, 2)
            self.assertEqual(result.best_development.loss, 0.5)
            self.assertEqual(
                result.best_development.evaluation.arguments.f1, 0.0
            )
            self.assertEqual(runtime.test_observed_updates, [4])

    def test_rejects_empty_retained_split_and_nonfinite_loss(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_dataset(root / "prepared")
            too_short = _config(manifest.dataset_fingerprint, max_length=3)
            runtime = FakeRuntime()
            with self.assertRaisesRegex(ValueError, "empty retained train"):
                run_srl_experiment(
                    root / "prepared",
                    too_short,
                    root / "empty-checkpoint",
                    seed=11,
                    context=CONTEXT,
                    runtime=runtime,
                    utc_now=_utc_now,
                )
            self.assertFalse((root / "empty-checkpoint").exists())

            nonfinite = FakeRuntime(nonfinite_train=True)
            with self.assertRaisesRegex(ValueError, "must be finite"):
                run_srl_experiment(
                    root / "prepared",
                    _config(manifest.dataset_fingerprint),
                    root / "nan-checkpoint",
                    seed=11,
                    context=CONTEXT,
                    runtime=nonfinite,
                    utc_now=_utc_now,
                )
            self.assertFalse((root / "nan-checkpoint").exists())

        evaluation = evaluate_supplied_predicate_srl(
            [("B-ARG0", "B-V")],
            [("B-ARG0", "B-V")],
            [1],
        )
        invalid_arguments = replace(evaluation.arguments, f1=float("nan"))
        with self.assertRaisesRegex(ValueError, "argument f1 must be finite"):
            EvaluationSummary(
                loss=0.0,
                evaluation=replace(
                    evaluation, arguments=invalid_arguments
                ),
            )

    def test_deterministic_batches_are_complete_and_epoch_specific(self):
        first = deterministic_batch_indices(
            9, batch_size=4, seed=17, epoch=1, shuffle=True
        )
        repeated = deterministic_batch_indices(
            9, batch_size=4, seed=17, epoch=1, shuffle=True
        )
        next_epoch = deterministic_batch_indices(
            9, batch_size=4, seed=17, epoch=2, shuffle=True
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(
            sorted(index for batch in first for index in batch),
            list(range(9)),
        )
        self.assertEqual(tuple(map(len, first)), (4, 4, 1))
        self.assertEqual(
            deterministic_batch_indices(
                5, batch_size=2, seed=17, epoch=0, shuffle=False
            ),
            ((0, 1), (2, 3), (4,)),
        )

    def test_runs_and_validates_three_exactly_paired_seeds(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_dataset(root / "prepared")
            conditioned = _config(manifest.dataset_fingerprint)
            ablated = _config(
                manifest.dataset_fingerprint,
                variant="no_predicate_signal",
            )
            calls = []
            runtimes = []
            conditioned_scores = {11: 0.2, 17: 0.4, 23: 0.6}
            ablated_scores = {11: 0.1, 17: 0.2, 23: 0.3}

            def run_one(config, seed):
                calls.append((config.variant, seed))
                runtime = FakeRuntime()
                runtimes.append(runtime)
                result = run_srl_experiment(
                    root / "prepared",
                    config,
                    root / f"{config.variant}-{seed}",
                    seed=seed,
                    context=CONTEXT,
                    runtime=runtime,
                    utc_now=_utc_now,
                )
                score = (
                    conditioned_scores[seed]
                    if config.variant == "predicate_signal"
                    else ablated_scores[seed]
                )
                return _replace_result_scores(
                    result,
                    development_f1=score,
                    test_f1=score / 2,
                )

            paired = run_paired_srl_experiments(
                conditioned,
                ablated,
                run_one=run_one,
            )

            self.assertEqual(len(calls), 6)
            self.assertEqual(tuple(pair.seed for pair in paired.pairs), (11, 17, 23))
            self.assertEqual(
                tuple(
                    round(pair.development_argument_f1_delta, 10)
                    for pair in paired.pairs
                ),
                (0.1, 0.2, 0.3),
            )
            self.assertEqual(
                tuple(item.variant for item in paired.aggregates),
                ("predicate_signal", "no_predicate_signal"),
            )
            self.assertAlmostEqual(
                paired.paired_effect.development_argument_f1_delta_mean,
                0.2,
            )
            self.assertAlmostEqual(
                paired.paired_effect.development_argument_f1_delta_sample_stddev,
                0.1,
            )
            self.assertAlmostEqual(
                paired.paired_effect.test_argument_f1_delta_mean,
                0.1,
            )
            self.assertAlmostEqual(
                paired.paired_effect.test_argument_f1_delta_sample_stddev,
                0.05,
            )
            self.assertAlmostEqual(
                paired.aggregates[0].development_argument_f1_sample_stddev,
                0.2,
            )
            self.assertAlmostEqual(
                paired.aggregates[1].development_argument_f1_sample_stddev,
                0.1,
            )
            self.assertTrue(
                all(
                    value == 0
                    for runtime in runtimes
                    if runtime.loaded_configs[0].variant == "no_predicate_signal"
                    for _, token_types, _ in runtime.train_batches
                    for row in token_types
                    for value in row
                )
            )
            encoded = paired.canonical_json_bytes()
            self.assertIn(b"sample_stddev", encoded)
            self.assertNotIn(b"population_stddev", encoded)
            self.assertEqual(
                paired.digest, hashlib.sha256(encoded).hexdigest()
            )

            called = []
            with self.assertRaisesRegex(ValueError, "batch_size"):
                run_paired_srl_experiments(
                    conditioned,
                    replace(ablated, batch_size=2),
                    run_one=lambda config, seed: called.append((config, seed)),
                )
            self.assertEqual(called, [])
            with self.assertRaisesRegex(ValueError, "device_request"):
                run_paired_srl_experiments(
                    conditioned,
                    replace(ablated, device_request="auto"),
                    run_one=lambda config, seed: called.append((config, seed)),
                )
            self.assertEqual(called, [])

            def mismatched_initial(config, seed):
                result = run_srl_experiment(
                    root / "prepared",
                    config,
                    root / f"bad-{config.variant}-{seed}",
                    seed=seed,
                    context=CONTEXT,
                    runtime=FakeRuntime(),
                    utc_now=_utc_now,
                )
                if config.variant == "no_predicate_signal":
                    result = replace(
                        result,
                        metadata=replace(
                            result.metadata,
                            initial_state_fingerprint="9" * 64,
                        ),
                    )
                return result

            with self.assertRaisesRegex(ValueError, "initial state fingerprint"):
                run_paired_srl_experiments(
                    conditioned,
                    ablated,
                    run_one=mismatched_initial,
                )

    def test_real_runtime_pins_fast_tokenizer_and_resolves_device(self):
        tokenizer = FakeFastTokenizer()

        class FakeAutoTokenizer:
            calls = []

            @classmethod
            def from_pretrained(cls, identifier, **kwargs):
                cls.calls.append((identifier, kwargs))
                return tokenizer

        cuda = SimpleNamespace(is_available=lambda: False)
        backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False)
        )
        torch = SimpleNamespace(cuda=cuda, backends=backends)
        transformers = SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
        runtime = TorchTrainingRuntime(
            torch_module=torch,
            transformers_module=transformers,
        )
        config = _config("d" * 64)

        loaded = runtime.load_tokenizer(config)

        self.assertIs(loaded, tokenizer)
        self.assertEqual(
            FakeAutoTokenizer.calls,
            [
                (
                    config.tokenizer_id,
                    {
                        "revision": config.tokenizer_revision,
                        "use_fast": True,
                    },
                )
            ],
        )
        self.assertEqual(runtime.resolve_device("auto"), "cpu")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            runtime.resolve_device("cuda")
        cuda.is_available = lambda: True
        self.assertEqual(runtime.resolve_device("auto"), "cuda")
        cuda.is_available = lambda: False
        backends.mps.is_available = lambda: True
        self.assertEqual(runtime.resolve_device("auto"), "mps")
        self.assertEqual(runtime.resolve_device("mps"), "mps")

        tokenizer.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            runtime.load_tokenizer(config)

    def test_real_runtime_seeds_and_executes_adamw_training_step(self):
        calls = []

        class FakeLoss:
            def detach(self):
                return self

            def cpu(self):
                return self

            def item(self):
                return 0.75

            def backward(self):
                calls.append(("backward",))

        class FakeModel:
            def train(self):
                calls.append(("train",))

            def parameters(self):
                return ("encoder", "classifier")

            def __call__(self, **inputs):
                calls.append(("model", inputs))
                return SimpleNamespace(loss=FakeLoss())

        class FakeOptimizer:
            def zero_grad(self, *, set_to_none):
                calls.append(("zero_grad", set_to_none))

            def step(self):
                calls.append(("optimizer_step",))

        class FakeScheduler:
            def step(self):
                calls.append(("scheduler_step",))

        optimizer = FakeOptimizer()
        scheduler = FakeScheduler()

        def adamw(parameters, **kwargs):
            calls.append(("adamw", tuple(parameters), kwargs))
            return optimizer

        def constant_schedule(candidate, **kwargs):
            calls.append(("constant_schedule", candidate, kwargs))
            return scheduler

        cuda = SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("cuda_seed", seed)),
        )
        cudnn = SimpleNamespace(deterministic=False, benchmark=True)
        torch = SimpleNamespace(
            cuda=cuda,
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False),
                cudnn=cudnn,
            ),
            manual_seed=lambda seed: calls.append(("torch_seed", seed)),
            use_deterministic_algorithms=lambda enabled: calls.append(
                ("deterministic", enabled)
            ),
            optim=SimpleNamespace(AdamW=adamw),
            nn=SimpleNamespace(
                utils=SimpleNamespace(
                    clip_grad_norm_=lambda parameters, value, **kwargs: calls.append(
                        ("clip", tuple(parameters), value, kwargs)
                    )
                )
            ),
            device=lambda value: value,
            long="long",
            tensor=lambda values, **kwargs: (
                tuple(tuple(row) for row in values),
                kwargs,
            ),
        )
        transformers = SimpleNamespace(
            set_seed=lambda seed: calls.append(("transformers_seed", seed)),
            get_constant_schedule_with_warmup=constant_schedule,
        )
        runtime = TorchTrainingRuntime(
            torch_module=torch,
            transformers_module=transformers,
        )
        config = _config("d" * 64)

        runtime.seed_everything(17)
        model = FakeModel()
        created_optimizer = runtime.create_optimizer(model, config)
        created_scheduler = runtime.create_scheduler(
            created_optimizer,
            warmup_steps=2,
            total_steps=10,
        )
        batch = PaddedSRLBatch(
            example_ids=("invented",),
            word_counts=(1,),
            word_ids=((None, 0, None),),
            input_ids=((101, 10, 102),),
            attention_mask=((1, 1, 1),),
            token_type_ids=((0, 1, 0),),
            labels=((-100, 0, -100),),
        )
        loss = runtime.train_batch(
            model,
            batch,
            created_optimizer,
            created_scheduler,
            gradient_clip_norm=1.0,
            device="cpu",
        )

        self.assertEqual(loss, 0.75)
        self.assertIn(("transformers_seed", 17), calls)
        self.assertIn(("torch_seed", 17), calls)
        self.assertIn(("cuda_seed", 17), calls)
        self.assertIn(("deterministic", True), calls)
        self.assertTrue(cudnn.deterministic)
        self.assertFalse(cudnn.benchmark)
        self.assertIn(
            (
                "adamw",
                ("encoder", "classifier"),
                {"lr": config.learning_rate, "weight_decay": 0.0},
            ),
            calls,
        )
        self.assertIn(
            (
                "constant_schedule",
                optimizer,
                {"num_warmup_steps": 2},
            ),
            calls,
        )
        self.assertIn(("zero_grad", True), calls)
        self.assertIn(("backward",), calls)
        clip_call = (
            "clip",
            ("encoder", "classifier"),
            1.0,
            {"error_if_nonfinite": True},
        )
        self.assertIn(clip_call, calls)
        self.assertLess(
            calls.index(("backward",)),
            calls.index(clip_call),
        )
        self.assertLess(
            calls.index(clip_call),
            calls.index(("optimizer_step",)),
        )
        self.assertLess(
            calls.index(("optimizer_step",)),
            calls.index(("scheduler_step",)),
        )

    def test_optional_libraries_are_loaded_only_for_real_runtime(self):
        with patch(
            "semantic_action_extractor.srl.training_engine.importlib.import_module",
            side_effect=ImportError("not installed"),
        ) as import_module:
            with self.assertRaises(OptionalTrainingDependencyError):
                TorchTrainingRuntime()

        import_module.assert_called_once_with("torch")

    def test_run_context_rejects_noncanonical_revision_and_times(self):
        with self.assertRaisesRegex(ValueError, "git_revision"):
            RunContext(
                git_revision="main",
                started_at=CONTEXT.started_at,
            )
        with self.assertRaisesRegex(ValueError, "ending in Z"):
            RunContext(
                git_revision=GIT_REVISION,
                started_at="2026-08-10T09:00:00+00:00",
            )


if __name__ == "__main__":
    unittest.main()
