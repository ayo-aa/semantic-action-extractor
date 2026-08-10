import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from semantic_action_extractor.srl.checkpoint_bundle import (
    CheckpointLabelConfig,
    LABEL_CONFIG_FILENAME,
    METADATA_FILENAME,
    STATE_DICT_FILENAME,
    load_checkpoint_bundle,
    save_checkpoint_bundle,
    validate_checkpoint_bundle,
)
from semantic_action_extractor.srl.experiment_config import (
    TrainingConfig,
    load_training_config,
    parse_training_config,
)
from semantic_action_extractor.srl.run_metadata import (
    RunMetadata,
    require_exact_run_metadata,
)


MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40
DATA_FINGERPRINT = "c" * 64
GIT_REVISION = "d" * 40
INITIAL_STATE_FINGERPRINT = "e" * 64
LABELS = ("O", "B-ARG0", "B-V", "I-ARG0", "I-V")


def _config_source(**overrides: str) -> str:
    values = {
        "config_version": "1",
        "variant": '"predicate_signal"',
        "model_id": '"google-bert/bert-base-uncased"',
        "model_revision": f'"{MODEL_REVISION}"',
        "tokenizer_id": '"google-bert/bert-base-uncased"',
        "tokenizer_revision": f'"{TOKENIZER_REVISION}"',
        "prepared_data_fingerprint": f'"{DATA_FINGERPRINT}"',
        "max_length": "512",
        "batch_size": "32",
        "learning_rate": "1e-5",
        "epochs": "2",
        "weight_decay": "0.0",
        "warmup_ratio": "0.0",
        "gradient_clip_norm": "1.0",
        "paired_seeds": "[13, 17, 23]",
        "checkpoint_selection_metric": '"development_argument_f1"',
        "device_request": '"auto"',
    }
    values.update(overrides)
    return "\n".join(f"{key} = {value}" for key, value in values.items()) + "\n"


def _config(**overrides: str) -> TrainingConfig:
    return parse_training_config(_config_source(**overrides))


def _metadata(**overrides) -> RunMetadata:
    config = overrides.pop("config", _config())
    values = {
        "git_revision": GIT_REVISION,
        "config_digest": config.digest,
        "dataset_fingerprint": config.prepared_data_fingerprint,
        "labels": LABELS,
        "package_versions": {
            "python": "3.12.11",
            "torch": "2.7.1",
            "transformers": "4.53.0",
        },
        "hardware": {
            "accelerator": "Apple M4",
            "machine": "arm64",
            "operating_system": "macOS 15.6",
        },
        "started_at": "2026-08-10T08:00:00Z",
        "recorded_at": "2026-08-10T09:30:00.125Z",
        "seed": 13,
        "variant": config.variant,
        "requested_device": config.device_request,
        "resolved_device": "mps",
        "counts": {
            "development_examples": 2,
            "test_examples": 2,
            "train_examples": 4,
            "training_steps": 8,
        },
        "drop_stats": {"tokenization_overlength": 1},
        "initial_state_fingerprint": INITIAL_STATE_FINGERPRINT,
    }
    values.update(overrides)
    return RunMetadata.create(**values)


class TrainingConfigTests(unittest.TestCase):
    def test_parses_complete_reproduction_config_and_canonical_digest(self) -> None:
        config = _config()
        same_config = parse_training_config("\n" + _config_source() + "\n")

        self.assertEqual(config.config_version, 1)
        self.assertEqual(config.model_revision, MODEL_REVISION)
        self.assertEqual(config.tokenizer_revision, TOKENIZER_REVISION)
        self.assertEqual(config.paired_seeds, (13, 17, 23))
        self.assertTrue(config.predicate_signal)
        self.assertEqual(config, same_config)
        self.assertEqual(config.digest, same_config.digest)
        self.assertEqual(
            config.digest,
            hashlib.sha256(config.canonical_json_bytes()).hexdigest(),
        )
        self.assertNotIn(b" ", config.canonical_json_bytes())
        with self.assertRaises(FrozenInstanceError):
            config.batch_size = 4

    def test_loads_utf8_toml_from_a_caller_path(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.toml"
            path.write_text(_config_source(), encoding="utf-8")

            loaded = load_training_config(path)

        self.assertEqual(loaded, _config())

    def test_rejects_missing_unknown_and_nested_keys(self) -> None:
        missing = _config_source().replace("epochs = 2\n", "")
        with self.assertRaisesRegex(ValueError, "missing keys: epochs"):
            parse_training_config(missing)

        with self.assertRaisesRegex(ValueError, "unknown keys: surprise"):
            parse_training_config(_config_source() + "surprise = 4\n")

        with self.assertRaisesRegex(ValueError, "unknown keys: optimizer"):
            parse_training_config(_config_source() + "[optimizer]\nname = 'AdamW'\n")

    def test_requires_exact_toml_types(self) -> None:
        cases = (
            ("config_version", "true", "integer"),
            ("max_length", "512.0", "integer"),
            ("batch_size", "32.0", "integer"),
            ("learning_rate", "1", "TOML float"),
            ("epochs", "2.0", "integer"),
            ("weight_decay", "0", "TOML float"),
            ("warmup_ratio", "0", "TOML float"),
            ("gradient_clip_norm", "1", "TOML float"),
            ("paired_seeds", "'13,17,23'", "TOML array"),
            ("paired_seeds", "[13, true, 23]", "must be an integer"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    _config(**{field: value})

    def test_rejects_out_of_range_numeric_values(self) -> None:
        cases = (
            ("max_length", "2", "between 3 and 512"),
            ("max_length", "513", "between 3 and 512"),
            ("batch_size", "0", "between 1 and 1024"),
            ("learning_rate", "0.0", "greater than 0"),
            ("learning_rate", "1.1", "at most 1"),
            ("epochs", "0", "between 1 and 1000"),
            ("weight_decay", "-0.1", "between 0 and 1"),
            ("weight_decay", "1.1", "between 0 and 1"),
            ("warmup_ratio", "-0.1", "at least 0"),
            ("warmup_ratio", "1.0", "less than 1"),
            ("gradient_clip_norm", "0.0", "greater than 0"),
            ("paired_seeds", "[13, 17]", "exactly three"),
            ("paired_seeds", "[13, 13, 23]", "unique"),
            ("paired_seeds", "[13, 17, 4294967296]", "4294967295"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    _config(**{field: value})

    def test_requires_immutable_artifact_pins_and_declared_enums(self) -> None:
        cases = (
            ("model_id", "'../model'", "model-hub identifier"),
            ("model_revision", "'main'", "pinned 40-character"),
            ("tokenizer_revision", f"'{TOKENIZER_REVISION.upper()}'", "lowercase"),
            ("prepared_data_fingerprint", "'abc'", "64 lowercase"),
            ("variant", "'different'", "variant must"),
            ("checkpoint_selection_metric", "'test_f1'", "selection_metric"),
            ("device_request", "'cuda:0'", "device_request"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    _config(**{field: value})

        ablation = _config(variant='"no_predicate_signal"')
        self.assertFalse(ablation.predicate_signal)

    def test_rejects_invalid_toml_and_non_string_source(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            parse_training_config(b"config_version = 1")
        with self.assertRaisesRegex(ValueError, "invalid training configuration"):
            parse_training_config("not = [valid")


class RunMetadataTests(unittest.TestCase):
    def test_freezes_mappings_and_round_trips_canonical_json(self) -> None:
        metadata = _metadata(
            package_versions={"torch": "2.7.1", "python": "3.12.11"},
            counts={"train_examples": 4, "development_examples": 2},
        )

        self.assertEqual(
            tuple(key for key, _ in metadata.package_versions),
            ("python", "torch"),
        )
        self.assertEqual(
            tuple(key for key, _ in metadata.counts),
            ("development_examples", "train_examples"),
        )
        self.assertEqual(
            metadata.digest,
            hashlib.sha256(metadata.canonical_json_bytes()).hexdigest(),
        )
        self.assertEqual(
            RunMetadata.from_canonical_json_bytes(
                metadata.canonical_json_bytes()
            ),
            metadata,
        )
        with self.assertRaises(FrozenInstanceError):
            metadata.seed = 23

    def test_rejects_noncanonical_or_ambiguous_metadata_json(self) -> None:
        metadata = _metadata()
        pretty = json.dumps(metadata.to_dict(), indent=2).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "not in canonical form"):
            RunMetadata.from_canonical_json_bytes(pretty)

        text = metadata.canonical_json_bytes().decode("utf-8")
        duplicate = text.replace(
            '"metadata_version":1',
            '"metadata_version":1,"metadata_version":1',
        ).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            RunMetadata.from_canonical_json_bytes(duplicate)

        wrong_version = metadata.to_dict()
        wrong_version["metadata_version"] = 2
        with self.assertRaisesRegex(ValueError, "metadata_version must be 1"):
            RunMetadata.from_dict(wrong_version)

        missing = metadata.to_dict()
        del missing["counts"]
        with self.assertRaisesRegex(ValueError, "missing keys: counts"):
            RunMetadata.from_dict(missing)

        unknown = metadata.to_dict()
        unknown["note"] = "invented"
        with self.assertRaisesRegex(ValueError, "unknown keys: note"):
            RunMetadata.from_dict(unknown)

    def test_validates_run_identity_timestamps_devices_and_counts(self) -> None:
        cases = (
            ({"git_revision": "main"}, "git_revision"),
            ({"config_digest": "0"}, "config_digest"),
            ({"dataset_fingerprint": "0"}, "dataset_fingerprint"),
            ({"initial_state_fingerprint": "0"}, "initial_state_fingerprint"),
            ({"started_at": "2026-08-10"}, "started_at"),
            (
                {"recorded_at": "2026-08-10T07:59:59Z"},
                "cannot precede",
            ),
            ({"seed": True}, "seed must be an integer"),
            ({"variant": "other"}, "variant must"),
            ({"requested_device": "cuda:0"}, "requested_device"),
            ({"resolved_device": "gpu"}, "resolved_device"),
            (
                {"requested_device": "cuda", "resolved_device": "cpu"},
                "must satisfy",
            ),
            ({"counts": {}}, "counts cannot be empty"),
            ({"counts": {"train_examples": -1}}, "non-negative"),
            ({"drop_stats": {"Bad-Key": 1}}, "invalid key"),
            ({"package_versions": {}}, "package_versions cannot be empty"),
            ({"hardware": {}}, "hardware cannot be empty"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    _metadata(**override)

        with self.assertRaisesRegex(ValueError, "missing continuation"):
            _metadata(labels=("O", "B-V"))

    def test_checks_training_config_and_exact_metadata_compatibility(self) -> None:
        config = _config()
        metadata = _metadata(config=config)
        metadata.assert_config_compatible(config)
        require_exact_run_metadata(metadata, metadata)

        wrong_seed = replace(metadata, seed=99)
        with self.assertRaisesRegex(ValueError, "seed"):
            wrong_seed.assert_config_compatible(config)

        wrong_dataset = replace(metadata, dataset_fingerprint="f" * 64)
        with self.assertRaisesRegex(ValueError, "dataset_fingerprint"):
            wrong_dataset.assert_config_compatible(config)

        changed = replace(metadata, counts=(("train_examples", 5),))
        with self.assertRaisesRegex(ValueError, "counts"):
            require_exact_run_metadata(changed, metadata)


class CheckpointBundleTests(unittest.TestCase):
    def test_saves_exact_layout_validates_then_loads_through_callbacks(self) -> None:
        config = _config()
        metadata = _metadata(config=config)
        labels = CheckpointLabelConfig.from_labels(LABELS)
        state = {"classifier.weight": [1, 2, 3]}
        loaded_paths = []

        def save_state_dict(value, path):
            path.write_bytes(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            )

        def load_state_dict(path):
            loaded_paths.append(path)
            return json.loads(path.read_text(encoding="utf-8"))

        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "checkpoint"
            saved = save_checkpoint_bundle(
                destination,
                metadata=metadata,
                config=config,
                label_config=labels,
                state_dict=state,
                save_state_dict=save_state_dict,
            )
            self.assertEqual(saved, destination)
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                {METADATA_FILENAME, LABEL_CONFIG_FILENAME, STATE_DICT_FILENAME},
            )
            checkpoint_metadata = json.loads(
                (destination / METADATA_FILENAME).read_text(encoding="utf-8")
            )
            self.assertNotIn("training_config", checkpoint_metadata)
            self.assertEqual(
                checkpoint_metadata["run_metadata"]["config_digest"],
                config.digest,
            )

            validated = validate_checkpoint_bundle(
                destination,
                expected_metadata=metadata,
                expected_config=config,
                expected_label_config=labels,
            )
            self.assertEqual(validated.metadata, metadata)
            self.assertEqual(validated.label_config, labels)
            self.assertEqual(validated.state_dict_path.name, STATE_DICT_FILENAME)
            self.assertEqual(
                validated.state_dict_sha256,
                hashlib.sha256(validated.state_dict_path.read_bytes()).hexdigest(),
            )

            loaded = load_checkpoint_bundle(
                destination,
                expected_metadata=metadata,
                expected_config=config,
                expected_label_config=labels,
                load_state_dict=load_state_dict,
            )

        self.assertEqual(loaded.state_dict, state)
        self.assertEqual(len(loaded_paths), 1)

    def test_rejects_metadata_or_labels_before_state_callback(self) -> None:
        config = _config()
        metadata = _metadata(config=config)
        labels = CheckpointLabelConfig.from_labels(LABELS)
        callback_calls = []

        def save_state_dict(value, path):
            path.write_text(repr(value), encoding="utf-8")

        def load_state_dict(path):
            callback_calls.append(path)
            return path.read_text(encoding="utf-8")

        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "checkpoint"
            save_checkpoint_bundle(
                destination,
                metadata=metadata,
                config=config,
                label_config=labels,
                state_dict={"weight": 1},
                save_state_dict=save_state_dict,
            )

            changed_config = _config(batch_size="16")
            with self.assertRaisesRegex(ValueError, "config_digest"):
                load_checkpoint_bundle(
                    destination,
                    expected_metadata=metadata,
                    expected_config=changed_config,
                    expected_label_config=labels,
                    load_state_dict=load_state_dict,
                )

            changed_metadata = replace(
                metadata, recorded_at="2026-08-10T09:31:00Z"
            )
            with self.assertRaisesRegex(ValueError, "recorded_at"):
                load_checkpoint_bundle(
                    destination,
                    expected_metadata=changed_metadata,
                    expected_config=config,
                    expected_label_config=labels,
                    load_state_dict=load_state_dict,
                )

            changed_labels = CheckpointLabelConfig.from_labels(
                ("O", "B-ARG1", "B-V", "I-ARG1", "I-V")
            )
            with self.assertRaisesRegex(ValueError, "label configuration"):
                load_checkpoint_bundle(
                    destination,
                    expected_metadata=metadata,
                    expected_config=config,
                    expected_label_config=changed_labels,
                    load_state_dict=load_state_dict,
                )

        self.assertEqual(callback_calls, [])

    def test_rejects_tampered_or_noncanonical_bundle_layout(self) -> None:
        config = _config()
        metadata = _metadata(config=config)
        labels = CheckpointLabelConfig.from_labels(LABELS)

        def save_state_dict(value, path):
            path.write_text(repr(value), encoding="utf-8")

        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "checkpoint"
            save_checkpoint_bundle(
                destination,
                metadata=metadata,
                config=config,
                label_config=labels,
                state_dict={"weight": 1},
                save_state_dict=save_state_dict,
            )
            (destination / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown files: extra.txt"):
                validate_checkpoint_bundle(
                    destination,
                    expected_metadata=metadata,
                    expected_config=config,
                    expected_label_config=labels,
                )
            (destination / "extra.txt").unlink()

            state_path = destination / STATE_DICT_FILENAME
            original_state = state_path.read_bytes()
            state_path.write_bytes(original_state + b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_checkpoint_bundle(
                    destination,
                    expected_metadata=metadata,
                    expected_config=config,
                    expected_label_config=labels,
                )
            state_path.write_bytes(original_state)

            metadata_path = destination / METADATA_FILENAME
            metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "not in canonical form"):
                validate_checkpoint_bundle(
                    destination,
                    expected_metadata=metadata,
                    expected_config=config,
                    expected_label_config=labels,
                )

    def test_save_is_atomic_fail_closed_and_never_overwrites(self) -> None:
        config = _config()
        metadata = _metadata(config=config)
        labels = CheckpointLabelConfig.from_labels(LABELS)

        def omit_state_dict(value, path):
            del value, path

        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "checkpoint"
            with self.assertRaisesRegex(ValueError, "state_dict.pt"):
                save_checkpoint_bundle(
                    destination,
                    metadata=metadata,
                    config=config,
                    label_config=labels,
                    state_dict={},
                    save_state_dict=omit_state_dict,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

            destination.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                save_checkpoint_bundle(
                    destination,
                    metadata=metadata,
                    config=config,
                    label_config=labels,
                    state_dict={},
                    save_state_dict=omit_state_dict,
                )

        mismatched_labels = CheckpointLabelConfig.from_labels(
            ("O", "B-ARG1", "B-V", "I-ARG1", "I-V")
        )
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "do not match"):
                save_checkpoint_bundle(
                    Path(temporary) / "checkpoint",
                    metadata=metadata,
                    config=config,
                    label_config=mismatched_labels,
                    state_dict={},
                    save_state_dict=omit_state_dict,
                )


if __name__ == "__main__":
    unittest.main()
