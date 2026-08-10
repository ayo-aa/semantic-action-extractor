import unittest
from types import SimpleNamespace
from unittest.mock import patch

from semantic_action_extractor.srl.model import (
    OptionalMLDependencyError,
    build_predicate_conditioned_bert,
)


class FakeTensor:
    def __init__(self, name):
        self.name = name
        self.reshape_calls = []

    def reshape(self, *shape):
        self.reshape_calls.append(shape)
        return self


class FakeModule:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FakeLinear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.inputs = []

    def __call__(self, hidden_states):
        self.inputs.append(hidden_states)
        return FakeTensor("logits")


class FakeLoss:
    def __init__(self, ignore_index):
        self.ignore_index = ignore_index
        self.calls = []

    def __call__(self, logits, labels):
        self.calls.append((logits, labels))
        return "loss"


class FakeEncoder(FakeModule):
    def __init__(self, *, type_vocab_size=2):
        self.config = SimpleNamespace(
            hidden_size=768,
            type_vocab_size=type_vocab_size,
        )
        self.hidden_states = FakeTensor("hidden")
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(last_hidden_state=self.hidden_states)


FAKE_TORCH = SimpleNamespace(
    nn=SimpleNamespace(
        Module=FakeModule,
        Linear=FakeLinear,
        CrossEntropyLoss=FakeLoss,
    )
)


class ModelTests(unittest.TestCase):
    def test_builds_linear_head_and_uses_predicate_indicator(self) -> None:
        encoder = FakeEncoder()
        model = build_predicate_conditioned_bert(
            "bert-base-uncased", 53, encoder=encoder, torch_module=FAKE_TORCH
        )
        input_ids = FakeTensor("ids")
        attention_mask = FakeTensor("mask")
        predicate_indicator = FakeTensor("predicate")
        labels = FakeTensor("labels")

        output = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=predicate_indicator,
            labels=labels,
        )

        self.assertEqual(model.classifier.in_features, 768)
        self.assertEqual(model.classifier.out_features, 53)
        self.assertIs(encoder.calls[0]["token_type_ids"], predicate_indicator)
        self.assertIs(model.classifier.inputs[0], encoder.hidden_states)
        self.assertEqual(model.loss_function.ignore_index, -100)
        self.assertIsInstance(output, dict)
        self.assertEqual(output["loss"], "loss")
        self.assertEqual(output.loss, "loss")
        self.assertEqual(output.logits.reshape_calls, [(-1, 53)])
        self.assertEqual(labels.reshape_calls, [(-1,)])

    def test_imports_optional_libraries_only_when_building(self) -> None:
        with patch(
            "semantic_action_extractor.srl.model.importlib.import_module",
            side_effect=ImportError("not installed"),
        ) as import_module:
            with self.assertRaises(OptionalMLDependencyError):
                build_predicate_conditioned_bert("bert-base-uncased", 2)

        import_module.assert_called_once_with("torch")

    def test_pins_model_revision_and_requires_explicit_predicate_input(self) -> None:
        encoder = FakeEncoder()

        class FakeAutoModel:
            calls = []

            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                cls.calls.append((model_name, kwargs))
                return encoder

        transformers = SimpleNamespace(AutoModel=FakeAutoModel)
        model = build_predicate_conditioned_bert(
            "bert-base-uncased",
            3,
            model_revision="refs/pr/1",
            torch_module=FAKE_TORCH,
            transformers_module=transformers,
        )

        self.assertEqual(
            FakeAutoModel.calls,
            [("bert-base-uncased", {"revision": "refs/pr/1"})],
        )
        self.assertEqual(model.base_model_revision, "refs/pr/1")
        with self.assertRaisesRegex(ValueError, "token_type_ids"):
            model(FakeTensor("ids"))

    def test_rejects_encoder_without_two_token_type_embeddings(self) -> None:
        with self.assertRaisesRegex(ValueError, "two token-type embeddings"):
            build_predicate_conditioned_bert(
                "bert-base-uncased",
                3,
                encoder=FakeEncoder(type_vocab_size=1),
                torch_module=FAKE_TORCH,
            )

    def test_validates_configuration_before_loading_dependencies(self) -> None:
        with patch(
            "semantic_action_extractor.srl.model.importlib.import_module"
        ) as import_module:
            with self.assertRaises(ValueError):
                build_predicate_conditioned_bert("", 53)
            with self.assertRaises(ValueError):
                build_predicate_conditioned_bert("bert-base-uncased", 0)
            with self.assertRaises(TypeError):
                build_predicate_conditioned_bert("bert-base-uncased", True)
            with self.assertRaises(ValueError):
                build_predicate_conditioned_bert(
                    "bert-base-uncased", 53, model_revision=""
                )
            with self.assertRaises(TypeError):
                build_predicate_conditioned_bert(
                    "bert-base-uncased", 53, model_revision=True
                )

        import_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
