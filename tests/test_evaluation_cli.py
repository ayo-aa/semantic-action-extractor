from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from semantic_action_extractor.datasets.common import DatasetFormatError
from semantic_action_extractor.evaluation.bundle import (
    EVALUATION_BUNDLE_VERSION,
    EvaluationBundle,
    load_evaluation_bundle,
)
from semantic_action_extractor.evaluation.scorers import PRIMARY_END_TO_END_V1
from semantic_action_extractor.evaluation.types import (
    EvaluationArgument,
    EvaluationCorpus,
    EvaluationMentionQualifier,
    EvaluationPredicate,
    EvaluationQAPair,
    EvaluationQuestion,
    PredicateKey,
)
from semantic_action_extractor.evaluation_cli import main


def _bundle(
    *,
    predicate_source: str,
    source_id: str = "sentence-1",
    metadata: dict[str, object] | None = None,
) -> EvaluationBundle:
    question = EvaluationQuestion(
        surface_form="Who approved something?",
        wh="who",
        aux="_",
        subj="_",
        verb="past",
        obj="something",
        prep="_",
        obj2="_",
        is_passive=False,
        is_negated=False,
    )
    pair = EvaluationQAPair(
        pair_id="pair-1",
        role_id="role-1",
        question=question,
        argument=EvaluationArgument(
            token_spans=((0, 1),),
            character_spans=((0, 4),),
        ),
    )
    predicate = EvaluationPredicate(
        key=PredicateKey(source_id, 1, 2, "verbal"),
        is_eventive=True,
        lemma="approve",
        pairs=(pair,),
        mention_qualifiers=(
            EvaluationMentionQualifier(
                kind="reported",
                evidence=EvaluationArgument(
                    token_spans=((2, 3),),
                    character_spans=((5, 13),),
                ),
            ),
        ),
    )
    return EvaluationBundle(
        corpus=EvaluationCorpus((predicate,)),
        predicate_source=predicate_source,
        consolidation_rule="valid-judgment-union-v1",
        metadata={"fixture": True} if metadata is None else metadata,
    )


class EvaluationCliTests(unittest.TestCase):
    def test_loads_legacy_bundle_as_unassessed_for_qualifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            payload = _bundle(predicate_source="legacy-model").to_dict()
            payload["bundle_version"] = "1.0.0"
            payload["corpus"]["predicates"][0].pop("mention_qualifiers")
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_evaluation_bundle(path)

            self.assertEqual(loaded.bundle_version, EVALUATION_BUNDLE_VERSION)
            self.assertIsNone(loaded.corpus.predicates[0].mention_qualifiers)

    def test_bundle_round_trip_and_cli_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            predicted_path = root / "predicted.json"
            output_path = root / "score.json"
            _bundle(predicate_source="gold-annotations").write(gold_path)
            _bundle(predicate_source="fixture-model").write(predicted_path)

            loaded = load_evaluation_bundle(predicted_path)
            self.assertEqual(loaded.to_dict(), _bundle(predicate_source="fixture-model").to_dict())

            captured = io.StringIO()
            with redirect_stdout(captured):
                status = main(
                    [
                        str(gold_path),
                        str(predicted_path),
                        "--mode",
                        PRIMARY_END_TO_END_V1,
                        "--output",
                        str(output_path),
                    ]
                )

            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(status, 0)
            self.assertEqual(report["result"]["labeled_arguments"]["f1"], 1.0)
            self.assertEqual(
                report["result"]["mention_qualifier_exact_evidence"]["f1"],
                1.0,
            )
            self.assertEqual(
                report["result"]["settings"]["predicate_source"],
                "fixture-model",
            )
            self.assertEqual(json.loads(captured.getvalue())["scorer"], PRIMARY_END_TO_END_V1)

    def test_score_output_cannot_alias_either_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            predicted_path = root / "predicted.json"
            _bundle(predicate_source="gold-annotations").write(gold_path)
            _bundle(predicate_source="fixture-model").write(predicted_path)
            original_gold = gold_path.read_bytes()
            original_predicted = predicted_path.read_bytes()

            for output_path in (gold_path, predicted_path):
                with self.subTest(output=output_path.name):
                    with self.assertRaisesRegex(
                        ValueError,
                        "score output must be different",
                    ):
                        main(
                            [
                                str(gold_path),
                                str(predicted_path),
                                "--mode",
                                PRIMARY_END_TO_END_V1,
                                "--output",
                                str(output_path),
                                "--overwrite",
                            ]
                        )
                    self.assertEqual(gold_path.read_bytes(), original_gold)
                    self.assertEqual(predicted_path.read_bytes(), original_predicted)

    def test_rejects_predictions_from_gold_quarantined_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            predicted_path = root / "predicted.json"
            _bundle(
                predicate_source="pilot-gold",
                metadata={"excluded_source_ids": ["quarantined-source"]},
            ).write(gold_path)
            _bundle(
                predicate_source="fixture-model",
                source_id="quarantined-source",
            ).write(predicted_path)

            with self.assertRaisesRegex(
                DatasetFormatError,
                "prediction bundle contains quarantined sources",
            ):
                main(
                    [
                        str(gold_path),
                        str(predicted_path),
                        "--mode",
                        PRIMARY_END_TO_END_V1,
                    ]
                )

    def test_pilot_scoring_requires_matching_source_and_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            predicted_path = root / "predicted.json"
            _bundle(
                predicate_source="pilot-gold",
                metadata={
                    "contract_version": "candidate-pilot-normalized-v1",
                    "authoring_fingerprint": "pilot-source-fingerprint",
                    "tokenization_version": "pilot-source-tokenizer-v1",
                    "excluded_source_ids": [],
                },
            ).write(gold_path)
            _bundle(
                predicate_source="fixture-model",
                metadata={
                    "authoring_fingerprint": "different-source-fingerprint",
                    "tokenization_version": "pilot-source-tokenizer-v1",
                },
            ).write(predicted_path)

            with self.assertRaisesRegex(
                DatasetFormatError,
                "authoring_fingerprint does not match",
            ):
                main(
                    [
                        str(gold_path),
                        str(predicted_path),
                        "--mode",
                        PRIMARY_END_TO_END_V1,
                    ]
                )


if __name__ == "__main__":
    unittest.main()
