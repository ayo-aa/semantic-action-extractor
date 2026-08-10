"""Command-line scoring for reusable evaluation bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import uuid

from .challenge_workbook import PILOT_WORKBOOK_CONTRACT_VERSION
from .datasets.common import DatasetFormatError
from .datasets.io import sha256_file
from .evaluation.bundle import EvaluationBundle, load_evaluation_bundle
from .evaluation.scorers import SCORER_MODES, score_corpora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-action-evaluate",
        description="Score a prediction bundle against a gold evaluation bundle.",
    )
    parser.add_argument("gold", type=Path)
    parser.add_argument("predicted", type=Path)
    parser.add_argument("--mode", choices=sorted(SCORER_MODES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None:
        _require_distinct_output(args.output, (args.gold, args.predicted))
    gold, gold_sha256 = _load_stable_bundle(args.gold)
    predicted, predicted_sha256 = _load_stable_bundle(args.predicted)
    _validate_pilot_compatibility(gold, predicted)
    _validate_record_quarantine(gold, predicted)
    result = score_corpora(
        gold.corpus,
        predicted.corpus,
        mode=args.mode,
        predicate_source=predicted.predicate_source,
        consolidation_rule=gold.consolidation_rule,
    )
    payload = {
        "gold_bundle": {
            "path": str(args.gold),
            "sha256": gold_sha256,
            "predicate_source": gold.predicate_source,
            "consolidation_rule": gold.consolidation_rule,
        },
        "predicted_bundle": {
            "path": str(args.predicted),
            "sha256": predicted_sha256,
            "predicate_source": predicted.predicate_source,
            "consolidation_rule": predicted.consolidation_rule,
        },
        "result": result.to_dict(),
    }
    if args.output is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _write_json(payload, args.output, overwrite=args.overwrite)
        print(
            json.dumps(
                {"output": str(args.output), "scorer": result.scorer},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    return 0


def _load_stable_bundle(path: Path) -> tuple[EvaluationBundle, str]:
    initial_digest = sha256_file(path)
    bundle = load_evaluation_bundle(path)
    final_digest = sha256_file(path)
    if final_digest != initial_digest:
        raise RuntimeError(f"evaluation bundle changed while it was read: {path}")
    return bundle, initial_digest


def _validate_record_quarantine(
    gold: EvaluationBundle,
    predicted: EvaluationBundle,
) -> None:
    """Refuse to score source records that gold explicitly quarantined."""

    raw_source_ids = gold.metadata.get("excluded_source_ids")
    if raw_source_ids is None:
        return
    if not isinstance(raw_source_ids, list) or any(
        not isinstance(source_id, str) or not source_id.strip()
        for source_id in raw_source_ids
    ):
        raise DatasetFormatError(
            "gold metadata excluded_source_ids must be a list of source IDs"
        )
    if len(raw_source_ids) != len(set(raw_source_ids)):
        raise DatasetFormatError(
            "gold metadata excluded_source_ids must not contain duplicates"
        )
    excluded = set(raw_source_ids)
    gold_leaks = sorted(
        {predicate.key.source_id for predicate in gold.corpus.predicates} & excluded
    )
    if gold_leaks:
        raise DatasetFormatError(
            "gold bundle contains predicates from quarantined sources: "
            f"{gold_leaks}"
        )
    predicted_leaks = sorted(
        {
            predicate.key.source_id
            for predicate in predicted.corpus.predicates
        }
        & excluded
    )
    if predicted_leaks:
        raise DatasetFormatError(
            "prediction bundle contains quarantined sources; remove them before "
            f"scoring: {predicted_leaks}"
        )


def _validate_pilot_compatibility(
    gold: EvaluationBundle,
    predicted: EvaluationBundle,
) -> None:
    if gold.metadata.get("contract_version") != PILOT_WORKBOOK_CONTRACT_VERSION:
        return
    for field in ("authoring_fingerprint", "tokenization_version"):
        gold_value = gold.metadata.get(field)
        if not isinstance(gold_value, str) or not gold_value:
            raise DatasetFormatError(
                f"pilot gold metadata requires a non-empty {field}"
            )
        if predicted.metadata.get(field) != gold_value:
            raise DatasetFormatError(
                f"prediction bundle {field} does not match pilot gold"
            )


def _require_distinct_output(output: Path, inputs: Sequence[Path]) -> None:
    resolved_output = output.resolve()
    for input_path in inputs:
        try:
            same_file = output.exists() and input_path.samefile(output)
        except OSError:
            same_file = False
        if resolved_output == input_path.resolve() or same_file:
            raise ValueError(
                "score output must be different from the gold and predicted bundles"
            )


def _write_json(payload: object, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace {path}")
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"score output is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        partial.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        partial.replace(path)
    except Exception:
        if partial.exists() and partial.is_file():
            partial.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
