"""Command-line scoring for reusable evaluation bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import uuid

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
    gold, gold_sha256 = _load_stable_bundle(args.gold)
    predicted, predicted_sha256 = _load_stable_bundle(args.predicted)
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
