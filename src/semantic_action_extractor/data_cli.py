"""Command-line data preparation without changing the extraction CLI."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from .datasets.io import (
    download_verified,
    extract_verified,
    sha256_file,
    verify_sha256,
)
from .datasets.manifest import ManifestBuilder, SourceArtifactIdentity
from .datasets.leakage import (
    DatasetPartition,
    TrainingQuarantineReport,
    build_training_quarantine,
    iter_after_document_quarantine,
    load_training_quarantine_report,
)
from .datasets.prepare import write_adapted_jsonl
from .datasets.qanom import QANOM_ADAPTER_VERSION, QANOM_RELEASE, iter_qanom_csv
from .datasets.qasrl import (
    QASRL_ADAPTER_VERSION,
    QASRL_BANK_RELEASE,
    QASRL_GOLD_RELEASE,
    iter_qasrl_records,
    load_qasrl_index,
)
from .datasets.registry import ARTIFACTS, get_artifact
from .datasets.serialization import (
    iter_adapted_jsonl,
    load_preparation_manifest,
)
from .evaluation.bundle import EvaluationBundle
from .evaluation.consolidation import consolidate_annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-action-data",
        description="Verify and adapt the pinned QA-SRL and QANom releases.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="verify a pinned archive checksum")
    verify.add_argument("artifact", choices=sorted(ARTIFACTS))
    verify.add_argument("archive", type=Path)

    fetch = commands.add_parser("fetch", help="download and verify a pinned archive")
    fetch.add_argument("artifact", choices=sorted(ARTIFACTS))
    fetch.add_argument("output", type=Path)
    fetch.add_argument("--overwrite", action="store_true")

    extract = commands.add_parser(
        "extract", help="verify and safely extract a pinned archive"
    )
    extract.add_argument("artifact", choices=sorted(ARTIFACTS))
    extract.add_argument("archive", type=Path)
    extract.add_argument("output", type=Path)

    qasrl = commands.add_parser(
        "adapt-qasrl", help="adapt one QA-SRL JSONL split"
    )
    qasrl.add_argument("input", type=Path)
    qasrl.add_argument("output", type=Path)
    qasrl.add_argument("--manifest", type=Path)
    qasrl.add_argument("--release", required=True)
    qasrl.add_argument("--split", required=True)
    qasrl.add_argument(
        "--layer", choices=("orig", "expanded", "dense", "gold"), required=True
    )
    qasrl.add_argument("--index", type=Path)
    qasrl.add_argument("--quarantine-report", type=Path)
    qasrl.add_argument("--overwrite", action="store_true")

    qanom = commands.add_parser("adapt-qanom", help="adapt one QANom CSV split")
    qanom.add_argument("input", type=Path)
    qanom.add_argument("output", type=Path)
    qanom.add_argument("--manifest", type=Path)
    qanom.add_argument("--release", default=QANOM_RELEASE)
    qanom.add_argument("--split", choices=("train", "dev", "test"), required=True)
    qanom.add_argument("--quarantine-report", type=Path)
    qanom.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser(
        "validate-adapted",
        help="validate canonical annotation JSONL against its manifest",
    )
    validate.add_argument("input", type=Path)
    validate.add_argument("--manifest", type=Path, required=True)

    consolidate = commands.add_parser(
        "consolidate",
        help="convert canonical annotations into a scorer-ready gold bundle",
    )
    consolidate.add_argument("input", type=Path)
    consolidate.add_argument("output", type=Path)
    consolidate.add_argument("--manifest", type=Path, required=True)
    consolidate.add_argument("--overwrite", action="store_true")

    leakage = commands.add_parser(
        "check-joint-splits",
        help="build the fixed QA-SRL/QANom training quarantine report",
    )
    leakage.add_argument("--qasrl-train", type=Path, required=True)
    leakage.add_argument("--qasrl-index", type=Path, required=True)
    leakage.add_argument("--qasrl-gold-dev", type=Path, required=True)
    leakage.add_argument("--qasrl-gold-test", type=Path, required=True)
    leakage.add_argument("--qanom-train", type=Path, required=True)
    leakage.add_argument("--qanom-dev", type=Path, required=True)
    leakage.add_argument("--qanom-test", type=Path, required=True)
    leakage.add_argument("--output", type=Path, required=True)
    leakage.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        artifact = get_artifact(args.artifact)
        actual = verify_sha256(args.archive, artifact.sha256)
        _print_json({"artifact": artifact.key, "sha256": actual, "verified": True})
        return 0
    if args.command == "fetch":
        artifact = get_artifact(args.artifact)
        path = download_verified(artifact, args.output, overwrite=args.overwrite)
        _print_json(
            {"artifact": artifact.key, "path": str(path), "sha256": artifact.sha256}
        )
        return 0
    if args.command == "extract":
        artifact = get_artifact(args.artifact)
        path = extract_verified(artifact, args.archive, args.output)
        _print_json({"artifact": artifact.key, "output": str(path)})
        return 0
    if args.command == "adapt-qasrl":
        quarantine = _load_quarantine(
            args.quarantine_report,
            split=args.split,
            source=args.input,
            source_label="qasrl_train",
        )
        document_index = load_qasrl_index(args.index) if args.index else None
        layer = None if args.layer == "gold" else args.layer
        records = iter_qasrl_records(
            args.input,
            release=args.release,
            split=args.split,
            layer=layer,
            document_index=document_index,
        )
        manifest = _write(
            records=records,
            output=args.output,
            manifest=args.manifest,
            dataset="qa-srl",
            release=args.release,
            split=args.split,
            adapter="qa-srl-jsonl",
            adapter_version=QASRL_ADAPTER_VERSION,
            source=args.input,
            quarantine=quarantine,
            quarantine_path=args.quarantine_report,
            additional_sources=(args.index,) if args.index is not None else (),
            overwrite=args.overwrite,
        )
        _print_json(manifest.to_dict())
        return 0
    if args.command == "adapt-qanom":
        quarantine = _load_quarantine(
            args.quarantine_report,
            split=args.split,
            source=args.input,
            source_label="qanom_train",
        )
        records = iter_qanom_csv(
            args.input,
            split=args.split,
            release=args.release,
        )
        manifest = _write(
            records=records,
            output=args.output,
            manifest=args.manifest,
            dataset="qanom",
            release=args.release,
            split=args.split,
            adapter="qanom-csv",
            adapter_version=QANOM_ADAPTER_VERSION,
            source=args.input,
            quarantine=quarantine,
            quarantine_path=args.quarantine_report,
            overwrite=args.overwrite,
        )
        _print_json(manifest.to_dict())
        return 0
    if args.command == "validate-adapted":
        manifest = load_preparation_manifest(args.manifest)
        records = sum(
            1
            for _ in iter_adapted_jsonl(
                args.input,
                manifest_path=args.manifest,
            )
        )
        _print_json(
            {
                "valid": True,
                "records": records,
                "dataset": manifest.dataset,
                "release": manifest.release,
                "split": manifest.split,
                "record_fingerprint": manifest.record_fingerprint,
            }
        )
        return 0
    if args.command == "consolidate":
        manifest = load_preparation_manifest(args.manifest)
        result = consolidate_annotations(
            iter_adapted_jsonl(
                args.input,
                manifest_path=args.manifest,
            )
        )
        bundle = EvaluationBundle(
            corpus=result.corpus,
            predicate_source="consolidated-gold-annotations",
            consolidation_rule=result.rule,
            metadata={
                "source_manifest": manifest.to_dict(),
                "consolidation_counts": dict(result.counts),
            },
        )
        bundle.write(args.output, overwrite=args.overwrite)
        _print_json(
            {
                "output": str(args.output),
                "predicate_count": len(result.corpus.predicates),
                "consolidation_rule": result.rule,
                "counts": dict(result.counts),
            }
        )
        return 0
    if args.command == "check-joint-splits":
        qasrl_index = load_qasrl_index(args.qasrl_index)
        report = build_training_quarantine(
            (
                DatasetPartition(
                    "qasrl-bank-expanded-train",
                    "train",
                    iter_qasrl_records(
                        args.qasrl_train,
                        release=QASRL_BANK_RELEASE,
                        split="train",
                        layer="expanded",
                        document_index=qasrl_index,
                    ),
                ),
                DatasetPartition(
                    "qanom-train",
                    "train",
                    iter_qanom_csv(
                        args.qanom_train,
                        split="train",
                        release=QANOM_RELEASE,
                    ),
                ),
            ),
            (
                DatasetPartition(
                    "qasrl-gold-dev",
                    "development",
                    iter_qasrl_records(
                        args.qasrl_gold_dev,
                        release=QASRL_GOLD_RELEASE,
                        split="dev",
                    ),
                ),
                DatasetPartition(
                    "qanom-dev",
                    "development",
                    iter_qanom_csv(
                        args.qanom_dev,
                        split="dev",
                        release=QANOM_RELEASE,
                    ),
                ),
                DatasetPartition(
                    "qasrl-gold-test",
                    "test",
                    iter_qasrl_records(
                        args.qasrl_gold_test,
                        release=QASRL_GOLD_RELEASE,
                        split="test",
                    ),
                ),
                DatasetPartition(
                    "qanom-test",
                    "test",
                    iter_qanom_csv(
                        args.qanom_test,
                        split="test",
                        release=QANOM_RELEASE,
                    ),
                ),
            ),
        )
        report = replace(
            report,
            source_artifacts={
                label: sha256_file(path)
                for label, path in (
                    ("qasrl_train", args.qasrl_train),
                    ("qasrl_index", args.qasrl_index),
                    ("qasrl_gold_dev", args.qasrl_gold_dev),
                    ("qasrl_gold_test", args.qasrl_gold_test),
                    ("qanom_train", args.qanom_train),
                    ("qanom_dev", args.qanom_dev),
                    ("qanom_test", args.qanom_test),
                )
            },
        )
        report.write(args.output, overwrite=args.overwrite)
        _print_json(
            {
                "output": str(args.output),
                "policy": report.policy,
                "contaminated_document_count": len(
                    report.contaminated_document_ids
                ),
                "trigger_count": len(report.triggers),
                "training_record_counts": dict(report.training_record_counts),
                "quarantined_record_counts": dict(
                    report.quarantined_record_counts
                ),
            }
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _write(
    *,
    records,
    output: Path,
    manifest: Path | None,
    dataset: str,
    release: str,
    split: str,
    adapter: str,
    adapter_version: str,
    source: Path,
    overwrite: bool,
    quarantine: TrainingQuarantineReport | None = None,
    quarantine_path: Path | None = None,
    additional_sources: tuple[Path, ...] = (),
):
    manifest_path = manifest or output.with_name(f"{output.name}.manifest.json")
    tracked_sources = (source,) + additional_sources
    if quarantine_path is not None:
        tracked_sources += (quarantine_path,)
    source_digests = {
        path: sha256_file(path)
        for path in tracked_sources
    }
    source_artifacts = [
        SourceArtifactIdentity(name=path.name, sha256=digest)
        for path, digest in source_digests.items()
    ]
    builder = ManifestBuilder(
        dataset=dataset,
        release=release,
        split=split,
        adapter=adapter,
        adapter_version=adapter_version,
        source_artifacts=tuple(source_artifacts),
    )

    prepared_records = records
    if quarantine is not None:
        prepared_records = iter_after_document_quarantine(
            prepared_records,
            quarantine,
            on_exclude=lambda _: builder.add_exclusion(quarantine.policy),
        )

    def records_from_unchanged_source():
        yield from prepared_records
        for tracked_source, initial_digest in source_digests.items():
            final_digest = sha256_file(tracked_source)
            if final_digest != initial_digest:
                raise RuntimeError(
                    "source file changed while it was being adapted: "
                    f"{tracked_source}"
                )

    return write_adapted_jsonl(
        records_from_unchanged_source(),
        output,
        manifest_path,
        builder,
        overwrite=overwrite,
    )


def _load_quarantine(
    path: Path | None,
    *,
    split: str,
    source: Path,
    source_label: str,
) -> TrainingQuarantineReport | None:
    if path is None:
        return None
    if split != "train":
        raise ValueError("a quarantine report can be applied only to training data")
    report = load_training_quarantine_report(path)
    expected = report.source_artifacts.get(source_label)
    actual = sha256_file(source)
    if expected is None:
        raise ValueError(
            f"quarantine report does not identify the {source_label} source"
        )
    if actual != expected:
        raise ValueError(
            f"training source does not match the quarantine report: {source}"
        )
    return report


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
