# Data directory

No dataset is included in Git.

`data/raw/` holds local archive downloads, `data/extracted/` holds their verified extraction, and `data/processed/` holds canonical annotation JSONL, preparation manifests, the joint-training quarantine report, and scorer-ready evaluation bundles. These paths are local-only and ignored by Git; a processed file can retain the source dataset’s redistribution restrictions.

The data CLI recognizes three pinned artifact keys:

- `qa-srl-bank-2.1`
- `qa-srl-gold-standard`
- `qanom-2020`

The reproducible path is fetch, checksum verification, safe extraction, joint-split inspection, strict adaptation, manifest validation, and named judgment consolidation. By default, adaptation writes `<output>.manifest.json` beside the canonical JSONL. That manifest records dataset and split identity, source hashes, adapter and schema versions, record fingerprint, record and annotation counts, anomaly counts, and intentional exclusions.

[DATA_USAGE.md](../DATA_USAGE.md) defines the publication and split boundaries. The [QA-SRL card](../docs/datasets/qa_srl.md) and [QANom card](../docs/datasets/qanom.md) record release-specific fields, counts, anomalies, rights, and scorer behavior. Complete commands appear in the final “How to run the project” section of [README.md](../README.md).
