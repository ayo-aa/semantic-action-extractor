# Data directory

No dataset is included.

Raw and processed data paths are ignored by Git. See `DATA_USAGE.md` before preparing, evaluating, or distributing any dataset.

The repository includes a source-neutral PropBank/PTB conversion core and a read-only aggregate MASC audit, but no corpus preparation adapter. Universal Proposition Bank 1.0 English EWT and MASC PropBank were both rejected for the restored BIO-span objective. A locally acquired MASC ZIP may exist under ignored `data/raw/` as audit evidence only; it is not included, adopted, written out as member payloads, or used for training. Replacement-source selection is pending. See the [gate record](../docs/datasets/masc_propbank_gate.md) and [completed audit](../reports/masc_propbank_audit.md).
