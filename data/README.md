# Data directory

No dataset is included.

Raw and processed data paths are ignored by Git. See `DATA_USAGE.md` before preparing, evaluating, or distributing any dataset.

The repository includes a source-neutral PropBank/PTB conversion core, but no MASC archive adapter or preparation command. Universal Proposition Bank 1.0 English EWT was rejected for the restored BIO-span objective because it supplies dependency heads rather than gold spans. MASC PropBank is the next candidate, but it remains blocked behind the feasibility and rights gate in `docs/datasets/masc_propbank_gate.md`; do not treat it as an adopted dependency or a completed dataset stage.
