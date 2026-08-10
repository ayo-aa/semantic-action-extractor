# Data directory

No dataset is included. Raw archives and prepared outputs are ignored by Git.

- `data/raw/Propbank-original-format.zip`, if present, is rejected MASC audit
  evidence only. It is not prepared or used for training.
- `data/raw/BabySRL.zip`, if present, is the pinned read-only BabySRL audit
  artifact. The aggregate gate passes at 18,397 / 18,536 = 99.2501%, but the
  archive is not approved for preparation until access confirmation is
  recorded.
- No prepared split directory exists. The safe in-memory eligibility counts are
  13,713 train, 1,356 development, and 1,274 test.

The frozen 133-document assignment is tracked as
[`babysrl_split_manifest.json`](../docs/datasets/babysrl_split_manifest.json),
whose canonical SHA-256 is
`73ecae9f81d1d2b9f8495b13b297da9c3d24e387630e71a38ef2420d4c9a5de7`.
That manifest contains document assignments, not corpus content.

TalkBank/CHILDES registration and acceptance of the current
[ground rules](https://talkbank.org/0share/rules.html) are required before
writing provisional prepared BabySRL data. The prepared/raw pair must then pass
the authorized private manual review before training. CourseWorks and Columbia
course data are not used. Training, results, and checkpoints have not started,
and checkpoint redistribution remains on hold.

After those holds are cleared, `semantic-action-train-srl` will accept only a
fingerprint-matched prepared dataset and a new Git-ignored run directory. No
valid experiment configuration can be frozen before preparation supplies that
fingerprint.

See [DATA_USAGE.md](../DATA_USAGE.md), the
[BabySRL gate](../docs/datasets/babysrl_gate.md), and the
[aggregate audit](../reports/babysrl_audit.md) before any data operation. After
authorized provisional preparation, follow the
[private manual-review workflow](../docs/datasets/babysrl_manual_review.md)
before training.
