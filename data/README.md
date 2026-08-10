# Data directory

No dataset is included. Raw sources, prepared outputs, review material, and
learned artifacts are ignored by Git.

## Selected EWT route

The implemented `semantic-action-prepare-ewt` adapter expects two public,
revision-pinned Git checkouts:

- `data/raw/ewt_sources/propbank-release` at
  `4abade0b53ce4a181e1d98b3518101c1a44d395a`; and
- `data/raw/ewt_sources/UD_English-EWT` at
  `6e064999a75b9c941c515ce1be98352e6f9831e0` (UD r2.2).

No account, registration, CourseWorks login, LDC download, or user-provided
file is required. The adapter verifies both revisions and relevant worktrees,
then performs the validated inferred cross-release join documented in the
[EWT gate](../docs/datasets/ewt_propbank_gate.md) and
[aggregate audit](../reports/ewt_propbank_audit.md). The
[aggregate-only private source review](../reports/ewt_private_source_review.md)
records inspection of all 13 predicate-anchor divergences, the one token-width
mismatch, and 30/30 deterministically selected aligned verbal records without
publishing corpus text or identifiers.

The frozen aggregate stages are:

| Stage | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Structural verbal gate | — | — | — | 38,639 |
| Word-aligned | 31,174 | 3,806 | 3,655 | 38,635 |
| Prepared eligible | 31,101 | 3,775 | 3,610 | 38,486 |
| Modeled at `max_length=128` | 31,039 | 3,775 | 3,610 | 38,424 |

The pinned tokenizer preflight builds 111 train-derived labels, including `O`
and continuation closure, with no development or test label outside train. No
real EWT prepared split, data fingerprint, training run, result, or checkpoint
exists yet.

Audit without writing prepared records:

```bash
semantic-action-prepare-ewt \
  data/raw/ewt_sources/propbank-release \
  data/raw/ewt_sources/UD_English-EWT
```

The first real private preparation will add
`--output-directory data/processed/ewt`. That command has not yet been run. Its
canonical split JSONL, manifest, and provenance receipt must remain ignored.

## Rejected and fallback artifacts

- `data/raw/Propbank-original-format.zip`, if present, is rejected MASC audit
  evidence only. It is not prepared or used for training.
- `data/raw/BabySRL.zip`, if present, is historical fallback audit evidence. Its
  18,397 / 18,536 structural pass is preserved, but BabySRL is not the active
  source and no registration or manual-review action is needed for the EWT path.

## Publication boundary

The UD distribution expressly separates its CC BY-SA annotation/database grant
from copyrights in the underlying text. Raw checkouts, reconstructed words,
prepared examples, and future trained weights remain private and ignored. Only
source-neutral code and non-reconstructive aggregate metrics are public pending
a separate checkpoint-release review.

After private preparation supplies a fingerprint, `semantic-action-train-srl`
will accept only a matching prepared dataset and a new ignored run directory.
See [DATA_USAGE.md](../DATA_USAGE.md) before any data or artifact operation.
