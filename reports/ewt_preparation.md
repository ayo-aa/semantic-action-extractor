# EWT private preparation and real-data MPS preflight

Status: **PASS for private preparation and one optimizer-step preflight; paired research training and benchmarking not run**

Record date: 2026-08-10

## Scope

The selected pinned PropBank/UD EWT route passed its source gate and was written
to Git-ignored private storage by the production adapter implemented at commit
`9b7c94ec9be4a9b56c3cd7df3cb9a83b34b87f42`. This report retains only
non-reconstructive counts, fingerprints, digests, and runtime measurements. It
contains no corpus text, token sequence, document identifier, sentence
identifier, example identifier, prepared record, or local private path.

The frozen source and pretrained-component identities are:

| Component | Exact revision |
| --- | --- |
| PropBank release | `4abade0b53ce4a181e1d98b3518101c1a44d395a` |
| UD English EWT r2.2 | `6e064999a75b9c941c515ce1be98352e6f9831e0` |
| `google-bert/bert-base-uncased` model and tokenizer | `86b5e0934494bd15c9632b12f734a8a67f723594` |

## Prepared artifact identity

The canonical prepared-data fingerprint is
`2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b`.

| Private artifact | Examples | SHA-256 |
| --- | ---: | --- |
| Train split JSONL | 31,101 | `c2c7c290ac43f30bf2fba782ef5ec74f008c55b4049c382dce6cca3a776ce0fb` |
| Development split JSONL | 3,775 | `e292aa5073d7ed73bba0d3ed566b6e8bdd107cddea7a62c7283855c8fc11b622` |
| Test split JSONL | 3,610 | `314407d5ce93eb1be23fc7597bdfa1b33fd64a8e2251637a28bf52fb110709e8` |
| EWT provenance receipt | — | `a352307370e82e5d7e1f998ede65887f96116fc7ba349c4b0ff167745ddf5b52` |
| **Total prepared examples** | **38,486** | — |

The training-only vocabulary contains 111 labels. Full tokenization at
`max_length=128` excludes 62 train examples and no development or test
examples, leaving 31,039 train, 3,775 development, and 3,610 test model inputs.
These values reproduce the frozen source-gate and preflight accounting.

## Frozen paired configurations

| Variant | Tracked configuration | Canonical configuration digest |
| --- | --- | --- |
| Predicate signal | `configs/ewt_predicate_signal.toml` | `9bf8cd7c7a839bd9bfb6b39fde616f7e6f42d2ef163ea5f47b7afeeee1120bdc` |
| No predicate signal | `configs/ewt_no_predicate_signal.toml` | `19ff94ff8bf839ee2fd5ebdab8ffed24b1ad7b243cc413a2ba687262f8f9d866` |

Both configurations bind the same prepared-data fingerprint, exact model and
tokenizer revision, maximum length, batch size, optimizer protocol, paired
seeds, and development-only checkpoint-selection rule. They differ only in the
predeclared predicate-signal variant.

## Real-data MPS optimizer-step preflight

The predicate-signal configuration completed one full optimizer step on the
actual prepared EWT data using Apple MPS:

| Item | Result |
| --- | --- |
| Prepared-data fingerprint | `2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b` |
| Variant | Predicate signal |
| Batch size | 32 |
| Longest retained sequence | 118 model tokens |
| Output labels | 111 |
| Operations completed | Forward, backward, gradient clipping, AdamW update, scheduler step |
| Elapsed time | 3.0069 seconds |
| MPS allocated bytes | 3,211,741,952 |
| Outcome | PASS |

The canonical machine-readable receipt is
[`ewt_mps_preflight.json`](ewt_mps_preflight.json), with SHA-256
`a667149e4443ff582c6dd816b51dc7ac5fe3a80b20373e652a0723db895c4fbd`.
It binds the implementation revision, data and configuration digests, seed,
stable longest-retained-batch selection policy, runtime versions, completed
operations, synchronization, and memory-measurement semantics.

This establishes that one full batch at the longest retained real-data shape
fits and updates successfully under the frozen predicate configuration. It is
not a sustained-memory, thermal, duration, convergence, accuracy, ablation, or
systems-benchmark result.

## Remaining boundary

No paired seed run, development selection, test evaluation, research
checkpoint, paired result, error analysis, or real checkpoint benchmark was
created by preparation or preflight. The prepared corpus and any future run
artifacts remain ignored and private. Public reporting remains limited to
source-neutral code and non-reconstructive aggregate evidence pending the
separate trained-weight review.
