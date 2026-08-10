# Results

No research neural-model result is available. No prepared EWT dataset,
research training run, research checkpoint, development score, test score,
ablation effect, error analysis, or research systems benchmark exists. A full
invented-data runtime rehearsal did complete, but the blank fields below are
intentional and must not be filled with synthetic or historical course values.

## Software and data-feasibility evidence

| Evidence | Result | What it establishes |
| --- | --- | --- |
| Dependency-free rule baseline | API and CLI implemented | Runnable product interface, not corpus quality |
| SRL software foundation | Data, model, evaluation, provenance, checkpoint, paired engine, and strict CLI boundaries implemented | Tested invariants, not model behavior |
| Full neural runtime rehearsal | Six BERT seed/variant runs completed on invented data at revision `86b5e0934494bd15c9632b12f734a8a67f723594`; every checkpoint was verified and reloaded and the paired result was published | Local optional-library and orchestration compatibility, not research-corpus training or model quality |
| Resume durability | Exact-identity resume, result/checkpoint and paired-seed validation, canonical complete journal, bounded interrupted-write/staging/tombstone recovery, unknown-artifact rejection, and per-output nonblocking lock passed independent synthetic fault review | Safe orchestration recovery, not EWT preparation, research training, or model quality |
| Checkpoint benchmark rehearsal | One invented-data checkpoint per variant completed fixed batch-1/batch-8 aggregate benchmarking on Apple MPS | Real benchmark execution, but no research systems measurement |
| MASC feasibility | Rejected; optimistic exact-span ceiling below 99% | Negative dataset decision, not model performance |
| EWT adapter and structural gate | Implemented; 38,639 structurally valid verbal predicates | Source/annotation feasibility, not model performance |
| EWT word alignment | 38,635 after one four-predicate sentence exclusion | Validated inferred cross-release join, not PropBank's prescribed LDC mapping |
| Frozen EWT prepared eligibility | 31,101 train / 3,775 development / 3,610 test | In-memory leakage, conflict, and eval-dedup accounting, not prepared files |
| EWT `max_length=128` preflight | 31,039 train / 3,775 development / 3,610 test; 111 train-derived labels including `O` and continuation closure; no unseen development/test labels | Model-input and vocabulary feasibility, not training |
| BabySRL historical fallback | 18,397 / 18,536 = 99.2501% structural coverage | Preserved fallback audit, not the active source |

The selected sources are public pinned Git revisions. The
[aggregate-only private source review](ewt_private_source_review.md) inspected
all 13 predicate-anchor divergences, the one token-width mismatch, and 30/30
deterministically selected aligned verbal records; no account, registration,
or user-supplied file is needed. The eligible examples have not yet been
written as prepared training data. See the [EWT audit](ewt_propbank_audit.md),
[data-use record](../DATA_USAGE.md), and
[EWT gate](../docs/datasets/ewt_propbank_gate.md). The
[neural runtime record](neural_runtime_smoke.md) documents the separate
invented-data rehearsal and its non-claim boundary.

## Future supplied-predicate results

An em dash means “not run,” not zero.

| Variant | Seeds | Development exact argument F1, mean ± sample SD | Test exact argument P | Test exact argument R | Test exact argument F1, mean ± sample SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Predicate signal | — | — | — | — | — |
| No predicate signal | — | — | — | — | — |
| Paired signal effect | — | — | — | — | — |

## Future systems results

| Measure | Rule baseline | Predicate-signal model | No-signal model |
| --- | ---: | ---: | ---: |
| Single-example p50 latency | — | — | — |
| Single-example p95 latency | — | — | — |
| Batched throughput | — | — | — |
| Peak memory | — | — | — |
| Checkpoint size | n/a | — | — |

## Required disclosure before filling the tables

Every result must identify both EWT source commits, the prepared-data
fingerprint, official split and duplicate/conflict policies, overlength policy,
training-only label inventory, model/tokenizer revisions, configuration digest,
Git revision,
three paired seeds, checkpoint-selection rule, scorer version, hardware,
resolved device, package versions, drop and repair counts, and whether the test
split influenced development. Systems results must also state the exact fixed
batch protocol and memory method: resettable CUDA allocator peak or
process-lifetime peak RSS for CPU/MPS, including model load.

Per-role support and scores, supplied-predicate diagnostics, and the
preregistered error categories must accompany the aggregate table. Predicate-
candidate and raw-text end-to-end results belong in separate tables; they may
not be inferred from supplied-predicate F1.

CourseWorks and Columbia course data or historical course outputs are not used
as values in this report.
