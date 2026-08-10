# Neural runtime rehearsal and EWT fit preflight

Status: **PASS for synthetic orchestration and one actual prepared-EWT optimizer step; no research result**

On 2026-08-09, the complete paired neural path was exercised at Git revision
`3d93c10ac3d92b0695326f0816a5fadaa05fa5c8`. The rehearsal used 14 invented
examples (4 train, 2 development, and 8 test) with no EWT, BabySRL,
CourseWorks, or other corpus content. All inputs, configurations, checkpoints,
and result files remain Git-ignored.

## Frozen runtime

| Item | Value |
| --- | --- |
| Host | Apple M4, 16 GiB memory |
| Accelerator | Apple MPS |
| Python | 3.12.13 |
| PyTorch | 2.13.0 |
| Transformers | 5.14.1 |
| Model and tokenizer | `google-bert/bert-base-uncased` |
| Immutable revision | `86b5e0934494bd15c9632b12f734a8a67f723594` |

The complete 213-test suite also passed under the supported Python 3.12
environment after the rehearsal record was added.

## Maximum-shape optimizer preflight

One synthetic AdamW optimizer step completed at the proposed maximum input
shape: batch size 32, sequence length 128, and 21 output labels. Forward pass,
backward pass, gradient clipping, and parameter update all completed on MPS in
3.0977 seconds. PyTorch reported 2,939,641,088 bytes currently allocated by MPS
and 5,401,001,984 bytes allocated by its driver after the step.

This establishes that the original batch-size anchor fits this host for one
maximum-shape step. It is not a full-run memory guarantee, throughput result,
or model-quality measurement.

## Actual prepared-EWT optimizer-step preflight

On 2026-08-10, the frozen predicate-signal configuration ran one full step on
the real private prepared EWT data. The preparation and configuration identities
were:

| Item | Value |
| --- | --- |
| Adapter implementation commit | `9b7c94ec9be4a9b56c3cd7df3cb9a83b34b87f42` |
| Prepared-data fingerprint | `2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b` |
| Predicate config digest | `9bf8cd7c7a839bd9bfb6b39fde616f7e6f42d2ef163ea5f47b7afeeee1120bdc` |
| Batch size | 32 |
| Longest retained sequence | 118 model tokens |
| Output labels | 111 |
| Completed operations | Forward, backward, gradient clipping, AdamW update, scheduler step |
| Elapsed time | 3.0069 seconds |
| MPS allocated bytes | 3,211,741,952 |
| Outcome | PASS |

This actual-data preflight establishes that the longest retained batch shape
fits and completes one optimizer update under the frozen predicate variant. It
does not establish sustained memory or thermal behavior, total runtime,
convergence, accuracy, an ablation effect, or checkpoint benchmark performance.
See the aggregate-only [EWT preparation record](ewt_preparation.md).

## Paired training rehearsal

The strict training command completed all six combinations of three paired
seeds (`13`, `17`, and `23`) and two variants (predicate signal and all-zero
signal). The rehearsal retained the planned batch size 32, maximum length 128,
learning rate `1e-5`, weight decay `0.01`, and gradient clipping norm `1.0`, but
used one epoch rather than the planned two because its purpose was runtime
validation.

Each run constructed the pinned BERT model, optimized its parameters, selected
a checkpoint using development argument F1, wrote the integrity-checked bundle,
verified its state digest, reloaded it, and performed the final synthetic test
evaluation. The trainer then atomically published the paired aggregate.

| Artifact | SHA-256 |
| --- | --- |
| Synthetic prepared-data fingerprint | `44551a644ba68f4a78ba9cf5026e7b7447ddf1d4ec216a45f06ea7715458c1fb` |
| Canonical paired-result file | `db3c84be08ff5d84ce52840af8df6235736b4d665f41e820219b81d65d1941b1` |

Synthetic development, test, and ablation scores are intentionally omitted.
Fourteen invented examples cannot support a model-quality claim.

## Checkpoint benchmark rehearsal

One saved checkpoint from each variant was passed through the strict benchmark
command. Both runs validated the configuration, prepared-data fingerprint,
label inventory, checkpoint metadata, and serialized-state digest before model
loading. Each used 10 warmups and 100 measured iterations for fixed batch sizes
1 and 8 on MPS, then wrote only canonical aggregate output.

| Variant | Canonical benchmark-file SHA-256 |
| --- | --- |
| Predicate signal | `17ecb7fedb142ce2d187a133f46330a0a7cae73e4925da837a6c0ebf05c0c4fc` |
| No predicate signal | `c18b966fc96604d9de98efa57d90e54d4a78c93083607cc8d68ecd2481747a2a` |

Synthetic latency, throughput, memory, checkpoint-size, and accuracy values are
not portfolio results and are intentionally not copied into the public result
tables.

## Independent resume-durability review

On 2026-08-10, the interruption-recovery boundary passed independent review
with injected failures and adversarial filesystem artifacts. `--resume` accepts
only the identical partial run: output and provenance, paired configuration
digests, prepared dataset and fingerprint, exact Git revision, and runtime
identity must all match. Every reusable result/checkpoint pair is revalidated,
as is each completed predicate/no-predicate seed pair.

The command retains a canonical complete journal and safely recovers only exact
writer-owned interrupted atomic writes, next-checkpoint staging, and
checkpoint-tombstone cleanup. Lookalike or unknown artifacts fail closed, and
a per-output nonblocking lock rejects concurrent writers. Operationally, the
recovery command is the original training command with only `--resume` added.

This durability review itself used synthetic and injected test boundaries. EWT
was prepared separately afterward, but neither activity ran the paired research
experiment, produced a score, or created a research checkpoint.

## Interpretation and remaining gate

The evidence validates the installed optional dependencies, local Apple-MPS
execution, synthetic six-run orchestration, checkpoint round trip, paired
aggregation, systems-benchmark boundary, exact-match interruption recovery,
and one actual prepared-data optimizer step. It also shows that Colab or
Columbia compute is not required for the planned run.

It does **not** establish semantic-role performance. The selected EWT adapter
and source gate pass using public pinned sources. The
[aggregate-only private source review](ewt_private_source_review.md) inspected
all 13 predicate-anchor divergences, the one token-width mismatch, and 30/30
deterministically selected aligned verbal records; no account, registration,
CourseWorks session, LDC download, or user manual review is required. Private
ignored EWT preparation and both paired configurations are now frozen, and the
actual-data fit preflight passes. The two-epoch, three-seed paired training run
and real checkpoint benchmark have not run and await reliable power.
Raw/prepared text and future trained weights remain private; public evidence is
limited to code and non-reconstructive aggregates pending a separate weights
review.
