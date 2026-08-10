# Neural runtime rehearsal

Status: **PASS on invented synthetic data only; no research result**

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

This review used synthetic and injected test boundaries. It did not prepare
EWT data, run the research experiment, produce a score, or create a research
checkpoint.

## Interpretation and remaining gate

The rehearsal validates the installed optional dependencies, local Apple-MPS
execution, six-run orchestration, checkpoint round trip, paired aggregation,
systems-benchmark boundary, and exact-match interruption recovery. It also
shows that Colab or Columbia compute is not required for the planned run.

It does **not** establish semantic-role performance. The selected EWT adapter
and source gate pass using public pinned sources. The
[aggregate-only private source review](ewt_private_source_review.md) inspected
all 13 predicate-anchor divergences, the one token-width mismatch, and 30/30
deterministically selected aligned verbal records; no account, registration,
CourseWorks session, LDC download, or user manual review is required. The
research experiment still awaits private ignored EWT preparation, a frozen
data fingerprint and model configurations, and then the two-epoch paired
training run. Raw/prepared text and trained weights remain private; public
evidence is limited to code and non-reconstructive aggregates pending a
separate weights review.
