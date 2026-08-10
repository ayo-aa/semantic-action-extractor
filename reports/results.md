# Results

No neural model result is available. No prepared BabySRL dataset, training run,
checkpoint, development score, test score, ablation effect, error analysis, or
systems benchmark exists. The blank fields below are intentional and must not
be filled with synthetic or historical course values.

## Software and data-feasibility evidence

| Evidence | Result | What it establishes |
| --- | --- | --- |
| Dependency-free rule baseline | API and CLI implemented | Runnable product interface, not corpus quality |
| SRL software foundation | Data, model, evaluation, provenance, checkpoint, paired engine, and strict CLI boundaries implemented | Tested invariants, not model behavior |
| Real neural model smoke | BERT revision `86b5e0934494bd15c9632b12f734a8a67f723594` completed a synthetic forward/backward pass on Apple MPS with PyTorch 2.13.0 and Transformers 5.14.1; finite loss and expected logits shape | Optional-library compatibility, not corpus training or model quality |
| Systems benchmark boundary | Canonical aggregate p50/p95 latency, batched throughput, peak-memory, checkpoint-size, hardware, and package schema implemented | Reporting contract only; no trained checkpoint was measured |
| MASC feasibility | Rejected; optimistic exact-span ceiling below 99% | Negative dataset decision, not model performance |
| BabySRL structural gate | 18,397 / 18,536 = 99.2501%; 139 fail-closed rejections | Annotation representability, not permission or model accuracy |
| Frozen BabySRL eligibility | 13,713 train / 1,356 development / 1,274 test | In-memory split and leakage-policy accounting, not prepared files |

The 133-document assignment manifest has canonical SHA-256
`73ecae9f81d1d2b9f8495b13b297da9c3d24e387630e71a38ef2420d4c9a5de7`.
TalkBank registration/current-rules acceptance and an authorized manual sample
remain on hold, so the eligible examples have not been written as prepared
training data. See the [BabySRL audit](babysrl_audit.md) and
[data-use record](../DATA_USAGE.md).

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

Every result must identify the BabySRL archive pin, prepared-data fingerprint,
split-manifest digest, duplicate and overlength policies, training-only label
inventory, model/tokenizer revisions, configuration digest, Git revision,
three paired seeds, checkpoint-selection rule, scorer version, hardware,
resolved device, package versions, drop and repair counts, and whether the test
split influenced development.

Per-role support and scores, supplied-predicate diagnostics, and the
preregistered error categories must accompany the aggregate table. Predicate-
candidate and raw-text end-to-end results belong in separate tables; they may
not be inferred from supplied-predicate F1.

CourseWorks and Columbia course data or historical course outputs are not used
as values in this report.
