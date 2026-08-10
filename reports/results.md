# Results

## Current status

No corpus-level extraction result is available. Automated tests and full-release preparation checks verify data-pipeline and software behavior only; they do not provide evidence of model quality. The implemented schemas can preserve optional, source-grounded mention qualifiers, and the primary scorer can measure qualifier labels and exact evidence; no challenge annotation or model result is claimed yet.

QA-SRL Bank 2.1, QA-SRL Gold Standard, and QANom are the selected research sources. Add model values only when each result bundle identifies the dataset release, split, scorer mode, predicate source, consolidation rule, seed, training allowance, hardware, preprocessing revision, and Git commit.

| E1 component | Implementation status | Current evidence | Remaining work |
| --- | --- | --- | --- |
| Archive registry, verification, and extraction | Implemented | Pinned checksums, safety fixtures, and live verified archives | None for selected releases |
| QA-SRL adapter | Implemented | Synthetic fixtures and every release file; documented counts reproduced | Model-facing dataset construction |
| QANom adapter | Implemented | Synthetic fixtures and all three release splits; documented counts reproduced | Model-facing dataset construction |
| Manifests, canonical readers, and quarantine | Implemented | Round-trip, atomicity, drift, overlap, and full-corpus checks | Apply to every future training artifact |
| Consolidation and scorer contracts | Implemented | Boundary, matching, duplicate, role, qualifier, and serialization fixtures; reference-compatible contracts unchanged | Model prediction regression and corpus results |
| Mention-qualifier representation and primary metrics | Implemented | Optional public, canonical, and evaluation-bundle fields with exact cue grounding; primary-only label and exact-evidence F1 fixtures | Annotated challenge records and model predictions |
| Operational-style challenge set | Candidate pilot materials ready | Candidate protocol, annotation guide, source-and-rights notice, and pilot workbook | Author 20 pilot notes; complete independent annotation, adjudication, split assignment, and freeze |

## Answer to the primary research question

TK after the complete multi-seed study.

## Structured versus generative parsing

The rule baseline has no semantic role-question output, so labeled QA-pair F1 is not applicable to it.

| System | Verbal labeled F1 | Nominal labeled F1 | Exact grounding | Ungrounded answer rate | Invalid QA rate | p50 latency | Seeds | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Rule baseline | N/A | N/A | TK | TK | N/A | TK | Deterministic | Corpus evaluation pending |
| QASem T5-small reproduction | TK | TK | TK | TK | TK | TK | TK | Pending |
| Structured verbal encoder | TK | N/A | TK | TK | TK | TK | TK | Pending |
| Structured joint encoder | TK | TK | TK | TK | TK | TK | TK | Pending |

## Predicate-conditioning study

All primary comparisons use the same encoder, data, optimizer, training-token allowance, model-selection rule, and paired seeds.

| Predicate signal | Verbal labeled F1 | Exact span F1 | Unseen-family F1 | Argument ECE | p50 latency | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No target signal | TK | TK | TK | TK | TK | Pending |
| BERT token type | TK | TK | TK | TK | TK | Pending |
| Boundary markers | TK | TK | TK | TK | TK | Pending |
| Learned predicate features | TK | TK | TK | TK | TK | Pending |

The marker and learned-feature finalists receive a matched RoBERTa-family portability run. Token-type conditioning is marked unsupported for encoders without segment embeddings rather than approximated with a different treatment.

## Separate and joint training

The primary joint comparison fixes training tokens or optimizer steps, records sampling ratios, and uses paired seeds. The declared verbal non-inferiority margin is one labeled-F1 point.

| Training strategy | Verbal labeled F1 | Nominal labeled F1 | Paired verbal change | Paired nominal change | Training allowance | Conclusion |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Verbal only | TK | Zero-shot TK or N/A | — | — | TK | TK |
| Nominal only | Zero-shot TK or N/A | TK | — | — | TK | TK |
| Joint, natural ratio | TK | TK | TK | TK | TK | TK |
| Joint, balanced | TK | TK | TK | TK | TK | TK |

## Predicate pipeline

This table prevents candidate-generation and predicate-classification errors from being hidden inside one end-to-end number.

| Stage | Verbal precision | Verbal recall | Verbal F1 | Nominal precision | Nominal recall | Nominal F1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Candidate generation | N/A | TK | N/A | N/A | TK | N/A | Pending |
| Supplied-candidate classification | TK | TK | TK | TK | TK | TK | Pending |
| Raw-text predicate detection | TK | TK | TK | TK | TK | TK | Pending |

## Gold-predicate and complete-pipeline extraction

| System | Predicate source | Labeled QA F1 | Unlabeled QA F1 | Exact span F1 | Duplicate rate | Offset validity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Rule baseline | Detected rules | N/A | TK | TK | TK | TK | Pending |
| QASem reproduction | Gold | TK | TK | TK | TK | TK | Pending |
| Structured encoder | Gold | TK | TK | TK | TK | TK | Pending |
| Best complete pipeline | Detected | TK | TK | TK | TK | TK | Pending |

## Mention-qualifier preservation

Mention qualifiers describe how the source frames an event mention; they do not establish truth, occurrence, completion, assignment, commitment, or execution. `primary-end-to-end-v1` reports qualifier-kind F1 and exact grounded evidence F1. `qasrl-gs-compatible-v1` and `qanom-reference-v1` remain unchanged and do not score qualifiers.

| System | Assessed-predicate coverage | Qualifier-kind precision | Qualifier-kind recall | Qualifier-kind F1 | Exact evidence F1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Rule baseline | TK | TK | TK | TK | TK | Corpus evaluation pending |
| QASem reproduction | TK | TK | TK | TK | TK | Pending |
| Structured encoder | TK | TK | TK | TK | TK | Pending |
| Best complete pipeline | TK | TK | TK | TK | TK | Pending |

An unassessed predicate is excluded from gold qualifier scoring; an assessed predicate with no supported cue remains an explicit negative case. Exact evidence requires the correct qualifier kind and identical grouped token and character spans.

## Generalization

| Evaluation | In-distribution F1 | Held-out F1 | Absolute gap | Split fingerprint | Status |
| --- | ---: | ---: | ---: | --- | --- |
| Naturally unseen predicate families | TK | TK | TK | TK | Pending |
| Controlled held-out families | TK | TK | TK | TK | Pending |
| Size-matched source-domain transfer | TK | TK | TK | TK | Pending |
| Frozen operational-style set | TK | TK | TK | TK | Candidate pilot materials ready; 20 notes, independent annotation, adjudication, and freeze pending |

## Calibration and confidence–coverage behavior

Predicate and argument confidence are evaluated separately because they answer different questions.

| Confidence target | ECE | Brier score | NLL | Risk at 80% coverage | Calibration method | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Predicate correctness | TK | TK | TK | TK | TK | Pending |
| Matched argument + role correctness | TK | TK | TK | TK | TK | Pending |

## Efficiency

| System | Hardware | Parameters | p50 latency | p95 latency | Throughput | Peak memory | Checkpoint size |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule baseline | TK | N/A | TK | TK | TK | TK | No checkpoint |
| QASem T5-small reproduction | TK | TK | TK | TK | TK | TK | TK |
| Structured encoder | TK | TK | TK | TK | TK | TK | TK |

## Planned figures

1. Structured versus generative labeled F1 and exact-grounding failures.
2. Predicate-conditioning performance across paired seeds.
3. Separate versus joint verbal and nominal performance.
4. In-distribution versus held-out predicate-family and source-domain results.
5. Mention-qualifier label versus exact-evidence performance by qualifier kind.
6. Predicate and argument risk–coverage curves.
7. Quality versus batch-one latency and peak memory.

Figures are added only after their result bundles pass source-identity, completeness, and matched-comparison checks.

## Conclusions and negative findings

TK. This section will retain failed hypotheses, regressions, and practical limitations alongside the strongest result.
