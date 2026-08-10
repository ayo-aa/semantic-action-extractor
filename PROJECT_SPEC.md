# Project specification

## Product problem

Short operational sentences often identify an event and its participants, but downstream systems need structured records with auditable links to the original words. This project converts text into source-grounded predicate–argument frames and, where justified, a simpler action-record view.

The research contract is:

```text
(sentence words, supplied predicate index) -> one PropBank BIO tag per word
```

The end-to-end product contract is:

```text
raw text -> predicate candidates -> PropBank role frames -> optional action records
```

These contracts are measured separately. The first matches the original homework; the second adds the predicate-discovery stage needed for raw user text.

## Research basis

The original completed homework implemented predicate-conditioned BERT semantic role labeling on course-provided OntoNotes-derived data. It used `bert-base-uncased`, WordPiece/BIO alignment, a token-type predicate indicator, a linear token-classification head, full fine-tuning, and exact labeled-span scoring.

This repository is rebuilding the software foundation for that method as package modules. The current implementation is incomplete: it contains source-neutral PropBank/PTB conversion, alignment, decoding, scoring, and model-construction primitives, but no public-corpus adapter, training pipeline, checkpoint, or corpus result. It does not include the course dataset, derived labels, starter materials, notebook code, checkpoint, or course examples. Results from that restricted run are not results for this repository.

## Research question

Can the original predicate-conditioned BERT SRL method be reproduced on one lawfully usable public English corpus with exact labeled-role metrics, and what does its explicit predicate indicator contribute relative to the same model without that signal?

## Scope

In scope:

- English predicate-conditioned semantic role labeling;
- one sentence–predicate pair per neural example;
- PropBank core and modifier roles represented as BIO tags;
- supplied verbal predicates for the first public reproduction;
- source-grounded gold argument spans from an approved public corpus;
- a separate rule-based predicate proposer and action-record demo for the raw-text path;
- exact labeled-role evaluation and one bounded predicate-conditioning ablation.

Out of scope:

- QA-SRL, QANom, question generation, and T5/QASem experiments;
- a learned qualifier head, separate nominal-eventivity model, or joint multi-task training;
- nominal-predicate coverage in the first public reproduction;
- reproducing or redistributing restricted course artifacts;
- universal actor/patient inference from `ARG0`/`ARG1`;
- implicit arguments, coreference, intent, task assignment, or completion state;
- autonomous consequential decisions or production-readiness claims.

## Representation contract

### Controlled SRL input

Each example contains:

- ordered surface words;
- one supplied predicate word index;
- one word-level BIO label per word;
- stable sentence, document, and split identifiers;
- optional token metadata needed to trace public-source annotations.

### Neural labels

The label set consists of `O`, predicate labels such as `B-V`, and observed PropBank argument labels such as `B-ARG0`, `I-ARG0`, or `B-ARGM-TMP`. Special and padding positions receive the loss ignore index `-100`.

`ARG0` and `ARG1` are predicate- and sense-relative. The research output preserves those labels exactly. A product adapter may offer actor/patient fields only as a declared convenience mapping, not as a universal semantic equivalence.

### Source grounding

Every emitted role span must map back to the source token sequence. Product-facing character spans must additionally satisfy:

```python
source_text[start:end] == span_text
```

## Candidate public dataset contract

No public corpus has been selected or ingested yet. Universal Proposition Bank 1.0 English EWT was reviewed and rejected for this milestone because it supplies dependency-head arguments rather than the gold argument spans required by the restored BIO contract.

The next candidate is the 88K-word MASC PropBank release in original pointer format with its referenced Penn Treebank parses. It remains behind a feasibility gate; there is no MASC archive adapter or preparation command. A source-neutral parser and fail-closed pointer-to-BIO converter now cover the documented record/tree contract with invented fixtures. Adoption still requires all of the following before corpus integration or training:

- resolve the authoritative archive, version, checksum, license, attribution, source-text lineage, and checkpoint implications;
- verify that every included PropBank document and sentence joins to the supplied parse material;
- inventory verbal predicates, roles, pointer operators, traces, references, continuations, discontinuities, multiword predicates, overlaps, and conversion losses;
- demonstrate exact pointer-to-surface-span conversion on a manually checked, genre-aware sample;
- define explicit include, exclude, and hard-error reasons with no silent gold-label repair;
- freeze a deterministic document-disjoint, genre-aware train/development/test manifest before inspecting model outcomes.

Raw data and prepared examples stay outside Git. Only independently written adapter code, synthetic fixtures, fingerprints, aggregate counts, and approved reports may enter the repository. The full decision record is [docs/datasets/masc_propbank_gate.md](docs/datasets/masc_propbank_gate.md).

## Target architecture

For one sentence–predicate pair:

1. tokenize pre-split words with the `bert-base-uncased` WordPiece tokenizer;
2. align each word label to its pieces;
3. retain `B-*` on the first piece and use `I-*` on continuation pieces;
4. assign `token_type_id = 1` only to the first WordPiece of the supplied predicate;
5. pass token IDs, attention mask, and predicate indicator through BERT;
6. apply one learned linear classifier to every contextual token state;
7. optimize cross-entropy while ignoring special and padding positions;
8. collapse predictions to one label per input word and decode labeled spans.

The original reproduction fully fine-tunes the encoder. The no-predicate variant is a controlled ablation, not a substitute for the reproduction.

Current implementation status: classic/modern PropBank record parsing, Penn Treebank pointer resolution, fail-closed gold BIO conversion, a source-neutral word-level example contract, word/subword gold-label alignment, the first-subword predicate indicator, BERT-plus-linear-head construction, ignored-label loss, BIO repair and decoding, and generic micro exact labeled-span scoring are implemented with synthetic tests. MASC archive discovery and joins, a label-vocabulary builder, subword-prediction collapse, real-model smoke test, training loop, evaluation runner, and empirical results remain pending. Until those pieces exist, this repository is an SRL foundation rather than a usable neural pipeline.

## Hypotheses

- **H1 — predicate signal:** Explicit token-type predicate conditioning will improve exact labeled-role F1 over the same encoder trained without a predicate signal.
- **H2 — boundary difficulty:** Exact-span errors will concentrate in coordination, attachment, punctuation, discontinuities, and subordinate clauses.
- **H3 — pipeline gap:** Only after the controlled reproduction is complete, a separately evaluated raw-text path is expected to underperform supplied-predicate evaluation because candidate discovery adds missed and spurious predicates.

## Experiment plan

### E0 — rule baseline

- Validate JSON schema, exact source offsets, configuration, and CLI behavior.
- Keep the actor/patient action view separate from PropBank SRL claims.
- After a corpus is selected, optionally measure predicate-candidate recall and latency without treating the baseline as a role-labeling comparator.

### E1 — SRL and data foundation

- Implement tokenizer-independent word/subword alignment.
- Implement predicate indicators and the BERT token-classification boundary.
- Implement exact labeled-span decoding and scoring, excluding predicate `V` and continuation `C-V` spans.
- Keep the implemented pointer converter source-neutral; after the source review passes, add one corpus adapter with only the joins and representations justified by that archive.
- Preserve source document identifiers and either an official split or the predeclared document-disjoint manifest justified by the selected source.
- Use only synthetic fixtures in repository tests.

### E2 — public neural reproduction

- Fine-tune `bert-base-uncased` with token-type predicate conditioning.
- Start from the original homework hyperparameters: batch size 32, learning rate `1e-5`, two epochs, AdamW, and full encoder updates.
- Treat those settings as the reproduction anchor, then document any change required by the public corpus or available hardware.
- First complete one end-to-end development run. Freeze the final protocol, then run three fixed final seeds and report mean and sample standard deviation.
- Select checkpoints on development data; evaluate the frozen test split once for the declared primary run.

### E3 — predicate-conditioning ablation

Hold data, split, encoder, optimizer, training budget, and seed set constant while comparing only:

1. no predicate signal;
2. the original token-type predicate indicator.

### E4 — optional post-reproduction analysis

- Break down results by role, predicate frequency, sentence length, and WordPiece fragmentation.
- Report truncation and dropped-example rates.
- Measure single-example and batched p50/p95 latency, throughput, peak memory, checkpoint size, hardware, and package revisions.
- Consider held-out-predicate or raw-text pipeline studies only after the public reproduction and bounded ablation are complete; scope and report them separately.

## Metrics

Primary public annotation metric:

- micro-averaged exact labeled-span precision, recall, and F1 over gold PropBank argument roles.

The metric excludes predicate `V` and continuation `C-V` spans. It gives no partial credit for a correct boundary with the wrong role. Any source construct that cannot be represented faithfully in the declared word-level BIO scheme must be reported through conversion coverage and exclusion counts rather than silently simplified.

Secondary metrics:

- per-role precision, recall, F1, and support;
- token accuracy as a diagnostic;
- malformed-BIO and repair counts;
- predicate-candidate precision, recall, and F1;
- end-to-end frame metrics;
- truncation/drop rates;
- latency, throughput, memory, and artifact size;
- mean and sample standard deviation across paired seeds.

## Experimental controls

- An official public split is immutable when one exists; otherwise the versioned document-disjoint split manifest is frozen before model outcomes are inspected.
- Test labels are not used for model selection or preprocessing decisions.
- Every comparison uses the same prepared-data fingerprint.
- Ablations share optimizer, training-token budget, batch policy, and paired seeds.
- Pointer conversion, role filtering, and exclusion rules are frozen and versioned before model training.
- All result tables record the Git revision, configuration, data release, scorer version, hardware, runtime, and package versions.
- Unit tests are never reported as model-quality evidence.

## Definition of done

E1 is complete when:

- no QA-specific implementation or active data/experiment contract remains in the restored scope (the scope exclusion may name discarded directions);
- the dependency-free baseline still installs and runs;
- SRL BIO alignment, decoding, and exact metrics have synthetic automated tests;
- the approved public adapter preserves supplied predicates, gold role spans, split assignments, and documents;
- all pointer-conversion losses and unsupported source constructs are counted with explicit reasons;
- restricted course artifacts are absent;
- the README diagram accurately renders the implemented and planned boundaries.

The research project is complete when:

- public-data preparation is reproducible without committing the corpus;
- the original BERT reproduction and all declared baselines run from versioned configurations;
- multi-seed public results, robustness analysis, and systems costs are reported;
- supplied-predicate and end-to-end claims remain separate;
- any released checkpoint has verified redistribution rights and a completed model card;
- limitations and negative findings are reported alongside the strongest result.
