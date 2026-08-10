# Project specification

## Product problem

Short operational sentences often identify an event and its participants, but
downstream systems need structured records with auditable links to the original
words. This project converts text into source-grounded predicate–argument frames
and, where justified, a simpler action-record view.

The controlled research contract is:

```text
(sentence words, supplied predicate index) -> one PropBank BIO tag per word
```

The end-to-end product contract is:

```text
raw text -> predicate candidates -> PropBank role frames -> optional action records
```

The first contract measures role labeling when the predicate is known. The
second also measures predicate discovery. Their inputs, failure modes, and
results must remain separate.

## Research basis and boundary

The architecture preserves the method of the original course exercise:
`bert-base-uncased`, one sentence–predicate instance, WordPiece/BIO alignment,
a token-type predicate indicator, a linear token-classification head, full
fine-tuning, and exact labeled-span scoring.

This public implementation was written as independent package code.
CourseWorks is not used. No Columbia course dataset, label inventory, starter
material, assignment text, corpus example, checkpoint, or restricted-run
measurement is used as project data or public empirical evidence.

## Research question

Can predicate-conditioned BERT be reproducibly fine-tuned on a lawfully usable
public English span-SRL corpus, and what does its explicit predicate signal
contribute relative to the same model trained with an all-zero signal?

## Scope

In scope:

- English supplied-predicate semantic role labeling;
- one sentence–verbal-predicate pair per model example;
- PropBank core and modifier roles represented as word-level BIO tags;
- source-grounded, overt gold argument spans;
- a deterministic rule-based predicate proposer for the separate raw-text path;
- exact labeled argument-span evaluation and one bounded signal ablation;
- reproducible dataset, experiment, provenance, and checkpoint boundaries.

Out of scope:

- QA-SRL, QANom, question generation, and T5/QASem experiments;
- nominal predicates in the first reproduction;
- reproducing or redistributing course artifacts;
- inferring implicit arguments or controllers to manufacture gold spans;
- treating `ARG0`/`ARG1` as universal actor/patient categories;
- coreference, intent, task assignment, or completion-state inference;
- autonomous consequential decisions or production-readiness claims.

## Representation contract

Each prepared example must contain:

- ordered surface words;
- exactly one supplied predicate word index;
- exactly one validated word-level BIO sequence;
- stable example, sentence, document, and split identifiers;
- traceable source metadata that does not enter the model input.

`O` is label ID zero. The immutable label vocabulary is built from training
examples only and contains the continuation closure required by WordPiece
alignment. Development or test labels absent from that vocabulary are errors,
not silently mapped to `O`. Special and padding positions use the loss ignore
index `-100`.

Every output role span maps to the original word sequence. Product-facing
character spans additionally satisfy:

```python
source_text[start:end] == span_text
```

## Data decision and frozen EWT contract

[Universal Proposition Bank 1.0 English EWT](https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT)
was rejected as the target because its dependency-head roles do not provide the
full gold spans required here. That decision does not reject the underlying EWT
source. [MASC PropBank](https://anc.org/data/masc/downloads/data-download/) was
also rejected because trace-only arguments place its optimistic exact-span
ceiling below the predeclared 99% gate.

The selected route joins the pinned [PropBank EWT gold skeletons](https://github.com/propbank/propbank-release/tree/4abade0b53ce4a181e1d98b3518101c1a44d395a/data/google/ewt)
at commit `4abade0b53ce4a181e1d98b3518101c1a44d395a` to the words in
[UD English EWT r2.2](https://github.com/UniversalDependencies/UD_English-EWT/tree/6e064999a75b9c941c515ce1be98352e6f9831e0)
at commit `6e064999a75b9c941c515ce1be98352e6f9831e0`. The implemented adapter
pins both revisions and relevant worktrees, joins normalized document identity
and sentence position, validates role-column structure and token width, assigns
the official PropBank document splits, and applies fail-closed leakage and
ambiguity controls.

This is a validated inferred cross-release join, not the LDC-based mapping
prescribed by PropBank. The
[aggregate-only private source review](reports/ewt_private_source_review.md)
inspected all 13 metadata-versus-primary predicate-anchor divergences, the one
token-width mismatch, and 30/30 deterministically selected aligned verbal
records. It requires no account, registration, CourseWorks login, LDC download,
or user-supplied corpus file.

The frozen pre-outcome accounting is:

| Stage | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Structurally valid verbal predicates | — | — | — | 38,639 |
| Aligned after one token-width exclusion | 31,174 | 3,806 | 3,655 | 38,635 |
| Prepared eligible after leakage, conflict, and eval deduplication | 31,101 | 3,775 | 3,610 | 38,486 |
| Modeled at `max_length=128` | 31,039 | 3,775 | 3,610 | 38,424 |

The pinned tokenizer preflight builds a train-derived vocabulary of 111 labels,
including `O` and continuation closure, with no development or test label
absent from it. These values are in-memory audit and model-input preflight
counts, not prepared files, training evidence, or model accuracy.

The [EWT gate](docs/datasets/ewt_propbank_gate.md) keeps raw sources,
reconstructed text, prepared data, and future weights ignored and private.
Source-neutral code and non-reconstructive aggregate metrics may be public;
checkpoint release remains a separate review. BabySRL's 99.2501% structural
audit is preserved as historical fallback evidence, but its registration and
manual-review gates are not active while the EWT route remains viable.

## Target architecture

For one sentence–predicate pair:

1. tokenize pre-split words with a pinned `bert-base-uncased` tokenizer;
2. align each training word label to all of its WordPieces;
3. retain `B-*` on the first piece and use `I-*` on continuation pieces;
4. set `token_type_id = 1` only on the first WordPiece of the supplied
   predicate, or set every value to zero for the controlled ablation;
5. pass token IDs, attention mask, and explicit token-type IDs through a pinned
   BERT encoder;
6. apply one learned linear classifier to every contextual token state;
7. optimize cross-entropy over non-special, non-padding pieces;
8. collapse subword predictions by taking exactly the first piece for each
   source word;
9. deterministically repair malformed predicted BIO transitions; and
10. decode and score exact labeled argument spans.

No input is silently truncated. Full tokenization occurs first, and examples
longer than the configured maximum are dropped with explicit IDs and counts.
The encoder and classifier are fully fine-tuned.

## Implemented software

The following boundaries are implemented and tested with synthetic fixtures or
injected runtimes:

- source-neutral PropBank/PTB parsing and fail-closed pointer-to-BIO conversion;
- strict canonical three-split JSONL and manifest I/O;
- content fingerprints, duplicate identities, document/sentence leakage checks,
  and exact sentence-text leakage checks;
- immutable train-only label-vocabulary construction;
- gold WordPiece alignment and first-subword prediction collapse;
- deterministic split selection, full-length checks, collation, and explicit
  predicate-signal/no-signal batches;
- pinned model and tokenizer revision configuration;
- BERT-plus-linear-head construction and ignored-label cross-entropy;
- exact micro argument P/R/F1, per-role P/R/F1, token accuracy, repaired-tag
  counts, and supplied-predicate diagnostics;
- canonical configuration digests and run metadata containing Git revision,
  data fingerprint, labels, packages, hardware, seeds, device, timestamps,
  counts, drop statistics, and initial-state fingerprint;
- atomic checkpoint bundles containing labels, metadata, and a state dictionary
  whose SHA-256 is verified before load;
- pinned PropBank/UD EWT source verification, inferred cross-release joining,
  primary-`V` predicate anchoring, official splits, conflict/leakage controls,
  aggregate audit, and optional ignored preparation boundary;
- aggregate-only private source-review evidence covering 13/13 anchor
  divergences, the one width mismatch, and 30/30 deterministic aligned records;
- AdamW training, deterministic batches, gradient clipping, linear warmup
  followed by constant learning rate, development-only checkpoint selection,
  best-state reload, and one test evaluation per run;
- exactly three paired signal/no-signal seeds, with paired configurations
  required to differ only by variant and paired initial states required to have
  the same fingerprint; and
- a strict training CLI that validates both configs, the prepared-data
  fingerprint, a new Git-ignored output location, and an exact Git revision;
  stages all six runs, records non-sensitive partial-failure state, retains a
  canonical complete journal, and atomically publishes canonical paired
  results;
- independently reviewed exact-match resume semantics: the same command plus
  `--resume` must identify the same partial output, provenance, configs, data,
  Git revision, and runtime; completed result/checkpoint and paired-seed
  identities are revalidated; only exact interrupted atomic-write,
  checkpoint-staging, and checkpoint-tombstone residue is recovered; unknown or
  lookalike artifacts fail closed; and a per-output nonblocking lock excludes
  concurrent writers; and
- a validated-checkpoint systems benchmark CLI and canonical aggregate-only
  contract for p50/p95 latency, batched throughput, explicitly identified peak
  memory, checkpoint size, hardware, and package revisions without serializing
  examples, IDs, paths, or raw timing samples.

This software completeness is not empirical validation. The concrete
PyTorch/Transformers path has not been run on a prepared EWT dataset.

## Hypotheses

- **H1 — predicate signal:** Explicit token-type predicate conditioning will
  improve exact labeled argument-span F1 over the all-zero signal variant.
- **H2 — boundary difficulty:** Errors will concentrate in coordination,
  attachment, punctuation, discontinuities, and subordinate clauses.
- **H3 — pipeline gap:** A future raw-text pipeline will underperform the
  supplied-predicate setting because predicate discovery adds misses and false
  positives.

## Experiment plan

### E0 — rule baseline

- Keep the transparent rule extractor runnable through the API and CLI.
- Preserve its action-record interpretation as a product heuristic.
- Evaluate predicate-candidate recall and systems costs separately from neural
  supplied-predicate role labeling.

### E1 — private EWT preparation and verification

- Reproduce the passing aggregate audit from the two pinned public checkouts.
- Write prepared JSONL only to ignored local storage.
- Require the prepared counts, exclusions, train-derived label inventory, and
  zero unseen development/test labels to match the frozen preflight.
- Record the resulting dataset fingerprint without publishing corpus content.

### E2 — public neural reproduction

- Pin exact model and tokenizer repository revisions.
- Use the original anchor of batch size 32, learning rate `1e-5`, two epochs,
  AdamW, full fine-tuning, and constant learning rate unless a pre-training
  hardware check requires a documented change.
- Run exactly three fixed seeds after the protocol is frozen.
- Select each seed's checkpoint using development argument F1 or development
  loss, as declared in the configuration.
- Reload the selected state and evaluate test exactly once per run.
- Report mean and sample standard deviation, never only the strongest seed.
- If interrupted, rerun the identical training command with only `--resume`
  added; never edit, copy, or hand-reconcile the partial run.

### E3 — predicate-conditioning ablation

For every seed, compare:

1. `predicate_signal`, with one first-piece token-type indicator; and
2. `no_predicate_signal`, with all-zero token-type IDs.

Data fingerprint, labels, model/tokenizer revisions, optimizer, epochs, batch
policy, seed, resolved device, and initial state must match within each pair.

### E4 — analysis and systems evidence

- Report per-role performance and the preregistered error taxonomy.
- Report overlength exclusions and BIO repairs.
- Measure single-example and batched p50/p95 latency, throughput, peak memory,
  checkpoint size, hardware, and package revisions.
- Keep raw-text candidate and end-to-end analyses in separate tables.

## Metrics

The primary metric is micro-averaged exact labeled argument-span precision,
recall, and F1. A span is correct only when both its role and word boundaries
match. Predicate `V` and `C-V` spans are excluded because the predicate is
supplied.

Secondary measures are:

- per-role exact P/R/F1 and support;
- word-level token accuracy;
- malformed prediction repairs;
- supplied-predicate anchor, spurious-predicate, and predicate-overlap
  diagnostics;
- conversion coverage and every fail-closed exclusion reason;
- overlength drop rates;
- mean and sample standard deviation across paired seeds; and
- separately scoped candidate, end-to-end, latency, throughput, memory, and
  artifact-size measures.

## Experimental controls

- The pinned PropBank official EWT document assignments cannot change after
  outcomes are seen.
- Exact text crossing split boundaries is excluded from every affected split;
  conflicting identical inputs are excluded from every affected split; exact
  evaluation semantics are deduplicated within development and test while
  nonconflicting train frequency is preserved.
- The vocabulary is learned from training examples only.
- Test labels never select preprocessing, hyperparameters, or checkpoints.
- Paired variants may differ only in the predicate-signal switch.
- Every result records the prepared-data fingerprint, configuration digest,
  Git revision, model/tokenizer revisions, labels, seed, hardware, device,
  package versions, drop counts, and checkpoint hash.
- Unit tests and aggregate conversion counts are never presented as model
  accuracy.

## Definition of done

### Completed software milestone

The public code milestone is complete when the implemented boundaries above,
the paired training command, and their synthetic tests pass; the MASC no-go,
selected EWT gate, and BabySRL fallback status reconcile exactly; raw/prepared
data remain outside Git; and the documentation preserves the supplied-predicate
versus raw-text distinction. Those deliverables are present in the current
branch.

### Remaining evidence milestone

The research project is **not** complete until all of the following occur:

- ignored EWT prepared data are generated and reproduce every frozen gate count;
- the prepared fingerprint and training-derived label vocabulary are frozen for
  the experiment;
- the pinned real PyTorch/Transformers model completes a prepared-corpus smoke
  run;
- all three paired seeds complete for both variants;
- development, test, per-role, error, overlength, and repair results are
  reported without selecting the best seed;
- latency, throughput, memory, hardware, runtime, and checkpoint-size costs are
  measured;
- supplied-predicate and raw-text results remain separately labeled; and
- any proposed checkpoint passes an explicit redistribution, privacy,
  memorization/leakage, and model-card review.

Until then, result tables remain blank and the repository makes no trained-
model, operational-readiness, or checkpoint-availability claim.
