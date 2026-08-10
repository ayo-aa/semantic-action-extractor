# Model card

## Current status

The repository contains two separate components:

1. `rule-based-v0`, a runnable, dependency-free action-extraction baseline; and
2. `bert-srl-token-type`, implemented software for a predicate-conditioned BERT
   semantic-role model and paired ablation, with no repository-tracked or
   research-trained weights.

No prepared research data is tracked in the repository. A private Git-ignored
EWT dataset is prepared and fingerprinted, but no research-corpus training run,
model-quality result, or research checkpoint exists. A complete local rehearsal
did train six Git-ignored checkpoints on invented data, verify and reload them,
and benchmark one checkpoint per variant. This model card therefore describes
an intended experiment and implemented software boundary, not a released model.

## `rule-based-v0`

### Summary

`rule-based-v0` identifies configured English verbs and converts nearby text
into source-grounded action records. It is deterministic software and uses no
training data.

### Intended uses

- inspect the package schema and exact source offsets;
- prototype short-text workflows with human review;
- propose predicates for a future separately evaluated neural stage; and
- exercise the API and CLI without downloading a model.

### Limitations

- English-specific vocabulary and inflection rules;
- strongest on short active declarative clauses;
- weak coordination, embedding, passive voice, negation, and modality handling;
- no predicate sense, coreference, or implicit arguments;
- actor and patient are surface heuristics, not verified PropBank roles;
- its score is a ranking heuristic, not a calibrated probability; and
- no corpus-quality, latency, or memory result exists.

## `bert-srl-token-type`

### Intended task

Given pre-split English words and one supplied verbal predicate index, predict
one PropBank BIO label per source word. The supplied-predicate contract is not a
raw-text end-to-end contract; predicate discovery must be evaluated separately.

### Architecture

The experiment will use a revision-pinned BERT encoder compatible with two
token-type embeddings, plus one learned linear token-classification head. The
predicate-signal variant sets `token_type_id = 1` only on the first WordPiece of
the supplied predicate. The paired ablation explicitly supplies all-zero token-
type IDs. Both the encoder and classifier are trainable.

Gold labels are expanded across WordPieces, special and padding positions use
`-100`, and cross-entropy ignores those positions. Predictions collapse back to
one label per source word by selecting the first WordPiece. No input is silently
truncated: tokenized examples above the configured maximum are excluded with
explicit identifiers and counts.

### Implemented software boundary

| Component | Status |
| --- | --- |
| Canonical prepared dataset I/O and fingerprints | Implemented |
| Duplicate identity and cross-split leakage validation | Implemented |
| Immutable training-only label vocabulary | Implemented |
| Word/BIO-to-WordPiece alignment | Implemented |
| First-subword prediction collapse | Implemented |
| Explicit signal and all-zero ablation collation | Implemented |
| Revision-pinned BERT plus linear head construction | Implemented |
| Exact argument-span and per-role evaluation | Implemented |
| Token accuracy and supplied-predicate diagnostics | Implemented |
| Strict experiment configuration and run provenance | Implemented |
| SHA-256 integrity-checked checkpoint bundle | Implemented |
| Pinned PropBank/UD EWT adapter, aggregate gate, official splits, and duplicate/conflict policy | Implemented |
| Public-source private review with aggregate-only evidence | All 13 predicate-anchor divergences, one token-width mismatch, and 30/30 deterministic aligned records inspected; no user account or review required |
| BabySRL adapter and structural audit | Implemented historical fallback; not the active data path |
| AdamW training with optional linear warmup then constant LR | Implemented |
| Development-only checkpoint selection and one final test evaluation | Implemented |
| Three-seed paired predicate/no-predicate experiment aggregation | Implemented |
| Strict paired training CLI with ignored-output and exact-Git-revision checks | Implemented |
| Exact-match interruption recovery and per-output nonblocking lock | Implemented and independently durability-reviewed with injected failures and adversarial artifacts |
| Validated-checkpoint systems benchmark CLI and aggregate-only contract | Implemented; both synthetic checkpoint variants completed it |
| Full paired PyTorch/Transformers runtime rehearsal | Six invented-data seed/variant runs passed on Apple MPS |
| Prepared EWT training data | Completed privately in ignored storage: 31,101/3,775/3,610 examples; fingerprint `2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b` |
| Actual prepared-data MPS preflight | Predicate variant passed one full batch-32 optimizer step at longest retained sequence 118 with 111 labels |
| Research training/evaluation | Not run |
| Research checkpoint | None |

The implementation is exercised by synthetic tests, injected runtimes, and a
full optional-library rehearsal. Six paired runs loaded
`google-bert/bert-base-uncased` at revision
`86b5e0934494bd15c9632b12f734a8a67f723594`, optimized and checkpointed it,
verified and reloaded every checkpoint, published the paired result, and ran
the benchmark command against one checkpoint per variant on Apple MPS with
PyTorch 2.13.0 and Transformers 5.14.1. See the
[runtime record](reports/neural_runtime_smoke.md). This validates the execution
boundary only; it is not research-corpus training, an accuracy result, or
evidence of operational fitness.

### Training data status

Universal Proposition Bank 1.0 English EWT was rejected as a label source
because it projects roles to dependency heads. MASC remains rejected because
trace-only arguments keep optimistic exact-span conversion below the
predeclared 99% threshold.

The selected route instead pairs pinned [PropBank EWT gold skeletons](https://github.com/propbank/propbank-release/tree/4abade0b53ce4a181e1d98b3518101c1a44d395a/data/google/ewt)
with pinned [UD English EWT r2.2 words](https://github.com/UniversalDependencies/UD_English-EWT/tree/6e064999a75b9c941c515ce1be98352e6f9831e0).
The adapter and aggregate source gate are implemented. The corrected audit
finds 38,639 structurally valid verbal predicates and 38,635 word-aligned
instances. The
[aggregate-only private source review](reports/ewt_private_source_review.md)
inspected all 13 predicate-anchor divergences, the one token-width mismatch,
and 30/30 deterministically selected aligned verbal records. Leakage,
identical-input conflict, and evaluation-deduplication controls leave
31,101/3,775/3,610 train/development/test prepared examples. Private ignored
preparation at implementation commit
`9b7c94ec9be4a9b56c3cd7df3cb9a83b34b87f42` produced fingerprint
`2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b`.
At `max_length=128`, 62 train examples are overlength, leaving
31,039/3,775/3,610 modeled examples. The pinned tokenizer produced a
train-derived vocabulary of 111 labels, including `O` and continuation closure,
with no development or test label outside it. Split and provenance SHA-256
values are in the [preparation record](reports/ewt_preparation.md).

These are prepared-data and preflight facts, not training or model-quality
evidence. The public Git sources require no account, registration, CourseWorks
login, LDC download, or user-provided file. The join is a validated inferred
cross-release reconstruction, not PropBank's prescribed LDC mapping. Raw and
prepared text and future weights remain ignored and private; public artifacts
are limited to source-neutral code and non-reconstructive aggregate metrics
pending a separate weights review. BabySRL's earlier 99.2501% structural audit
is retained only as fallback evidence.

### Planned training protocol

The fixed experiment contract requires:

- exact model and tokenizer repository revisions, never mutable branch names;
- one prepared-data fingerprint shared by both variants;
- a training-derived immutable label order;
- AdamW, gradient clipping, optional linear warmup, then constant learning rate;
- deterministic batches and exactly three paired seeds;
- paired initial-state fingerprints and otherwise identical configurations;
- development-only checkpoint selection; and
- one test evaluation for each predeclared final seed/variant run.

The original anchor is batch size 32, learning rate `1e-5`, and two epochs.
They are now frozen in `configs/ewt_predicate_signal.toml` and
`configs/ewt_no_predicate_signal.toml` with canonical digests
`9bf8cd7c7a839bd9bfb6b39fde616f7e6f42d2ef163ea5f47b7afeeee1120bdc`
and `19ff94ff8bf839ee2fd5ebdab8ffed24b1ad7b243cc413a2ba687262f8f9d866`.
Both bind the prepared fingerprint. The predicate variant passed one actual-
data MPS step at batch size 32 and longest retained sequence 118 in 3.0069
seconds with 3,211,741,952 allocated bytes. This freezes the fit preflight; it
does not establish full-run duration, stability, or quality.

The installed `semantic-action-train-srl` command accepts the two strict paired
configs, prepared dataset, new Git-ignored output root, and exact 40-character
Git revision. It validates the data fingerprint before execution, stages all
six seed/variant runs, records a non-sensitive partial failure marker if needed,
retains a canonical complete journal, and atomically publishes canonical
results only after all runs complete. A per-output nonblocking lock prevents
two processes from writing the same run.

An interrupted study is resumed by rerunning the same command with only
`--resume` added. Resume requires identical partial-output, provenance,
configuration, prepared-data and fingerprint, Git-revision, and runtime
identity. It revalidates every completed result/checkpoint pair and paired seed
before reuse. Recovery is limited to exact writer-owned interrupted atomic
writes, next-checkpoint staging, and checkpoint-tombstone cleanup; lookalikes
and unknown artifacts are rejected. These are verified software durability
semantics, not evidence that paired research training or evaluation completed.

### Evaluation contract

The primary metric is micro-averaged exact labeled argument-span precision,
recall, and F1. Predicate `V` and `C-V` spans are excluded because the predicate
is supplied. No partial credit is given for a correct label with the wrong
boundary or a correct boundary with the wrong label.

Secondary outputs are per-role P/R/F1 and support, deterministic BIO-repair
counts, word-level token accuracy, and diagnostics for correct/missing predicate
anchors, spurious predicate labels, and predicted argument spans overlapping
the predicate. Token accuracy is diagnostic because frequent `O` labels can
hide poor role extraction.

Future raw-text reporting must separately measure predicate candidates and
downstream role frames. It may not reuse supplied-predicate F1 as an end-to-end
claim.

### Intended uses

- controlled research on supplied-predicate English SRL;
- a bounded predicate-conditioning ablation;
- source-grounded role extraction with human review after completed training;
- a downstream component after independently measured predicate discovery.

### Out-of-scope and prohibited interpretations

- treating `ARG0` and `ARG1` as universal actor and patient labels;
- claiming performance on raw text from a supplied-predicate result;
- assuming EWT's web genres represent operational domains;
- multilingual or cross-domain use without separate evaluation;
- autonomous decisions in consequential settings; and
- processing sensitive text without application-level privacy controls.

### Risks and mitigations

- **Boundary fidelity:** fail closed when a source proposition cannot fit one
  exact word-level BIO sequence; report every rejection reason.
- **Leakage:** freeze documents before outcomes, exclude exact text crossing
  splits, deduplicate evaluation semantics, and validate prepared manifests.
- **Role overinterpretation:** preserve PropBank labels before any convenience
  action mapping.
- **Predicate assumption:** keep controlled and raw-text metrics separate.
- **Domain shift:** require representative evaluation before any use claim
  outside EWT's web-text domains.
- **Reproducibility:** bind each run to data/config/Git fingerprints, package and
  hardware metadata, seeds, devices, counts, and initial model state.
- **Checkpoint integrity:** save labels and canonical metadata beside a state
  dictionary and verify its SHA-256 before loading.
- **Data rights and privacy:** keep raw/prepared data outside Git and perform a
  separate checkpoint-release and memorization review.

### Checkpoint and release status

There is no research checkpoint. Git-ignored synthetic rehearsal checkpoints
were created only to validate the round trip. The checkpoint code refuses
incompatible metadata, configuration, labels, or a changed state-dictionary
hash; that integrity contract does not grant redistribution rights.

Checkpoint redistribution remains on hold until the pinned PropBank and UD EWT
terms and underlying-text notices, intended use, encoder license, privacy and
memorization risks, required attribution, checkpoint license, and distribution
channel have all received an affirmative written review.

## Results statement

No research development score, test score, ablation effect, error-analysis
result, systems benchmark, operational-readiness claim, or redistributable
checkpoint is available. The values in the EWT audit and preparation record are
data-conversion, prepared-artifact, and single-step fit evidence only; synthetic
rehearsal metrics are intentionally not reported as model results.
