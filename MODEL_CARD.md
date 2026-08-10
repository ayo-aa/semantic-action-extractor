# Model card

## Current status

The repository contains two separate components:

1. `rule-based-v0`, a runnable, dependency-free action-extraction baseline; and
2. `bert-srl-token-type`, implemented software for a predicate-conditioned BERT
   semantic-role model and paired ablation, with no repository-tracked or
   research-trained weights.

No prepared research data, research-corpus training run, model-quality result,
or research checkpoint exists. A complete local rehearsal did train six
Git-ignored checkpoints on invented data, verify and reload them, and benchmark
one checkpoint per variant. This model card therefore describes an intended
experiment and implemented software boundary, not a released model.

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
| BabySRL archive adapter, structural audit, split freeze, and duplicate policy | Implemented |
| Private prepared/raw manual-review workflow with aggregate-only decision | Implemented; not run on BabySRL |
| AdamW training with optional linear warmup then constant LR | Implemented |
| Development-only checkpoint selection and one final test evaluation | Implemented |
| Three-seed paired predicate/no-predicate experiment aggregation | Implemented |
| Strict paired training CLI with ignored-output and exact-Git-revision checks | Implemented |
| Validated-checkpoint systems benchmark CLI and aggregate-only contract | Implemented; both synthetic checkpoint variants completed it |
| Full paired PyTorch/Transformers runtime rehearsal | Six invented-data seed/variant runs passed on Apple MPS |
| Authorized prepared training data | None |
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

Universal Proposition Bank English EWT and MASC PropBank were rejected for the
fixed full-span target. MASC's trace-only arguments keep optimistic exact-span
conversion below the predeclared 99% threshold.

[BabySRL](https://talkbank.org/childes/access/Derived/BabySRL.html) is the
technical replacement candidate. Its
[annotation format](https://cogcomp.seas.upenn.edu/Data/BabySRL.html) matches
the overt verbal span target. The pinned adapter losslessly converts 18,397 of
18,536 declared propositions, or 99.2501%, and freezes 13,713/1,356/1,274
eligible train/development/test examples after duplicate controls. The
133-document assignment manifest SHA-256 is
`73ecae9f81d1d2b9f8495b13b297da9c3d24e387630e71a38ef2420d4c9a5de7`.

Those are aggregate feasibility counts, not prepared files and not training
evidence. The [CHILDES access requirements](https://talkbank.org/childes/access.html),
current [TalkBank ground rules](https://talkbank.org/0share/rules.html), and an
authorized privacy-preserving manual sample remain on hold. Registration and
rules acceptance must precede provisional ignored preparation; the resulting
prepared/raw pair must pass private review before training. CourseWorks and
Columbia course data are not used.

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
Those values are not yet a runnable checked-in neural configuration because an
authorized prepared-data fingerprint does not exist. Any hardware-driven change
must be declared before final outcomes.

The installed `semantic-action-train-srl` command accepts the two strict paired
configs, prepared dataset, new Git-ignored output root, and exact 40-character
Git revision. It validates the data fingerprint before execution, stages all
six seed/variant runs, records a non-sensitive partial failure marker if needed,
and atomically publishes canonical results only after all runs complete.

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
- source-grounded role extraction with human review after authorized training;
- a downstream component after independently measured predicate discovery.

### Out-of-scope and prohibited interpretations

- treating `ARG0` and `ARG1` as universal actor and patient labels;
- claiming performance on raw text from a supplied-predicate result;
- assuming child-directed parental speech represents operational domains;
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
- **Domain shift:** require representative authorized evaluation before any use
  claim outside BabySRL's child-directed-speech domain.
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

Checkpoint redistribution remains on hold until the then-current TalkBank and
source-corpus terms, intended use, encoder license, privacy and memorization
risks, required attribution, checkpoint license, and distribution channel have
all received an affirmative written review.

## Results statement

No research development score, test score, ablation effect, error-analysis
result, systems benchmark, operational-readiness claim, or redistributable
checkpoint is available. The values in the BabySRL audit are data-conversion
counts only; synthetic rehearsal metrics are intentionally not reported as
model results.
