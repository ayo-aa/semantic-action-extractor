# Model card

## Current repository status

The repository contains two distinct system components:

1. `rule-based-v0`, a runnable dependency-free action-extraction baseline;
2. `bert-srl-token-type`, a unit-tested construction and utility layer for a predicate-conditioned semantic-role model, not a complete trained pipeline.

The absence of a checkpoint is intentional. Model claims begin only after a public-data source is approved, the missing pipeline is implemented, and multi-seed training, evaluation, and redistribution review are complete.

## `rule-based-v0`

### Summary

`rule-based-v0` identifies configured English verbs and converts nearby text into source-grounded action records. It is deterministic software, not a trained statistical model.

### Version and dependencies

- Extractor ID: `rule-based-v0`
- Training data: none
- Runtime dependencies: Python standard library only
- Output: actor, predicate, patient, prepositional qualifiers, sentence index, source offsets, and a heuristic score

### Intended uses

- inspect the package schema and exact source grounding;
- prototype short-text workflows with human review;
- propose predicate candidates for the neural SRL stage;
- provide a transparent product-interface baseline, candidate proposer, and latency comparator;
- exercise the API and CLI without downloading a model.

### Important interpretation limits

The actor and patient fields are surface heuristics. They are not verified semantic roles. Prepositional qualifiers preserve observed wording without deciding whether a phrase is temporal, locative, instrumental, or something else.

The score reflects which heuristic fields were found. It is not calibrated and is not a probability of correctness.

### Known limitations

- English-specific vocabulary and inflection rules;
- best on short, active, declarative clauses;
- false positives from noun/verb/adjective ambiguity;
- weak coordination, embedding, and passive-voice handling;
- no reliable negation, modality, predicate sense, coreference, or implicit arguments;
- no corpus-level quality result.

## `bert-srl-token-type`

### Summary

The neural component targets the original homework formulation: given pre-split sentence words and one supplied predicate index, predict one PropBank BIO label per word. Current code aligns gold word labels to WordPieces and produces subword logits; prediction collapse back to one label per word remains pending.

The design tokenizes with `bert-base-uncased`, aligns word labels to WordPieces, sets BERT `token_type_ids` to 1 only on the first predicate WordPiece, passes the sequence through BERT, and applies one linear classifier to each contextual token state. Full fine-tuning is the reproduction target. Cross-entropy ignores special and padding positions.

### Status

- Architecture construction boundary: implemented; tested with injected fakes only
- Gold-label WordPiece alignment: implemented
- BIO repair and span decoding: implemented
- Subword-prediction collapse: not implemented
- Generic micro exact labeled-span scorer: implemented
- UP head/derived-span integration and per-role reporting: not implemented
- Public dataset adapter: not implemented
- Training and evaluation pipeline: not implemented
- Public training run: not started
- Public checkpoint: none
- Corpus-level public results: none

The construction boundary returns dictionary-compatible logits and optional loss. A future training pipeline must save a `state_dict` together with the base-model revision, label mapping, and configuration; pickling the runtime-defined full model object is not a supported checkpoint format. A real PyTorch/Transformers smoke test remains pending.

### Planned training data

Universal Proposition Bank 1.0 English EWT is the leading candidate pending feasibility and rights review. If adopted, the project will evaluate its annotated argument heads directly and may create deterministic silver spans through a versioned Universal Dependencies subtree expansion.

Restricted OntoNotes-derived course files and checkpoints are not included or used as public evidence.

### Intended uses

- research on supplied-predicate English semantic role labeling;
- controlled predicate-conditioning experiments;
- source-grounded role extraction with human review;
- comparison of gold-head and derived-span evaluation;
- a downstream component after separately evaluated predicate discovery.

### Out-of-scope uses

- treating `ARG0` and `ARG1` as universal actor and patient categories;
- claiming gold full-span evaluation from the UP head annotations;
- assuming a supplied-predicate score represents raw-text end-to-end quality;
- multilingual or cross-domain use without separate evaluation;
- autonomous decisions in employment, credit, health, legal, safety, or other consequential settings;
- processing sensitive text without application-level privacy controls.

### Evaluation contract

The only implemented metric is generic micro exact labeled-span precision, recall, and F1 over word-level BIO sequences, excluding supplied-predicate `V` and continuation `C-V` spans and reporting repaired prediction tags. Planned public evaluation will separate annotated heads from any declared derived BIO targets. Per-role results, candidate detection, token accuracy, latency, memory, and seed variation remain pending.

Token accuracy is diagnostic only because frequent `O` labels can conceal poor argument extraction.

### Risks and mitigations

- **Boundary uncertainty:** separate annotated heads from derived spans in data structures and reports.
- **Role overinterpretation:** preserve PropBank labels before any convenience mapping.
- **Predicate assumption:** report supplied-predicate and end-to-end evaluation independently.
- **Domain shift:** require representative authorized evaluation before a deployment claim.
- **Opaque errors:** retain source tokens, offsets, labels, and error categories for inspection.
- **Data rights:** keep datasets outside Git and review checkpoint redistribution separately.

## Results statement

Automated tests verify schema validation, exact offsets, baseline behavior, BIO transformations, gold-label alignment, model-construction behavior through injected fakes, and generic metric calculations. They do not establish extraction accuracy.

No trained model result, operational-readiness claim, or redistributable checkpoint is available in the current repository state.
