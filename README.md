# Semantic Action Extractor

## TL;DR

This project extracts source-grounded actions from English text and studies a
narrower research task: given a sentence and one supplied verbal predicate,
predict one word-level PropBank BIO label per word.

The repository now contains the complete, reproducible software path for that
controlled SRL experiment: strict dataset I/O, a training-only label
vocabulary, WordPiece alignment and first-subword prediction collapse,
predicate-conditioned BERT construction with immutable model revisions,
padding and overlength accounting, exact argument/per-role/token/predicate
evaluation, run provenance, integrity-checked checkpoints, and a paired
three-seed training engine with a verified exact-match resume path, per-output
nonblocking locking, and constant learning rate after optional warmup.

It does **not** contain a prepared research dataset, research-trained
checkpoint, or model-quality result. A complete six-run rehearsal on invented
data passed locally on Apple MPS, including checkpoint reload and aggregate
benchmark execution; those synthetic artifacts and scores are not portfolio
results. MASC was rejected. The selected no-registration route now joins pinned
public PropBank EWT gold skeletons to pinned public UD English EWT r2.2 words.
Its adapter, fail-closed gate, and leakage controls are implemented and pass.
The [aggregate-only private source review](reports/ewt_private_source_review.md)
inspected all 13 predicate-anchor divergences, the one token-width mismatch,
and 30/30 deterministically selected aligned records; real preparation and
training have not run.

## What the project measures

The controlled research contract is:

```text
(sentence words, supplied predicate index) -> one PropBank BIO tag per word
```

The raw-text product contract is broader:

```text
raw text -> predicate candidates -> supplied-predicate SRL -> optional action records
```

These are separate claims. A supplied-predicate SRL score does not measure
whether a raw-text system found the correct predicate, and candidate detection
cannot be credited for downstream argument labeling.

## Example product output

This invented example demonstrates the runnable rule baseline; it is not a
corpus excerpt or neural-model result.

Input:

```text
Maya emailed the signed contract to Jordan on Tuesday.
```

Abridged output:

```json
{
  "actions": [
    {
      "actor": {"text": "Maya", "start": 0, "end": 4},
      "predicate": {"text": "emailed", "start": 5, "end": 12},
      "predicate_lemma": "email",
      "patient": {"text": "the signed contract", "start": 13, "end": 32},
      "qualifiers": [
        {"relation": "to", "value": {"text": "Jordan", "start": 36, "end": 42}},
        {"relation": "on", "value": {"text": "Tuesday", "start": 46, "end": 53}}
      ]
    }
  ]
}
```

The action view is deliberately separate from the neural PropBank output.
`ARG0` and `ARG1` are predicate- and sense-relative roles, not universal
synonyms for actor and patient.

## System status

```mermaid
flowchart LR
    A["Raw text"] --> B["Rule predicate proposer<br/>implemented"]
    B -.->|planned orchestration| C["Sentence plus supplied predicate"]
    D["Pinned public PropBank plus UD EWT<br/>no account required"] --> E["EWT adapter and source gate passed<br/>13/13 anchors plus 30/30 review"]
    E -.->|next private run| G["Prepared split JSONL<br/>not generated"]
    G --> C
    C --> H["Training-only labels, WordPiece alignment,<br/>batching and explicit predicate signal"]
    H --> I["Pinned BERT plus linear token head"]
    I --> J["Paired three-seed trainer and CLI<br/>synthetic rehearsal passed"]
    J --> K["First-subword collapse and exact<br/>argument, role, token and predicate metrics"]
    J --> L["Integrity-checked checkpoint bundle<br/>synthetic round trip passed; no research checkpoint"]
```

Implemented software and current evidence are intentionally distinguished:

| Boundary | Repository status | Empirical status |
| --- | --- | --- |
| Rule action extractor | API and CLI implemented | No corpus-quality or systems benchmark |
| Prepared dataset contract | Canonical three-split JSONL, manifest fingerprints, exact duplicate/leakage checks | EWT adapter and aggregate gate pass; no prepared EWT files written |
| Label and alignment boundary | Train-only immutable vocabulary, BIO/WordPiece alignment, first-subword collapse | Synthetic tests and invented-data rehearsal passed |
| Model boundary | Full-fine-tuning BERT plus linear head; model and tokenizer revisions must be pinned by experiment config | Six-run invented-data rehearsal passed on MPS; no research-corpus training run |
| Evaluation | Exact micro argument span P/R/F1, per-role metrics, token accuracy, and supplied-predicate diagnostics | No development or test score |
| Training and ablation | Paired three-seed engine and strict CLI; exact-match resume, per-output nonblocking lock, and constant LR after warmup | All six seed/variant runs completed on invented data and the recovery boundary passed independent fault-injection review; research experiment not run |
| Systems measurement | Canonical aggregate p50/p95, throughput, peak-memory, and checkpoint-size contract | Both invented-data variant checkpoints completed the real benchmark path; no research measurements |
| Provenance and checkpoints | Canonical configuration/run metadata, complete run journal, and SHA-256 validation of serialized state | Synthetic checkpoint save/verify/reload and interrupted-write recovery passed; no research checkpoint and redistribution remains on hold |

## Data reconciliation

### Rejected: MASC PropBank

The [MASC PropBank release](https://anc.org/data/masc/downloads/data-download/)
was rejected for the fixed gold-span milestone. In the strict diagnostic slice,
unlinked trace-only arguments cap optimistic exact-span recovery at 93.25%,
below the predeclared 99% gate. The archive remains ignored audit evidence; it
was never prepared or used for training. See the
[MASC audit](reports/masc_propbank_audit.md).

### Selected: PropBank EWT skeletons plus UD English EWT r2.2

The selected source pairs the pinned [PropBank release](https://github.com/propbank/propbank-release/tree/4abade0b53ce4a181e1d98b3518101c1a44d395a)
gold skeletons with the pinned [UD English EWT r2.2 release](https://github.com/UniversalDependencies/UD_English-EWT/tree/6e064999a75b9c941c515ce1be98352e6f9831e0).
The implemented `semantic-action-prepare-ewt` adapter verifies both Git
revisions and relevant worktrees, joins normalized document identities and
sentence positions, reconstructs exact PropBank span columns, and fails closed
on structural drift.

The aggregate gate reports:

| Stage | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Structurally valid verbal predicates | — | — | — | 38,639 |
| Word-aligned after one four-predicate sentence exclusion | 31,174 | 3,806 | 3,655 | 38,635 |
| Prepared eligible after leakage, conflict, and eval-dedup controls | 31,101 | 3,775 | 3,610 | 38,486 |
| Modeled at `max_length=128` | 31,039 | 3,775 | 3,610 | 38,424 |

The pinned tokenizer preflight builds a train-derived vocabulary of 111 labels,
including `O` and continuation closure, with no development or test label
outside it. These are aggregate feasibility counts, not prepared files or model
results. The private aggregate-only source review checked all 13
metadata-versus-primary predicate-anchor divergences, the one token-width
mismatch, and 30/30 deterministically selected aligned verbal records. No
account, registration, CourseWorks login, LDC download, or user-supplied corpus
file is required.

This is a validated inferred cross-release join, not PropBank's prescribed LDC
mapping. UD licenses its annotations and database while expressly noting
separate copyrights in the underlying text. Raw sources, reconstructed and
prepared data, and future trained weights therefore remain ignored and private;
only source-neutral code and non-reconstructive aggregate metrics are public
pending a separate weights review. See the [EWT gate](docs/datasets/ewt_propbank_gate.md)
and [audit](reports/ewt_propbank_audit.md).

### Audited fallback only: BabySRL

[BabySRL](https://talkbank.org/childes/access/Derived/BabySRL.html) is a derived
version of the CHILDES Brown corpus containing PropBank-style verbal role
annotations for selected parental utterances. Its
[format description](https://cogcomp.seas.upenn.edu/Data/BabySRL.html)
documents surface role spans in CHAT files, which match the project's supplied-
predicate BIO target.

The pinned, read-only structural audit reports:

| Item | Frozen value |
| --- | ---: |
| CHAT documents | 133 |
| Declared proposition columns | 18,536 |
| Losslessly converted | 18,397 |
| Fail-closed rejections | 139 |
| Conversion coverage | 99.2501% |
| Final eligible train examples | 13,713 |
| Final eligible development examples | 1,356 |
| Final eligible test examples | 1,274 |

The child-stratified 133-document assignment is frozen in
[`docs/datasets/babysrl_split_manifest.json`](docs/datasets/babysrl_split_manifest.json)
with canonical SHA-256
`73ecae9f81d1d2b9f8495b13b297da9c3d24e387630e71a38ef2420d4c9a5de7`.
Exact sentence sequences crossing split boundaries are excluded from every
affected split; development and test then remove repeated identical semantic
examples. Documents are never moved after the freeze.

This remains a historical structural-feasibility result, not permission or
model quality. BabySRL is no longer the active path. Its TalkBank registration,
rules, and manual-review gates would apply only if the public EWT route later
failed and this fallback were explicitly activated. No BabySRL data was
prepared or used for training.

## Evaluation contract

The primary neural metric is micro-averaged exact labeled argument-span
precision, recall, and F1 over word-level BIO predictions. Predicate `V` and
`C-V` spans are excluded because the predicate is supplied. The evaluator also
reports per-role P/R/F1, repaired prediction tags, token accuracy, correct or
missing predicate anchors, spurious predicate labels, and predicted argument
spans overlapping the predicate.

Candidate-predicate recall and any raw-text end-to-end frame score remain
separate future evaluations. Token accuracy is diagnostic because frequent
`O` labels can conceal poor argument extraction.

## Research plan

| Study | Purpose | Status |
| --- | --- | --- |
| E0 — rule baseline | Runnable, source-grounded product interface and predicate proposer | Software implemented; quality and systems results pending |
| E1 — SRL/data foundation | Fixed BIO contract, data gates, adapter, splits, leakage controls, evaluation, provenance | Selected EWT adapter and gate pass; 13/13 anchor divergences, one width mismatch, and 30/30 deterministic review records inspected; ignored preparation not yet run |
| E2 — neural reproduction | Fine-tune the predicate-conditioned BERT model on the frozen prepared split | Invented-data runtime rehearsal passed; research run awaits EWT preparation and frozen configs |
| E3 — bounded ablation | Compare predicate signal with an otherwise identical no-signal run over three paired seeds | Six-run invented-data rehearsal passed; research experiment not run |
| E4 — analysis | Report errors, latency, throughput, memory, artifact size, and limitations | Both synthetic checkpoint variants completed aggregate benchmarking; research measurements not run |

## Scope and limitations

- EWT contains web genres including blogs, email, reviews, social material, and
  answers; results would not imply operational-domain readiness.
- The selected words-to-skeleton pairing is a validated inferred join across
  public releases, not the official LDC reconstruction prescribed by PropBank.
- The controlled model assumes a supplied verbal predicate. Raw-text predicate
  discovery is a separate error source.
- The system does not resolve implicit arguments, coreference, intent, task
  ownership, completion state, or legal/business meaning.
- The rule baseline is English-specific and works best on short active clauses.
- A future run would fine-tune a pretrained encoder; this project does not
  pretrain a foundation model.
- No trained checkpoint may be published until its separate redistribution,
  license, privacy, leakage, and model-card review is complete.

## Documentation

- [PROJECT_SPEC.md](PROJECT_SPEC.md) defines the research question, experiments,
  controls, and definition of done.
- [DATA_USAGE.md](DATA_USAGE.md) records the data and publication boundary.
- [MODEL_CARD.md](MODEL_CARD.md) documents implemented and untrained components.
- [docs/datasets/ewt_propbank_gate.md](docs/datasets/ewt_propbank_gate.md) records
  the selected no-registration route and publication boundary.
- [reports/ewt_propbank_audit.md](reports/ewt_propbank_audit.md) records the
  aggregate EWT alignment, conversion, split, and preflight evidence.
- [reports/ewt_private_source_review.md](reports/ewt_private_source_review.md)
  records the non-reconstructive aggregate result of the private source review.
- [reports/babysrl_audit.md](reports/babysrl_audit.md) preserves BabySRL's
  historical structural pass as fallback evidence.
- [reports/neural_runtime_smoke.md](reports/neural_runtime_smoke.md) records the
  six-run invented-data rehearsal without promoting synthetic scores.
- [reports/results.md](reports/results.md) keeps future model results blank.
- [reports/error_analysis.md](reports/error_analysis.md) preregisters analysis
  categories without claiming observations.

## Run the implemented software

Python 3.11 or newer is required.

Install and run the dependency-free rule baseline:

```bash
python -m pip install -e .
semantic-action-extractor --pretty "Maya emailed the signed contract to Jordan on Tuesday."
```

Run the dependency-free test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

With both exact ignored public checkouts present, the EWT command reproduces the
aggregate gate without writing prepared data:

```bash
semantic-action-prepare-ewt \
  data/raw/ewt_sources/propbank-release \
  data/raw/ewt_sources/UD_English-EWT
```

No account or manual user review is required. To perform the first real private
preparation, add `--output-directory data/processed/ewt`; the command writes
canonical splits, a manifest, and EWT provenance only inside the ignored data
boundary. That preparation has not yet been run.

The paired training command is implemented, but the placeholders below cannot
become valid frozen configurations until the EWT preparation supplies its
prepared-data fingerprint:

```bash
semantic-action-train-srl \
  --predicate-config path/to/predicate_signal.toml \
  --ablation-config path/to/no_predicate_signal.toml \
  --dataset path/to/ignored/prepared-dataset \
  --output-root path/to/new/ignored/run-directory \
  --git-revision 40-character-lowercase-commit
```

The command requires strict paired configurations, an exact prepared-data
fingerprint match, a new Git-ignored output directory, and an exact Git commit.
It stages six seed/variant runs, preserves a machine-readable partial failure
record if a run stops, retains a canonical journal after completion, and
atomically publishes the complete paired result. A per-output nonblocking lock
rejects a concurrent writer.

After an interruption, rerun the identical command with only `--resume` added:

```bash
semantic-action-train-srl \
  --predicate-config path/to/predicate_signal.toml \
  --ablation-config path/to/no_predicate_signal.toml \
  --dataset path/to/ignored/prepared-dataset \
  --output-root path/to/the/same/ignored/run-directory \
  --git-revision the-same-40-character-lowercase-commit \
  --resume
```

Resume is fail-closed. It requires the identical partial-run destination,
provenance, paired configurations, prepared dataset and fingerprint, Git
revision, and runtime identity. Before reusing work it validates each completed
result/checkpoint pair and each completed predicate/no-predicate seed pair. It
recovers only exact writer-owned interrupted atomic writes, next-checkpoint
staging, and checkpoint-tombstone cleanup; lookalike or unknown artifacts are
rejected rather than removed. These verified durability controls do not imply
that EWT preparation or research training has run.

After a checkpoint exists, the real benchmark command validates its metadata,
labels, state digest, config, and dataset before loading it. It measures fixed
single-example and eight-example batches and writes only canonical aggregates:

```bash
semantic-action-benchmark-srl \
  --config path/to/exact-variant.toml \
  --dataset path/to/ignored/prepared-dataset \
  --checkpoint path/to/checkpoint-bundle \
  --output path/to/new/ignored/benchmark.json
```

CUDA reports a resettable allocator peak. CPU and MPS report process-lifetime
peak RSS, including model load, because MPS has no resettable peak-memory
counter. The result records that method explicitly.

## License

Original repository code is MIT licensed. That license does not apply to
datasets, pretrained models, or checkpoints. Raw and prepared corpora remain
outside Git, and checkpoint redistribution is currently on hold.
