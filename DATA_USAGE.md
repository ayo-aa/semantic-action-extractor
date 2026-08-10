# Data usage

## Current disposition

No raw or prepared research dataset is committed to this repository. The EWT
corpus has now been prepared for model input in private Git-ignored storage at
fingerprint
`2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b`.
No research-corpus training has run, no model-quality result exists, and no
research checkpoint exists. A separate Git-ignored rehearsal used only
invented examples to validate training and checkpoint software; it creates no
corpus-use claim.

The repository includes independently written code, synthetic fixtures, source
revision identifiers, split-policy documentation, and non-reconstructive
aggregate audit and preparation evidence. Raw sources and the prepared JSONL
remain ignored by Git.

## Columbia course-data boundary

The original course exercise used course-provided OntoNotes-derived material
under Columbia and Linguistic Data Consortium teaching/research access. That
material is **not used** in this project. CourseWorks is not an acquisition or
authentication dependency.

The repository excludes:

- course train, development, and test files;
- labels, statistics, or examples derived from those files;
- notebook outputs, screenshots, grading fixtures, and course results;
- starter code, assignment prose, and course diagrams; and
- checkpoints trained on course data.

The course exercise informs only the independently reimplemented architecture.
Authorization to use Columbia resources does not by itself grant permission to
publish LDC-derived data or weights, and it does not replace a third-party
corpus's access terms.

## Dataset decisions

### Rejected target: Universal Proposition Bank 1.0 English EWT heads

[UP English EWT](https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT)
places roles on dependency heads rather than supplying the complete gold
argument spans required by the fixed BIO contract. Expanding dependency
subtrees would create heuristic targets and change the research claim. UP 1.0
is not used as the gold-label source; its documentation is used only to
corroborate the EWT release join described below.

### Rejected: MASC PropBank

The [MASC PropBank download](https://anc.org/data/masc/downloads/data-download/)
was inspected only through a read-only, aggregate feasibility audit. The
pinned local artifact SHA-256 is
`b7e89cfbb7a0b7caf3ba5076ac95ee80810834d3522678eefd488f166ddcc4df`.
It remains ignored under `data/raw/`.

MASC is rejected for the fixed milestone. In the strict diagnostic slice,
unlinked, unindexed trace-only arguments cap optimistic exact-span conversion
at 93.25%; the wider sensitivity ceiling is 92.85%. Both miss the predeclared
99% gate. Rights/lineage and join issues also remain unresolved. No MASC
adapter-specific preparation, split, training run, or checkpoint was created.
See the [gate record](docs/datasets/masc_propbank_gate.md) and
[audit](reports/masc_propbank_audit.md).

### Selected: PropBank EWT skeletons plus UD English EWT r2.2

The selected source is an inferred cross-release pairing of:

- [PropBank EWT gold skeletons](https://github.com/propbank/propbank-release/tree/4abade0b53ce4a181e1d98b3518101c1a44d395a/data/google/ewt) at commit `4abade0b53ce4a181e1d98b3518101c1a44d395a`; and
- [UD English EWT r2.2](https://github.com/UniversalDependencies/UD_English-EWT/tree/6e064999a75b9c941c515ce1be98352e6f9831e0) at commit `6e064999a75b9c941c515ce1be98352e6f9831e0`.

Both Git sources are anonymously readable. No account, registration,
CourseWorks session, LDC credential, or user-provided data file is required.
The implemented adapter verifies the exact commits and clean relevant paths,
joins normalized document identity and sentence order, decodes the gold span
columns, uses the unique primary `V` span start as `predicate_index`, and
applies the frozen exclusions before any write.

This is not PropBank's prescribed LDC mapping. The public PropBank skeletons
replace words with `[WORD]`; the corresponding words come from the pinned UD
release. The [aggregate audit](reports/ewt_propbank_audit.md) validates the join,
the [gate](docs/datasets/ewt_propbank_gate.md) records its limitations, and the
[aggregate-only private source review](reports/ewt_private_source_review.md)
records inspection of all 13 predicate-anchor divergences, the one token-width
mismatch, and 30/30 deterministically selected aligned verbal records. No
corpus excerpt, identifier, or review item is retained in Git.

The corrected gate counts are:

| Stage | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Structurally valid verbal predicates | — | — | — | 38,639 |
| Word-aligned predicates | 31,174 | 3,806 | 3,655 | 38,635 |
| Prepared eligible after all data controls | 31,101 | 3,775 | 3,610 | 38,486 |
| Modeled at `max_length=128` | 31,039 | 3,775 | 3,610 | 38,424 |

The pinned tokenizer preflight builds a train-derived vocabulary of 111 labels,
including `O` and continuation closure, and finds no development or test label
absent from it. The private preparation reproduces the 31,101/3,775/3,610
counts at fingerprint
`2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b`.
Its split-file and provenance SHA-256 values are recorded without corpus content
in [reports/ewt_preparation.md](reports/ewt_preparation.md). This is prepared-
data evidence, not evidence that a research model or result exists.

The UD EWT README licenses annotations and database rights under CC BY-SA 4.0
but expressly notes separate copyrights in the underlying texts. Raw source
checkouts, reconstructed text, prepared records, and trained weights therefore
stay ignored and private. Only source-neutral code and non-reconstructive
aggregate metrics are public pending a separate weights review.

### Audited fallback only: BabySRL

[BabySRL](https://talkbank.org/childes/access/Derived/BabySRL.html) is a derived
CHILDES Brown resource. Its
[annotation documentation](https://cogcomp.seas.upenn.edu/Data/BabySRL.html)
describes selected parental utterances with Penn Treebank-style parses and
PropBank-style verbal roles represented over surface tokens. The underlying
[Brown corpus page](https://talkbank.org/childes/access/Eng-NA/Brown.html)
records the source history, citation, and DOI.

Its historical frozen structural gate is:

| Outcome | Proposition columns | Share |
| --- | ---: | ---: |
| Losslessly converted | 18,397 | 99.2501% |
| Rejected fail-closed | 139 | 0.7499% |
| **Declared** | **18,536** | **100.0000%** |

The accepted and rejected counts reconcile exactly. This is historical
annotation-fit evidence, not permission, preparation, or model accuracy.
BabySRL is not the active source. Its TalkBank registration, current-rules, and
manual-review requirements would become relevant only if the EWT route were
explicitly abandoned and this fallback activated. No BabySRL data was prepared
or used for training.

## Frozen EWT split and leakage controls

The selected route uses PropBank's pinned official EWT document split lists.
After the single token-width mismatch is excluded, those splits contain
31,174/3,806/3,655 aligned train/development/test examples. The adapter then:

1. excludes all 76 examples in exact-word-sequence groups crossing splits;
2. excludes every example in an identical `(words, predicate_index)` group with
   conflicting tags: 29 train and two development examples;
3. collapses an identical-target duplicate only when it repeats the same
   source document/sentence/predicate identity (zero in the pinned sources);
4. preserves nonconflicting train frequency across distinct source identities;
   and
5. removes 10 development and 32 test repetitions with exact identical
   `(words, predicate_index, tags)` semantics across distinct source
   identities.

The resulting prepared-eligibility counts are 31,101 train, 3,775 development,
and 3,610 test. The separate `max_length=128` preflight excludes 62 train
examples, leaving 31,039/3,775/3,610 model inputs. Prepared dataset I/O adds
independent duplicate, identity, split, and exact-text checks. The 111-label
vocabulary, including `O` and continuation closure, is fitted from train only;
absent development/test labels fail closed, and the current preflight finds
none.

The frozen predicate configuration completed one actual prepared-data batch-32
forward/backward, gradient-clipping, AdamW, and scheduler step on MPS at longest
retained sequence 118. It passed in 3.0069 seconds with 3,211,741,952 allocated
bytes. This is model-input and optimizer-fit evidence only, not a research run,
score, or benchmark.

## Frozen preparation and future experiment records

The completed private preparation freezes:

- canonical source URLs and exact source Git commits;
- adapter implementation commit
  `9b7c94ec9be4a9b56c3cd7df3cb9a83b34b87f42`;
- official split and duplicate/conflict policies, prepared counts, exclusions,
  split-file SHA-256 values, provenance receipt SHA-256, and prepared-data
  fingerprint;
- the 111-label training-only order and exact base-model/tokenizer revisions;
  and
- paired config files `configs/ewt_predicate_signal.toml` and
  `configs/ewt_no_predicate_signal.toml`, with digests
  `9bf8cd7c7a839bd9bfb6b39fde616f7e6f42d2ef163ea5f47b7afeeee1120bdc`
  and `19ff94ff8bf839ee2fd5ebdab8ffed24b1ad7b243cc413a2ba687262f8f9d866`.

The future six-run study must add its exact Git revision, seed and variant,
timestamps, resolved device, hardware, package versions, retained/drop counts,
initial-state fingerprint, checkpoint state-dictionary SHA-256, and result-
artifact digest.

No preprocessing rule may be changed after inspecting test outcomes. The test
split is evaluated once for each predeclared final run after development-only
checkpoint selection.

The training CLI additionally refuses a data/config fingerprint mismatch, a
non-ignored destination, or a non-exact Git revision. A normal start requires a
new output destination and a per-output nonblocking lock excludes concurrent
writers. After an interruption, the recovery command is the identical training
command with only `--resume` added. Resume accepts only the same partial output,
provenance, configs, dataset and fingerprint, Git revision, and runtime
identity; it revalidates completed result/checkpoint pairs and paired seeds.
Only exact writer-owned atomic-write residue, next-checkpoint staging, and
checkpoint-tombstone cleanup can be recovered. Lookalike and unknown artifacts
are rejected rather than removed, and the canonical journal remains in the
completed output. These output guards are verified safety boundaries, not
empirical evidence. The no-registration EWT route is now prepared privately and
has passed a single actual-data MPS optimizer-step preflight. The paired six-run
study has not started.

## Model and artifact rights

Dataset access does not automatically authorize trained-weight publication.
Before any checkpoint release, review:

- the pinned PropBank and UD EWT licenses, underlying-text notices, and required
  citations;
- whether training and the intended use comply with all applicable source
  terms;
- the pretrained encoder and tokenizer licenses;
- whether transformed training data or weights create additional obligations;
- privacy and memorization/leakage risk; and
- the proposed checkpoint license, model card, and distribution channel.

Until that review reaches an affirmative written decision, publish only
source-neutral code and non-reconstructive aggregate metrics—not raw data,
prepared examples, corpus excerpts, or weights.

## Repository controls

`.gitignore` is a guardrail, not proof of compliance. Inspect every staged file
before committing. Raw or prepared data must never enter Git history even
temporarily. Tests and documentation may use only independently invented text,
not paraphrased or transformed corpus excerpts.

Every future score must name both source commits, the prepared fingerprint,
split and duplicate/conflict policies, conversion coverage, scorer, predicate
source, seed set, configuration, Git revision, hardware, and whether test
outcomes influenced development. Unit-test success and the EWT aggregate gate
are software and data-feasibility evidence only.
