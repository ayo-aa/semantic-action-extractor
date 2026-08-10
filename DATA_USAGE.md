# Data usage

## Current disposition

No raw or prepared research dataset is committed to this repository. No corpus
has been prepared for model input, no neural training has run, no model-quality
result exists, and no checkpoint exists.

The repository includes independently written code, synthetic fixtures,
corpus-free split assignments, archive fingerprints, and non-reconstructive
aggregate audit counts. Raw archives and any future prepared JSONL remain
ignored by Git.

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

### Rejected: Universal Proposition Bank English EWT

[UP English EWT](https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT)
places roles on dependency heads rather than supplying the complete gold
argument spans required by the fixed BIO contract. Expanding dependency
subtrees would create heuristic targets and change the research claim. It is
not used.

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

### Technical pass, use hold: BabySRL

[BabySRL](https://talkbank.org/childes/access/Derived/BabySRL.html) is a derived
CHILDES Brown resource. Its
[annotation documentation](https://cogcomp.seas.upenn.edu/Data/BabySRL.html)
describes selected parental utterances with Penn Treebank-style parses and
PropBank-style verbal roles represented over surface tokens. The underlying
[Brown corpus page](https://talkbank.org/childes/access/Eng-NA/Brown.html)
records the source history, citation, and DOI.

The locally pinned archive has SHA-256
`a2d8d38b0818910d05154cb62adcb05ef36b00f0aeae2690b7e1677fd5154604`.
The adapter validates the exact digest and size, ZIP integrity, member set, and
safe paths before reading CHAT members in memory. It never extracts members
during the audit.

The frozen structural gate is:

| Outcome | Proposition columns | Share |
| --- | ---: | ---: |
| Losslessly converted | 18,397 | 99.2501% |
| Rejected fail-closed | 139 | 0.7499% |
| **Declared** | **18,536** | **100.0000%** |

The accepted and rejected counts reconcile exactly. This is an annotation-fit
result, not permission to prepare or train and not model accuracy.

The [CHILDES access page](https://talkbank.org/childes/access.html) identifies
registration requirements. The current
[TalkBank ground rules](https://talkbank.org/0share/rules.html) govern use even
when a particular derived archive is directly reachable. At the audit date,
those rules state a default CC BY-NC-SA 3.0 basis unless otherwise indicated,
exclude incorporation into commercial products including model systems,
require non-storage assurances for web processing, and impose ethics and
confidentiality duties.

The following remain on hold:

| Requirement | Status |
| --- | --- |
| TalkBank/CHILDES registration | HOLD; confirmation not recorded |
| Acceptance of the current access conditions and ground rules | HOLD; acceptance date not recorded |
| Provisional ignored preparation | BLOCKED until the two access confirmations above are recorded |
| Authorized privacy-preserving manual conversion sample | HOLD; not performed |
| Prepared training data | Not created |
| Local or approved Columbia training | Not started |
| Third-party web/cloud training | HOLD pending an enforceable no-storage basis |
| Checkpoint redistribution | HOLD pending a separate written review |

No credentials belong in the repository or this record. The required TalkBank
step is separate from CourseWorks and from Columbia course-data access.
After access confirmation, the
[private manual-review workflow](docs/datasets/babysrl_manual_review.md)
requires an exact prepared/raw identity match and keeps all selected source
fields under ignored `data/review/`; only aggregate decisions may leave that
boundary.

## Frozen split and leakage controls

The corpus-free assignment manifest covers all 133 BabySRL documents: 106
train, 13 development, and 14 test. Its canonical SHA-256 is
`73ecae9f81d1d2b9f8495b13b297da9c3d24e387630e71a38ef2420d4c9a5de7`.
Documents are assigned by a child-stratified SHA-256 policy before outcomes and
are never moved later.

After that assignment, the adapter excludes every occurrence of an exact
sentence-token sequence found in more than one split. It retains remaining
training frequency and deduplicates identical semantic fingerprints within
development and test. It rejects conflicting annotations for the same source
sentence/predicate identity.

The safe final eligibility counts from the in-memory audit are:

| Train | Development | Test | Total |
| ---: | ---: | ---: | ---: |
| 13,713 | 1,356 | 1,274 | 16,343 |

Prepared dataset I/O adds independent checks for duplicate IDs and semantic
identities, document and sentence identifiers crossing splits, and exact
sentence text crossing splits. The label vocabulary is fitted on training data
only. Development or test labels absent from that vocabulary fail closed.

## Preparation and experiment records

After access confirmation permits provisional ignored preparation, and before
any empirical run, the project must record:

- canonical source URL, retrieval date, archive size, and SHA-256;
- adapter revision, split-manifest digest, duplicate policy, prepared counts,
  exclusions, and prepared-data fingerprint;
- training-only label order;
- exact base-model and tokenizer repository revisions;
- complete configuration digest, Git revision, seed and variant;
- timestamps, resolved device, hardware, package versions, retained/drop
  counts, and initial-state fingerprint; and
- checkpoint state-dictionary SHA-256 and result-artifact digest.

No preprocessing rule may be changed after inspecting test outcomes. The test
split is evaluated once for each predeclared final run after development-only
checkpoint selection.

The training CLI additionally refuses a data/config fingerprint mismatch, an
existing output destination, a destination not covered by Git ignore rules, or
a non-exact Git revision. Its output guard is a safety boundary, not permission
to train: access confirmation must precede provisional preparation, and the
private manual review must reach `pass` before training.

## Model and artifact rights

Dataset access does not automatically authorize trained-weight publication.
Before any checkpoint release, review:

- the then-current BabySRL, CHILDES, Brown, and TalkBank terms and required
  citations;
- whether training and the intended use comply with the non-commercial and
  web-processing restrictions;
- the pretrained encoder and tokenizer licenses;
- whether transformed training data or weights create additional obligations;
- privacy and memorization/leakage risk; and
- the proposed checkpoint license, model card, and distribution channel.

Until that review reaches an affirmative written decision, publish only code,
configuration schemas, non-reconstructive aggregate metrics, and reproduction
instructions—not raw data, prepared examples, corpus excerpts, or weights.

## Repository controls

`.gitignore` is a guardrail, not proof of compliance. Inspect every staged file
before committing. Raw or prepared data must never enter Git history even
temporarily. Tests and documentation may use only independently invented text,
not paraphrased or transformed corpus excerpts.

Every future score must name the data pin, prepared fingerprint, split and
duplicate policies, conversion coverage, scorer, predicate source, seed set,
configuration, Git revision, hardware, and whether test outcomes influenced
development. Unit-test success and the 99.2501% structural gate are software
and data-feasibility evidence only.
