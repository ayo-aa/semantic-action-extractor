# Data usage

## Current milestone

The rule baseline is not trained and requires no dataset. It processes text supplied by the user at runtime. Repository examples and test fixtures are short synthetic sentences written for this project.

No user input is retained, transmitted, or logged by the package itself. Shell history, calling applications, and deployment environments may have their own retention behavior.

## Restricted prototype data

The prototype used course-provided, PropBank-style annotations derived from OntoNotes under Columbia/Linguistic Data Consortium access. The course material explicitly limited that data to authorized teaching or research use.

This repository therefore does not include or link to:

- the course train, development, or test files;
- role-list files derived from the restricted corpus;
- a checkpoint trained on those files;
- extracted sentences, labels, screenshots, or grading fixtures.

The MIT license in this repository does not apply to any external dataset.

## Requirements for a future dataset

Before adding a dataset adapter or publishing results, record the following in a dataset-specific file under `docs/datasets/`:

- canonical name, version, source, and citation;
- license and terms for access, modification, and redistribution;
- whether only instructions, derived statistics, or actual records may be committed;
- collection and annotation process;
- language, domains, label inventory, and known gaps;
- PII, consent, safety, and representational risks;
- deterministic split procedure and leakage checks;
- checksums for locally prepared artifacts;
- preprocessing code version;
- deletion or access-control requirements.

If redistribution rights are unclear, commit only an adapter and preparation instructions. Keep raw and processed records outside Git.

## Repository controls

The `.gitignore` excludes common dataset and model-artifact directories. This is a guardrail, not proof of compliance. Contributors must inspect staged files before every commit and must never add restricted content merely to remove it in a later commit.

## Evaluation disclosure

Unit-test success is not model-quality evidence. Any future score must identify the exact dataset, split, scorer, label filtering, number of seeds, and whether the test set was examined during development.
