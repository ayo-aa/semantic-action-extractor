# Provenance

## Direct prototype lineage

This project directly refactors and reengineers a completed COMS W4705 BERT semantic-role-labeling notebook. The notebook is the technical prototype for this repository, not an unrelated or quarantined artifact.

The prototype established the direction for:

- PropBank-style predicate/argument extraction;
- aligning word-level BIO labels with BERT WordPieces;
- conditioning an encoder on a supplied predicate position;
- fine-tuning a contextual encoder with a token classifier;
- decoding labels and evaluating exact labeled spans.

Those concepts define the neural roadmap and public schema here.

## What is new in milestone one

All code currently committed to this scaffold is newly written for this project. Milestone one adds capabilities the notebook did not provide as a reusable product:

- a stable, typed action-frame schema with source offsets;
- a dependency-free rule baseline;
- an installable Python package and CLI;
- TOML configuration;
- automated tests and CI;
- product, research, data-use, and model documentation.

The rule baseline does not reuse notebook code and does not reproduce the notebook's recorded predictions.

## Materials intentionally not imported

- instructor-authored assignment prose and TODO scaffolding;
- starter model and metric code;
- the assignment architecture image;
- course-provided OntoNotes/PropBank files or their download location;
- notebook outputs and training logs;
- trained weights or checkpoints;
- course grading examples or fixtures.

The original notebook interleaves starter material and student completions, so copying the notebook wholesale would obscure authorship and create academic-integrity and licensing problems.

## Future neural refactor policy

Neural modules may directly implement and improve the prototype's design, but each module must be reviewed as new project code. Before exact source lines or third-party figures are reused, maintainers must obtain and record permission or a compatible license.

Each future neural pull request should document:

1. which prototype concept it implements;
2. whether any exact source text was reused;
3. the license or written permission for any reused material;
4. the dataset and checkpoint lineage;
5. tests added for known prototype failure modes.

## Git-history policy

The public repository starts with fresh history because the local course download contains no meaningful Git provenance and mixes multiple authors in one notebook. History should reflect the actual reengineering work; it must not be backdated or presented as original research completed before this repository existed.

If permission for a selective import is later obtained, it should enter through one clearly labeled commit with a `NOTICE` update. Restricted artifacts must never be added temporarily and then removed, because they would remain in Git history.
