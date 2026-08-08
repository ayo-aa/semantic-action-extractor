# Provenance

## Predecessor notebook

A completed COMS W4705 BERT semantic-role-labeling notebook is one technical predecessor of this project. It motivates several engineering questions but does not define the public task, schema, data, or result claims.

The notebook explored:

- PropBank-style predicate and argument extraction;
- alignment between word-level BIO labels and BERT WordPieces;
- conditioning an encoder on a supplied predicate position;
- fine-tuning a contextual encoder with a token classifier;
- decoding labels and evaluating exact labeled spans.

The public research target extends beyond that notebook. It studies verbal and nominal predicates through QA-SRL and QANom, adds constrained seven-slot question prediction, compares multiple predicate-conditioning methods, reproduces a generative QASem baseline, and evaluates the complete predicate-to-argument pipeline.

## Newly written public implementation

The repository code is newly written for this project. The initial milestones add:

- a versioned, role-neutral predicate–argument schema with exact source offsets;
- a dependency-free rule baseline;
- an installable Python package and CLI;
- TOML configuration;
- automated tests and CI;
- product, research, data-use, provenance, and model documentation.

The rule baseline does not copy notebook code and does not reproduce the notebook’s recorded predictions.

## Materials intentionally excluded

- instructor-authored assignment prose and TODO scaffolding;
- starter model and metric code;
- the assignment architecture image;
- course-provided OntoNotes or PropBank files and access paths;
- notebook outputs and training logs;
- trained weights and checkpoints;
- course grading examples and fixtures.

The predecessor notebook interleaves starter material and student completions. Publishing the notebook wholesale would obscure authorship and conflict with the course dataset’s access boundary.

## Public research implementation policy

Each neural module is reviewed as new project code. A pull request that implements a predecessor concept records:

1. the concept it implements;
2. whether any exact source text or code was reused;
3. the license or written permission for any reused material;
4. dataset, base-model, tokenizer, and checkpoint lineage;
5. tests for alignment, decoding, truncation, and source-offset failure modes.

The first planned reproduction uses the QA-SRL Bank 2.1 training release and QA-SRL Gold Standard evaluation data rather than the predecessor’s restricted OntoNotes-derived course data. QANom then extends the study to nominal predicates.

## External research lineage

The project’s main public research foundations are:

- [Large-Scale QA-SRL Parsing](https://aclanthology.org/P18-1191/) for verbal QA-SRL data, argument detection, and structured question generation;
- [QANom](https://aclanthology.org/2020.coling-main.274/) for nominal candidate classification and QA-based argument representation;
- [QASem Parsing](https://aclanthology.org/2022.emnlp-main.528/) for unified text-to-text QA-based parsing, joint verbal/nominal learning, predicate markers, and domain-transfer evidence.

QASem already demonstrated that one generative model can parse verbal and nominal predicates. The new study does not claim that result as its contribution. It reproduces T5-small QASem as a required comparison and studies a source-constrained encoder that detects answer spans, predicts the seven question slots, and guarantees that serialized answers come from the input. The controlled extensions are token-type versus marker versus learned-feature predicate conditioning, exact-grounding and calibration behavior, unseen predicate-family transfer, operational-style evaluation, and matched systems measurements.

The earlier BERT notebook motivates the encoder implementation, but its PropBank BIO labels are not treated as if they could directly produce QA-SRL questions. The new model separately learns answer spans and structured question slots.

The repository cites and audits each implementation or dataset separately. A related paper’s publication license does not grant rights to every dataset, model, or source document associated with it.

## Git-history policy

The public repository uses fresh history because the local course download contains no meaningful Git provenance and mixes multiple authors in one notebook. The history reflects the actual reengineering work and does not backdate research claims.

If a selective import later receives permission, it enters through one clearly labeled commit with a `NOTICE` update. Restricted artifacts never enter Git temporarily because deleting a file does not remove it from prior commits.
