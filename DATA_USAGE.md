# Data usage

## Current repository state

The current rule baseline is not trained and requires no dataset. It processes text supplied at runtime. Repository examples and tests contain short synthetic sentences written for this project.

The package does not retain, transmit, or log input text. Shell history, calling applications, notebooks, and deployment environments can have separate retention behavior.

No QA-SRL, QANom, OntoNotes, operational, raw, or processed dataset is currently committed to this repository.

## Planned research data

The neural study uses pinned downloads, checksum verification, safe extraction, strict adaptation, and deterministic preparation manifests for three public research archives:

| Dataset | Research role | Repository status | Redistribution status |
| --- | --- | --- | --- |
| QA-SRL Bank 2.1 | Verbal training predicates, role questions, and answer spans | Adapter implemented; every release split and layer processed successfully | No raw or processed records in Git; annotation and source-text redistribution unresolved |
| QA-SRL Gold Standard | Primary verbal development and test evaluation | Adapter and compatible scorer implemented; both verified splits processed successfully | No raw or processed records in Git; annotation and source-text redistribution unresolved |
| QANom | Nominal predicate detection, role questions, and answer spans | Adapter and reference scorer implemented; all three verified splits processed successfully | No raw or processed records in Git; annotation and source-text redistribution unresolved |
| Operational-style challenge set | Held-out support and operations constructions | Not collected | Only newly authored or explicitly licensed text qualifies |

The [QA-SRL data card](docs/datasets/qa_srl.md) and [QANom data card](docs/datasets/qanom.md) record exact archive checksums, split statistics, scorer behavior, and unresolved rights.

Adaptation can be invoked directly on a local file, so an adapted manifest proves the exact input-file hash rather than independently proving that the file came from a pinned archive. The documented workflow verifies and safely extracts the pinned archive first. Future model-result bundles must add the exact code revision to the retained preparation fingerprint.

An MIT license on a code repository does not by itself prove that every underlying source document carries the same redistribution terms. None of the verified data archives contains a separate license that resolves all annotation and source-text rights. The project therefore records dataset, source-text, derived-example, and checkpoint rights separately before it publishes data or weights.

## Restricted predecessor data

The predecessor notebook used course-provided, PropBank-style annotations derived from OntoNotes under Columbia and Linguistic Data Consortium access. The course material limits that data to authorized teaching or research use.

The public repository therefore excludes:

- the course training, development, and test files;
- role-list files derived from the restricted corpus;
- checkpoints trained on those files;
- extracted sentences, labels, screenshots, outputs, and grading fixtures.

The predecessor remains architectural lineage only. Its data and results do not provide public benchmark evidence for this project.

## Operational-style challenge set

The evaluation set is constructed, annotated, adjudicated, and frozen during the data-and-scorer milestone before model comparisons use it. It can contain only:

- newly authored synthetic examples with documented authorship;
- public-domain text with recorded source evidence; or
- third-party text whose license explicitly permits the intended use and release.

The set excludes copied company tickets, private email, meeting transcripts, personal data, customer records, and school materials unless the repository records explicit authorization and a privacy review.

Annotation documentation defines predicate eligibility, verbal and nominal types, role questions, answer boundaries, ambiguous cases, annotator disagreement, adjudication, and excluded examples. The test set remains frozen and unavailable for iterative model selection. Unless representative real operational text is available under appropriate terms, reports call this an **operational-style** set rather than evidence of operational-domain performance.

This set has not yet been authored, independently annotated, adjudicated, or frozen. No challenge-set performance claim is currently available.

## Cross-task split policy

The selected joint protocol treats QA-SRL Gold Standard and QANom development material as one development role, and their test material as one test role. Same-role overlap is intentional. Direct source and document identifiers do not cross training, development, and test, and no exact sentence crosses development and test.

An exact canonical-text comparison nevertheless found 11 evaluation sentences copied from training sources under different upstream IDs. They touch 16 QA-SRL Bank training records and seven QANom training records across seven training documents. The fixed `cross-role-document-quarantine-v1` policy leaves evaluation unchanged and excludes every training record in each affected document: 120 of 44,477 QA-SRL Bank expanded-training sentences and 28 of 7,114 QANom training sentences. The quarantine report records the triggering training and evaluation identities, exact text hashes, source-file hashes, policy version, and per-partition exclusion counts.

## Dataset-card requirements

Before an adapter produces a research result, its data card records:

- canonical name, release, source, and citation;
- annotation license and each source corpus’s applicable terms;
- access, modification, redistribution, and checkpoint-training rights;
- whether the repository can contain records, derived statistics, or preparation instructions only;
- collection and annotation process;
- language, domains, label representation, and known gaps;
- privacy, consent, safety, and representational risks;
- official split and document identifiers;
- duplicate, leakage, and held-out-predicate checks;
- raw and processed checksums;
- preprocessing code version and output fingerprint;
- deletion or access-control requirements.

If any redistribution right remains unclear, the repository contains only an adapter and preparation instructions. Raw and processed records remain outside Git.

## Model and artifact rights

Dataset access does not automatically grant permission to redistribute trained weights. Before publishing a checkpoint, the model card records the dataset terms, base-model license, tokenizer license, derived-weight conditions, intended uses, and prohibited uses.

Aggregate metrics and figures identify their exact dataset release, split, scorer, seeds, preprocessing revision, and Git commit. A result remains unpublished when those fields or the underlying rights are unresolved.

## Repository controls

The `.gitignore` excludes common data and model-artifact directories. The exclusion is a guardrail, not proof of compliance. Contributors inspect staged files before each commit and never add restricted content temporarily because removed files remain in Git history.

## Evaluation disclosure

Unit-test success is software evidence, not extraction-quality evidence. Every future score identifies the dataset release, license status, split, scorer, label filtering, predicate source, seeds, and whether the test set influenced development.
