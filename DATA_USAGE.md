# Data usage

## Current repository state

The current rule baseline is not trained and requires no dataset. It processes text supplied at runtime. Repository examples and tests contain short synthetic sentences written for this project, including fixtures that validate source-grounded mention-qualifier cues.

The package does not retain, transmit, or log input text. Shell history, calling applications, notebooks, and deployment environments can have separate retention behavior.

No QA-SRL, QANom, OntoNotes, operational, raw, or processed dataset is currently committed to this repository. The challenge-set workbook is an empty annotation template, not an operational dataset.

## Planned research data

The neural study uses pinned downloads, checksum verification, safe extraction, strict adaptation, and deterministic preparation manifests for three public research archives:

| Dataset | Research role | Repository status | Redistribution status |
| --- | --- | --- | --- |
| QA-SRL Bank 2.1 | Verbal training predicates, role questions, and answer spans | Adapter implemented; every release split and layer processed successfully | No raw or processed records in Git; annotation and source-text redistribution unresolved |
| QA-SRL Gold Standard | Primary verbal development and test evaluation | Adapter and compatible scorer implemented; both verified splits processed successfully | No raw or processed records in Git; annotation and source-text redistribution unresolved |
| QANom | Nominal predicate detection, role questions, and answer spans | Adapter and reference scorer implemented; all three verified splits processed successfully | No raw or processed records in Git; annotation and source-text redistribution unresolved |
| Operational-style challenge set | Held-out support and operations constructions | Candidate pilot protocol, guide, rights template, and empty 20-record workbook prepared; no note accepted | Only newly authored or explicitly licensed text qualifies |

The [QA-SRL data card](docs/datasets/qa_srl.md) and [QANom data card](docs/datasets/qanom.md) record exact archive checksums, split statistics, scorer behavior, and unresolved rights.

Adaptation can be invoked directly on a local file, so an adapted manifest proves the exact input-file hash rather than independently proving that the file came from a pinned archive. The documented workflow verifies and safely extracts the pinned archive first. Future model-result bundles must add the exact code revision to the retained preparation fingerprint.

QA-SRL and QANom do not provide complete supervision for the project’s predicate-local mention qualifiers. Their adapted candidates therefore use `mention_qualifiers: null` to mean that qualifier assessment was not performed. The adapters do not write an empty list, which would incorrectly assert that assessment was performed and found no supported cue.

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

The evaluation set remains pending. The repository now contains candidate pilot materials:

- a versioned protocol;
- a predicate, role, and mention-qualifier annotation guide;
- a source-and-rights notice template; and
- a 20-record pilot workbook with no accepted note text or annotation.

These materials prepare the annotation process but do not constitute a collected, labeled, adjudicated, or frozen dataset. The source-and-rights notice is still a template: it does not cover any record and does not itself approve public release.

Any future challenge text can contain only:

- newly authored synthetic examples with documented authorship;
- public-domain text with recorded source evidence; or
- third-party text whose license explicitly permits the intended use and release.

The set excludes copied company tickets, private email, meeting transcripts, personal data, customer records, and school materials unless the repository records explicit authorization and a privacy review.

Each retained event mention can carry zero or more predicate-local qualifiers from the controlled kinds `negated`, `possible`, `necessary`, `planned`, `future`, `hypothetical`, `conditional`, `questioned`, and `reported`. Every populated qualifier must cite one or more exact source spans. In both the public and annotation schemas, `null` means qualifiers were not assessed, an empty list means assessment found no supported cue, and a populated list contains the assessed labels and their evidence. These annotations describe how the text frames a mention; they never establish truth, occurrence, completion, assignment, ownership, due dates, or execution status.

The candidate guide defines predicate eligibility, verbal and nominal types, role questions, answer boundaries, mention-qualifier cues, ambiguous cases, and exclusions. Its 20-record pilot is reserved for guide development and is excluded from reported challenge results. Ayo will annotate it twice after a 7–14 day washout, with shuffled order and without viewing the first pass or model output. That comparison can measure only intra-annotator repeatability.

No pilot note has been accepted, no second human annotator is available, and no independent annotation or disagreement adjudication has occurred. The proposed 50-record development and 150-record test sets have not been created, sealed, or used for model selection. E1 remains incomplete, and no challenge-set performance claim is available.

Before any scored challenge set can be called frozen, a second human must independently annotate every retained development and test record without seeing Ayo’s labels or model predictions; pre-resolution agreement and every disagreement resolution must be recorded; and the final text, labels, exclusions, splits, and fingerprints must be regenerated. Until then, no candidate artifact is gold, adjudicated, or frozen. Unless representative real operational text is available under appropriate terms, reports call the eventual set **operational-style** rather than evidence of operational-domain performance.

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
