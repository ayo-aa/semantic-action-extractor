# Data usage

## Repository boundary

No raw or processed research dataset is committed to this repository. The rule baseline is untrained and processes only text supplied at runtime. Tests and examples use short synthetic sentences written for this project.

The package itself does not retain, transmit, or log user input. Shell history, notebooks, calling applications, and deployment environments can have separate retention behavior.

## Restricted course data

The original homework used course-provided PropBank-style files derived from OntoNotes under Columbia and Linguistic Data Consortium access. Those materials were limited to authorized teaching or research use.

The public repository excludes:

- course train, development, and test files;
- label lists or statistics derived from the restricted files;
- sentences, annotations, screenshots, outputs, and grading fixtures;
- starter code, assignment prose, and course diagrams;
- checkpoints trained on the restricted files.

The restricted run is architectural background only. Its measurements are not presented as empirical results for this repository.

## Candidate public research source

The leading candidate, pending feasibility and rights review, is the frozen [Universal Proposition Bank 1.0 English EWT release](https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT), distributed as:

- `en_ewt-up-train.conllu`;
- `en_ewt-up-dev.conllu`;
- `en_ewt-up-test.conllu`.

Universal Proposition Bank 1.0 states a [CDLA-Sharing-1.0 license](https://github.com/UniversalPropositions/UP-1.0/blob/master/LICENSE). Its English documentation describes a merge of Universal Dependencies English EWT, the PropBank release, and English Web Treebank source material. The applicable lineage, licenses, permitted uses, and checkpoint-redistribution consequences must be resolved before final dataset selection or use.

The repository does not yet provide corpus parsing code or corpus-specific preparation instructions. If this source is adopted, users will obtain official files directly, record checksums, and keep raw and processed data outside Git.

## Annotation limitation

UP English EWT places argument labels on the heads of Universal Dependencies subtrees. Those heads are the public gold-supported target.

A planned adapter may expand a labeled head through a frozen dependency-subtree rule to produce a word-level BIO span. That expansion would be a deterministic project transformation, not a human-annotated gold span. Documentation and result tables must use terms such as **derived span** or **silver span**, record the adapter version, and never describe the measure as OntoNotes-equivalent exact-span performance.

## Split and leakage controls

- Preserve the official train, development, and test files.
- Preserve sentence and document identifiers from CoNLL-U comments.
- Keep all instances from one document in the same official split.
- Create one SRL instance per annotated predicate column.
- Fit label inventories and other learned preprocessing on training data only.
- Freeze the head-to-span conversion before training.
- Do not inspect test predictions while changing preprocessing, heuristics, or hyperparameters.
- Run duplicate and document-overlap checks before publishing a result.

## Local preparation record

Every empirical run must record:

- canonical source URL and retrieval date;
- source revision or release tag;
- SHA-256 of each downloaded file;
- adapter and configuration revision;
- output fingerprint and instance counts by split;
- dropped or malformed record counts with reasons;
- label inventory derived from training data;
- base model and tokenizer revisions;
- random seeds, hardware, and package versions.

## Model and artifact rights

Dataset access does not automatically grant permission to redistribute trained weights. Before publishing a checkpoint, verify and document:

- Universal Proposition Bank and Universal Dependencies obligations;
- the pretrained encoder and tokenizer licenses;
- whether the transformed training data creates additional sharing requirements;
- the license and intended use of the resulting checkpoint.

If any right remains unresolved, publish code, configuration, aggregate metrics, and reproduction instructions only—not the dataset or checkpoint.

## Repository controls

The `.gitignore` exclusions are guardrails, not proof of compliance. Inspect every staged file before a commit. Restricted or unreviewed data must never enter Git history temporarily.

## Evaluation disclosure

Every reported score must name the exact dataset release, split, annotation view (gold head or derived span), scorer, role filtering, predicate source, seed set, and whether the test split influenced development. Unit-test success is software evidence only.
