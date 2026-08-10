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

## Dataset decision record

### Rejected: Universal Proposition Bank 1.0 English EWT

[UP 1.0 English EWT](https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT) is not a valid primary source for the restored word-level BIO-span milestone. Its English conversion places roles on dependency heads, and the [UP 2.0 paper](https://aclanthology.org/2022.lrec-1.181/) states that those heads are insufficient to recover full argument spans. Expanding dependency subtrees would create heuristic targets and a different research claim.

This decision does not judge UP 1.0 as a resource. It rejects a mismatch between that resource and this project's fixed span-SRL objective. UP 1.0 remains a possible source only for a separately declared dependency-head SRL project; that pivot is out of the current scope.

### Rejected: MASC PropBank

The [MASC-PROPBANK-ORIG download](https://anc.org/data/masc/downloads/data-download/) is described by the Open American National Corpus as an 88K-word MASC subset with original-format PropBank annotations and the Penn Treebank annotations they reference. The [MASC overview](https://anc.org/data/masc/) describes MASC as available for any purpose under the Creative Commons Attribution 3.0 United States license.

With explicit approval, the archive was fetched from ANC's artifact URL into ignored local storage under the transport caveat recorded in the audit report: the host presented an expired TLS certificate. Its locally pinned SHA-256 is `b7e89cfbb7a0b7caf3ba5076ac95ee80810834d3522678eefd488f166ddcc4df`. No source payload was written outside the ZIP or committed. The archive reports 88,530 words and bundles full source text, inline PTB parses, and `.prop` annotations, but contains no archive-level `LICENSE`, `COPYING`, or `NOTICE` file. It includes 36 WSJ files in each layer; a `wsj_0120` versus `wsj_0122` mismatch produces 37 unique `wsj_*` identifiers across the union.

ANC's corpus-wide CC BY statement is genuine, but the source families also carry publisher and LDC lineage. The 48-document non-WSJ manifest is source-family mapped and permitted only for read-only diagnostic analysis; G1 remains incomplete because an item-level attribution manifest was not finished. It is not cleared for training, corpus redistribution, or checkpoint release. The distinct WSJ slice remains denied unless ANC confirms its bundled text and annotation coverage in writing.

MASC was rejected at the feasibility stage. G2 remains on hold because the pinned manifest has missing layers and unresolved parse-wrapper cases. G3 independently fails: a strict diagnostic slice has a 93.25% optimistic exact-span ceiling, and a wider non-WSJ sensitivity audit caps exact recovery at 92.85%, both below the predeclared 99% threshold. No MASC adapter, split, prepared dataset, training run, or checkpoint may proceed under the fixed milestone. See the [gate record](docs/datasets/masc_propbank_gate.md) and [completed audit](reports/masc_propbank_audit.md).

The repository retains source-neutral pointer conversion and read-only aggregate audit code with wholly synthetic tests. The local archive is audit evidence only, remains ignored by Git, and is not a package dependency.

## Split and leakage controls

- Prefer an official source split when one is documented for the eventual approved corpus.
- Otherwise freeze a versioned, deterministic, document-disjoint, genre-aware split manifest before inspecting model outcomes.
- Preserve source document and sentence identifiers exactly.
- Keep every sentence and predicate instance from one document in one split.
- Create one SRL instance per eligible verbal predicate.
- Fit label inventories and other learned preprocessing on training data only.
- Freeze pointer conversion, role filtering, and exclusion rules before training.
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

- MASC, source-text, and contributed-annotation obligations;
- the pretrained encoder and tokenizer licenses;
- whether the transformed training data creates additional sharing requirements;
- the license and intended use of the resulting checkpoint.

If any right remains unresolved, publish code, configuration, aggregate metrics, and reproduction instructions only—not the dataset or checkpoint.

If MASC is ever reconsidered under a changed scope, the attribution record must name MASC and the American National Corpus project; retain supplied title and notices plus source-author, publisher, Penn Treebank, and PropBank contributor credits; link the exact archive source and [CC BY 3.0 US](https://creativecommons.org/licenses/by/3.0/us/); identify conversion and filtering changes; avoid implying ANC endorsement; and add no legal or technical restriction to redistributed licensed material. Preserve dated copies or fingerprints of the authoritative web evidence because the ZIP has no embedded archive-level license notice.

The current [PropBank release repository](https://github.com/propbank/propbank-release) is a separate CC BY-SA 4.0 distribution and is used here only as format documentation. Copying its rolesets or annotation data would require a separate provenance and license record; its license neither proves nor replaces the rights basis for the pinned ANC-hosted archive.

Wholly invented fixtures are publishable project code. Corpus-derived sentences, trees, role records, and prepared examples remain on hold unless separately cleared and attributed. Aggregate metrics may be published only in non-reconstructive form. Checkpoint publication remains conditional on the final training manifest, provenance review, model card, and memorization/leakage review.

## Repository controls

The `.gitignore` exclusions are guardrails, not proof of compliance. Inspect every staged file before a commit. Restricted or unreviewed data must never enter Git history temporarily.

## Evaluation disclosure

Every reported score must name the exact dataset release, split manifest, gold-span conversion policy, conversion coverage, scorer, role filtering, predicate source, seed set, and whether the test split influenced development. Unit-test success is software evidence only.
