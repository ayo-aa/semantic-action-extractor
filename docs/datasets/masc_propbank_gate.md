# MASC PropBank feasibility gate

Status: **adapter-core go; corpus preparation, training, and release hold**

Decision date: 2026-08-09

## Reconciliation decision

The restored research claim is fixed:

```text
(sentence words, supplied verbal predicate index) -> one gold PropBank BIO tag per word
```

Universal Proposition Bank 1.0 English EWT does not satisfy that contract. Its argument annotations identify dependency heads, not full argument spans. The UP 2.0 authors explicitly state that the UP 1.0 heads are insufficient to recover spans. A dependency-subtree expansion would therefore create a different, silver-target experiment and reopen the scope fork this reconciliation is meant to close.

MASC PropBank is the next candidate because the Open American National Corpus advertises an 88K-word download containing original PropBank pointer annotations together with the Penn Treebank parses they reference. Those constituent pointers can, in principle, support exact surface-span BIO labels.

This is a candidate decision, not a training-dataset adoption decision. A metadata-only range inspection read the official ZIP directory, README, and selected MASC headers; it did not acquire text, PTB, or PropBank payloads. No corpus record has been parsed, trained on, or committed.

## Candidate comparison

| Source | Span fit | Access and rights fit | Decision |
| --- | --- | --- | --- |
| UP 1.0 English EWT | Dependency heads only; exact gold spans cannot be recovered | Public, but layered lineage and license signals require care | Reject for this milestone |
| MASC-PROPBANK-ORIG | Original PropBank constituent pointers plus referenced PTB parses | MASC is advertised under CC BY 3.0 US; bundled source-family provenance still needs classification | Proceed with source-neutral adapter core; hold corpus use |
| OntoNotes / CoNLL-2012 | Strong task match | Source text requires LDC access and redistribution is restricted | Do not use for the public portfolio reproduction |
| CoNLL-2005 | Closest classic span-SRL benchmark | Depends on licensed Treebank material and is not a clean redistributable portfolio source | Do not use for the public portfolio reproduction |

## Authoritative evidence

- [MASC data download](https://anc.org/data/masc/downloads/data-download/) describes `MASC-PROPBANK-ORIG` as an 88K-word subset with original-format PropBank annotations and the Penn Treebank annotations on which they rely.
- [MASC overview](https://anc.org/data/masc/) says MASC can be downloaded for any purpose under Creative Commons Attribution 3.0 United States.
- [MASC corpus description](https://anc.org/data/masc/corpus/) says the broader 500K corpus contains a 5.5K-word Wall Street Journal slice. Metadata inspection confirms that the PropBank ZIP contains 36 WSJ files per text, PTB, and PropBank layer, with 37 unique WSJ identifiers across the union because one layer names `wsj_0120` where another names `wsj_0122`.
- [Official PropBank release](https://github.com/propbank/propbank-release) documents the stand-off `.prop` format and the need for corresponding parse trees to recover text spans. Its current CC BY-SA 4.0 distribution is used here for format documentation only and does not establish the license of the separate 2014 ANC archive.
- [PropBank I release notes](https://catalog.ldc.upenn.edu/docs/LDC2004T14/readme.txt) define classic terminal-and-height tree pointers, including empty terminals and comma/`*` composite pointers.
- [English PropBank data-format notes](https://verbs.colorado.edu/~mpalmer/projects/ace/EPB-data-format.txt) document the later record dialect and link/concatenation conventions that the archive audit must distinguish from the classic format.
- [UP 2.0 paper](https://aclanthology.org/2022.lrec-1.181/) records the blocking UP 1.0 head-to-span limitation.

## Gate sequence

The gates are ordered. A failure pauses the sequence; later work does not erase an earlier failure.

### G1 — rights and lineage

Record, from the archive and authoritative MASC documentation:

- canonical acquisition URL, archive filename, release identifier, retrieval date, and SHA-256;
- the license and required attribution for source text, PTB parses, and PropBank annotations;
- every document collection represented in the 88K subset;
- the rights basis for every retained source family;
- exclusion of every `wsj_*` item unless ANC confirms the bundled WSJ text and annotation coverage in writing;
- publication class for each artifact: adapter code and wholly synthetic fixtures; corpus-derived examples; non-reconstructive aggregate metrics; and trained weights.

The archive itself contains only a README, not a legal notice. ANC calls the selected Language Understanding subset license-free and says restricted LU texts were excluded, so an LDC reference in a header is not by itself a reason to reject a document. Conversely, lack of a `wsj_` prefix is not proof of acceptable provenance. A pinned, source-family-reviewed allowlist is the primary control; anchored `wsj_*` denial is a fail-closed backstop applied independently to all layers.

Pass condition: every included layer and source family has a documented use basis and attribution plan. If trained-weight redistribution is unclear, code and aggregate results may proceed locally but checkpoint publication remains blocked.

### G2 — archive and join audit

Inventory the archive without modifying its contents:

- all `.prop`, parse, text, metadata, readme, and license files;
- normalized document identifiers and sentence counts;
- PropBank rows whose document or sentence cannot be joined to exactly one parse;
- malformed rows, duplicate predicate instances, missing parses, and empty sentences;
- stable document and genre metadata suitable for leakage-safe splitting.

The audit must explicitly reproduce the known `wsj_0120`/`wsj_0122` layer mismatch rather than aligning WSJ files by count. It must also ignore legacy `.svn` debris.

Pass condition: all retained documents and predicate rows join deterministically. Every exclusion has a machine-readable reason and count; nothing is silently repaired or dropped.

### G3 — annotation-fit audit

Before writing the corpus adapter, inventory:

- predicate Penn tags and rolesets;
- argument labels and supports;
- classic versus later fixed-field record dialects;
- simple, comma/semicolon concatenation, and `*` trace-chain pointer operators;
- trace-only nodes and co-indexed overt constituents;
- explicit `C-*`, `R-*`, `LINK-*`, and multiword-predicate structures;
- overlapping arguments and other cases that cannot fit one BIO label per word.

The restored verbal subset is limited to predicate terminals tagged `VB`, `VBD`, `VBG`, `VBN`, `VBP`, or `VBZ`. The roleset is retained as metadata but is not a model input. Nonverbal predicates and modal `MD` terminals are excluded with counts.

Pass condition: a documented conversion policy covers every observed pointer and label pattern. At least 99% of otherwise eligible verbal instances must convert losslessly to the fixed single-label surface BIO representation. Falling below that threshold triggers a design review or candidate rejection, not a relaxed claim.

### G4 — conversion verification

The conversion boundary must:

- detect and preserve the archive's actual `.prop` record dialect instead of inferring it from a synthetic example;
- preserve the raw inflection field because published descriptions of its character order are inconsistent;
- count PTB terminals exactly as the pointer format defines, including empty elements during pointer resolution;
- omit `-NONE-` terminals from the model's surface-word sequence;
- retain an explicit parse-terminal-to-surface-word mapping;
- preserve exact source document, sentence, predicate, roleset, pointer, and role metadata;
- preserve explicit continuation and reference labels rather than collapsing them into core roles, require each continuation to follow its base span, and never invent `C-*` or `R-*` labels that are absent from the source;
- treat `LINK-*` records as metadata rather than BIO roles, require each link to share at least one source node with exactly one semantic argument, and do not infer argument spans or reference direction from a link until a corpus-specific rule is verified;
- keep discontinuous fragments discontinuous rather than filling their gaps;
- produce one supplied-predicate instance per eligible verbal predicate;
- reject the whole predicate instance when any gold argument is unresolved, overlapping, conflicting, or otherwise unrepresentable; dropping only the difficult argument would create false gold `O` labels;
- produce a validated unsplit example during conversion and attach train/development/test only after G5 freezes whole-document assignments;
- close the future WordPiece label inventory under continuation tags, including `I-C-V` when a `B-C-V` word splits into multiple pieces;
- emit deterministic BIO labels and an auditable conversion report.

The current source-neutral core implements the documented record/tree parsing, pointer resolution, fail-closed conversion, unsplit prepared record, stable error reasons, default `wsj_*` denial, external source-policy hook, and wholly invented fixtures. It retains the full raw PropBank record in conversion provenance and keeps LINK semantics metadata-only. It intentionally does not implement MASC file discovery, archive joins, source-family allowlists, audit reports, or split generation.

Verification of the future MASC layer uses the existing invented fixtures plus a manually checked source sample stratified across genres and every observed pointer/operator class. Corpus examples do not enter Git.

Pass condition: synthetic tests pass, the manual audit finds no boundary or role mismatch, and the conversion report reconciles input, output, exclusion, and error counts.

### G5 — split freeze

No official MASC PropBank train/development/test split has been established. After G4 passes, commit a manifest that assigns whole documents—not sentences or predicates—to deterministic, genre-aware train, development, and test partitions.

The manifest must record document ID, genre, split, source checksum, and eligible-predicate count. The allocator, seed, and target proportions are committed before the first model outcome is inspected, and only then are prepared examples converted to split-bound model records. The manifest is immutable for the declared experiment; any later split is a new experiment.

Pass condition: document overlap and duplicate-text checks are zero, and each viable genre is represented without breaking document boundaries.

## Repository boundary during the hold

Allowed now:

- this decision record and source-neutral SRL primitives;
- synthetic tests that do not reproduce corpus text or annotations;
- source-neutral PropBank/PTB parsing and fail-closed conversion code;
- a local MASC audit script only after the archive schema is verified.

Not allowed yet:

- vendoring raw or processed MASC data;
- implementing MASC filename joins or normalization assumptions about the unacquired payload;
- training or tuning a model;
- publishing a dataset-derived fixture, checkpoint, or benchmark number;
- changing the gold-span objective to make a candidate fit.

## Next checkpoint

The next action is local acquisition of the official MASC-PROPBANK-ORIG archive under `data/raw/`, followed by a read-only inventory and written G1–G3 report. That action requires explicit approval because it crosses from metadata review into corpus acquisition. MASC-specific joins, normalization, and split code begin only after the report records a go decision.
