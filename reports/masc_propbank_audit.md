# MASC PropBank acquisition and feasibility audit

Status: **NO-GO for the fixed exact gold-span SRL milestone**

Audit date: 2026-08-09

## Decision

MASC-PROPBANK-ORIG is not an acceptable primary dataset for this project's
fixed contract:

```text
(sentence words, supplied verbal predicate index) -> one gold PropBank BIO tag per word
```

The read-only audit found a potentially rights-manageable subset, but both the
join and annotation-fit gates remain blocking. In a strict 44-document
diagnostic slice, 294 of 4,357 otherwise eligible verbal predicate instances
contain at least one unlinked, unindexed `*PRO*` argument. Those annotations do
not identify an overt surface span that can be represented as gold BIO labels.
Even if every other parser, LINK, overlap, and trace-chain failure were repaired
perfectly, conversion coverage could reach at most 4,063 / 4,357 = **93.25%**,
below the predeclared 99% threshold.

Dropping those arguments while labeling their surface words `O`, guessing a
controller, or excluding the affected predicates after seeing the audit would
relax the gold-span claim. The dataset is therefore rejected for this
milestone. G4 conversion integration, G5 split construction, training, and
checkpoint work were not started.

## Acquisition record

| Field | Recorded value |
| --- | --- |
| Official download | [ANC MASC data download](https://anc.org/data/masc/downloads/data-download/) |
| Artifact URL | `https://www.anc.org/MASC/download/Propbank-original-format.zip` |
| Local filename | `Propbank-original-format.zip` |
| Retrieval date | 2026-08-09 |
| Size | 11,402,565 bytes |
| SHA-256 | `b7e89cfbb7a0b7caf3ba5076ac95ee80810834d3522678eefd488f166ddcc4df` |
| Server `Last-Modified` | 2014-10-23 18:04:35 GMT |
| Server ETag | `"adfd45-5061ae5a8a8ad"` |

The ANC download page does not publish a release identifier or checksum. The
SHA-256 above is therefore this repository's local artifact pin, not an
upstream authenticity claim. At acquisition time the official download host
presented an expired TLS certificate. The exact official URL was retrieved
with certificate verification bypassed after recording that limitation. The
ZIP CRC and safety checks passed, but transport identity was not independently
established by a valid certificate.

The archive was read in place. No member payload was written outside the ZIP,
executed, or copied into tracked files. `data/raw/` remains ignored by Git, and
no sentence, tree, PropBank row, converted example, or corpus excerpt is
included in this report.

## G1 — rights and lineage

**Result: incomplete; provisional diagnostic manifest only; hold for the
complete archive.**

ANC states that MASC is available for any purpose under
[CC BY 3.0 US](https://creativecommons.org/licenses/by/3.0/us/). Its
[download page](https://anc.org/data/masc/downloads/data-download/) identifies
MASC-PROPBANK-ORIG as a bundle of original-format PropBank annotations and the
Penn Treebank parses they reference, while the separate
[annotation inventory](https://anc.org/data/masc/corpus/annotations/) supplies
the exact 88,530-word coverage. The archive itself contains a README but no
archive-level `LICENSE`, `COPYING`, or `NOTICE` file. Five frame XML files
happen to have `copying-*`, `license-*`, or `notice-*` lemma names; they are
annotation frames, not legal notices.

The [official MASC I contents map](https://www.anc.org/MASC/mascI_contents.html)
supports an exact 48-document, source-family-mapped candidate manifest across
ICIC, Slate, Verbatim, OUP, Switchboard, Berlitz, and PLOS. The
[ANC open-portion agreement](https://catalog.ldc.upenn.edu/license/anc-2nd-release-open.pdf)
corroborates the upstream lineage and credits for the first six families. The
PLOS document also has an
[article-level CC Attribution notice](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0010029).
This manifest was permitted only for read-only diagnostic analysis. It is not
cleared for training, corpus redistribution, or checkpoint release because the
audit did not complete its item-level attribution manifest. The remaining
non-WSJ source families also stay on hold.

All WSJ material is denied across text, PTB, and PropBank layers. ANC's current
corpus-wide web license statement does not expressly resolve the bundled WSJ
slice against the separately licensed
[Penn Treebank WSJ](https://catalog.ldc.upenn.edu/LDC99T42) and
[PropBank I](https://catalog.ldc.upenn.edu/LDC2004T14) lineage. Written ANC
confirmation would be required to reconsider it.

Across the 102 canonical document identifiers in the archive union, the
fail-closed disposition is:

| Disposition | Documents | Treatment |
| --- | ---: | --- |
| Provisional diagnostic manifest | 48 | Eligible only for read-only G2/G3 audit |
| WSJ deny | 37 | Exclude across every layer |
| Hold | 17 | Exclude pending exact source-family basis |

If a future, separately approved use revisits MASC, attribution must name MASC
and ANC; retain supplied source-author, publisher, Penn Treebank, and PropBank
credits; link the archive and CC license; identify the archive hash and all
conversion/filtering changes; and avoid implying ANC endorsement. Checkpoint
publication would still require its own review.

## G2 — archive and join audit

**Result: hold for the complete archive and the pinned candidate manifest.**

Archive safety and inventory:

| Measure | Count |
| --- | ---: |
| ZIP entries / files / directories | 8,091 / 8,068 / 23 |
| Uncompressed bytes | 30,576,573 |
| Text files | 99 (10 spoken, 89 written) |
| PTB files | 100 (10 spoken, 90 written) |
| PropBank files | 100 (10 spoken, 90 written) |
| Frame XML files | 7,698 |
| Legacy `.svn` files ignored | 67 |

CRC, path-traversal, absolute-path, backslash-path, symlink, encryption,
duplicate-name, and case-fold-collision checks all passed. No extraction was
needed.

Exact, case-sensitive filename normalization produces 99 text IDs, 100 PTB
IDs, 98 PropBank IDs, 102 IDs in the union, and 96 IDs common to all three
layers. Normalization uses only the observed finite alias table; it never maps
`wsj_0120` to `wsj_0122`. The mismatch is preserved explicitly:

- PropBank has `sw2017-ms98-a-trans` and `wsj_0122` without text or PTB.
- PTB has `sw2071-ms98-a-trans`, `20000815_AFP_ARB.0084.IBM-HA-NEW-en`,
  `ch5`, and `wsj_0120` without PropBank.
- `20000815_AFP_ARB.0084.IBM-HA-NEW-en` also lacks text.

The two exact `_LU_ANNOTATE` PropBank files are byte-identical mirrors of their
base Enron files and account for 118 duplicate physical rows. Excluding those
mirrors leaves 14,884 canonical PropBank rows. Of those, 14,698 join to a valid
parse sentence and predicate terminal. The remaining join failures are 173
rows without text/PTB documents, five out-of-range sentence indexes, and eight
out-of-range predicate terminals. There are also two rows without exactly one
`rel` field and three conflicting duplicate predicate keys.

A balanced-tree structural scan validates 99 of 100 PTB documents and 6,020
sentences. `ch5` has an unmatched closing parenthesis and no PropBank layer.
The production parser additionally rejects archive-specific serialization
wrappers in `VOL15_3` and `sw2025-ms98-a-trans`; the balanced scanner can audit
their joins, but no corpus-specific parser change was made after the G3 no-go.

Of the 48 provisionally mapped documents, `ch5` and
`sw2071-ms98-a-trans` lack a PropBank layer. Of the remaining 46, the
production parser rejects the wrapper
forms in `VOL15_3` and `sw2025-ms98-a-trans`, which together contain 2,607
PropBank rows. These are unresolved adapter gaps, not pre-approved exclusions,
so G2 does not pass. The remaining 44 documents and 4,357 deterministically
joined rows were used only for the diagnostic G3 audit. The audit output
records every exclusion and does not apply offset heuristics or silent repair.

## G3 — annotation fit

**Result: fail independently of G2.**

All 4,357 retained rows are modern-dialect, typed-verbal records whose
predicate POS passes the converter's verbal-tag check. The observed
annotations include simple and discontinuous pointers, comma and semicolon
concatenation, trace chains, multi-pointer predicates, and `LINK-PCR`,
`LINK-PRO`, `LINK-PSV`, and `LINK-SLC` records.

The current source-neutral converter produces 3,375 lossless BIO instances:

| Outcome | Records | Coverage |
| --- | ---: | ---: |
| Converted | 3,375 | 77.46% |
| Rejected with an explicit reason | 982 | 22.54% |
| Total eligible verbal records | 4,357 | 100.00% |

Current rejection reasons reconcile exactly to 982:

| Reason | Records |
| --- | ---: |
| Argument has no surface realization | 746 |
| LINK cannot be associated under the current exact-node rule | 173 |
| Trace chain has multiple surface candidates | 32 |
| Semantic roles overlap | 12 |
| Mixed semicolon/trace-chain grouping unsupported | 10 |
| Pointer pieces overlap | 7 |
| Predicate piece overlaps | 1 |
| Unsupported role label | 1 |

Several of these are implementation limitations and could be reduced by a
MASC-specific converter. They are not the rejection basis. The decisive bound
in this strict diagnostic slice comes from the 294 records containing an
unlinked, unindexed `*PRO*` argument:

```text
optimistic exact-span ceiling = (4,357 - 294) / 4,357 = 93.25%
```

This calculation grants perfect conversion to every other failure, including
all wrapper, LINK, overlap, and trace-chain cases. It still cannot reach the
99% gate. The result is therefore robust to repair of the current converter's
known limitations.

A second sensitivity audit used the wider set of deterministic non-WSJ layer
pairs, without treating their held rights status as cleared. Starting from
14,096 canonical non-WSJ rows, it removed the 164-row orphan `sw2017` file and
the entire 22-row mismatched Enron-Pearson document. The resulting 60-document
universe contains 13,910 rows: 13,839 verbal and 71 nominal predicates. Current
converter coverage is 9,244 / 13,839 = **66.80%**. In-memory tests of the two
wrapper forms, mixed semicolon/trace-chain pointers, and corpus-observed LINK
anchoring raise it to 11,416 / 13,839 = **82.49%**.

More importantly, 1,383 verbal rows in that sensitivity universe have a
trace-only argument with no LINK. Only 394 identify exactly one overt PTB
coindex candidate; 989 have no unique overt recovery. Those 989 cases alone
cap exact conversion at **92.85%**. Inferring a controller would create a
heuristic silver target rather than recover a gold span. Both independently
defined audit slices therefore fail the 99% requirement.

### Deferred source-neutral format findings

The audit also identified three implementation gaps that are separate from
the dataset rejection:

- two observed unlabeled PTB wrapper serializations are not accepted by the
  production parser;
- 22 rows in the wider clean universe use mixed semicolon/`*` pointer
  expressions that the current parser rejects;
- exact `TreePointer` identity is too strict as a LINK-to-argument association
  rule; 263 of 1,181 clean `LINK-SLC` fields use another anchoring structure.

These findings are recorded as future source-neutral parser work. They were
tested only in memory during the sensitivity analysis and were not folded into
the production converter after MASC failed the intrinsic ceiling. Any fix must
be specified with invented regression fixtures and must not introduce
controller inference.

## Gate reconciliation

| Gate | Result | Consequence |
| --- | --- | --- |
| G1 rights and lineage | **Incomplete; full archive hold** | Use the 48 mapped non-WSJ documents for diagnostics only |
| G2 archive and joins | **Hold: missing layers and two production-parser gaps** | Use 44 documents only as a diagnostic slice |
| G3 annotation fit | **Fail: 93.25% strict-slice ceiling and 92.85% wider ceiling** | Reject MASC for the fixed milestone |
| G4 conversion verification | Not started | Stop after G3 |
| G5 split freeze | Not started | No split or prepared dataset |
| Training and release | Not started | No model, result, or checkpoint |

## Reproduction

Place the exact archive at the ignored local path
`data/raw/Propbank-original-format.zip`, then run from the repository root:

```bash
PYTHONPATH=src python -m semantic_action_extractor.srl.masc_audit \
  data/raw/Propbank-original-format.zip
```

The command reads the ZIP in place and prints aggregate JSON only. It first
checks the filename, byte size, SHA-256, CRC, and archive safety controls. The
synthetic tests do not contain corpus-derived text or annotations:

```bash
python -m unittest tests.test_srl_masc_audit -v
```

## Next decision

The project remains dataset-blocked, not technically abandoned. The next step
is a new source-selection decision with one of three explicit scopes:

1. keep exact span SRL and evaluate a controlled-access Treebank/PropBank
   source under its access and publication restrictions;
2. identify another public corpus that directly supplies constituent argument
   spans and can pass the same gate unchanged; or
3. formally change the research target to dependency-head SRL and reconsider
   Universal Proposition Bank as a different experiment.

No option is selected by this audit.
