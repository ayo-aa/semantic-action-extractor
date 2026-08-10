# BabySRL archive and annotation-fit audit

Status: **technical PASS; data preparation and training remain on HOLD**

Audit date: 2026-08-10

## Result

The pinned BabySRL archive is technically suitable for the project's fixed
predicate-conditioned span-SRL experiment. The aggregate, read-only audit
found 18,536 declared proposition columns and converted 18,397 into complete
word-level BIO examples. The resulting conversion coverage is:

```text
18,397 / 18,536 = 0.9925010789814415 = 99.2501%
```

This passes the predeclared 99% structural annotation-fit gate by 46
proposition columns.
The other 139 columns are rejected in full under five mutually exclusive,
machine-counted reasons. No annotation fragment is silently relabeled `O`, no
span is guessed, and no rejected proposition enters a prepared split.

This is a feasibility result only. It does not authorize data preparation,
training, cloud upload, or checkpoint publication, and an authorized manual
sample review remains pending. Although the artifact URL currently permits a
direct anonymous download, the
[CHILDES access page](https://talkbank.org/childes/access.html) classifies
transcript and media data as registration-required. Training remains blocked
until the user registers, signs in, and records acceptance of the current
[TalkBank ground rules](https://talkbank.org/0share/rules.html). This process
does not involve CourseWorks or a Columbia login.

## Corpus identity

[BabySRL](https://talkbank.org/childes/access/Derived/BabySRL.html) is a derived
CHILDES corpus based on selected files from Roger Brown's longitudinal Adam,
Eve, and Sarah transcripts. The detailed
[SRL annotation description](https://cogcomp.seas.upenn.edu/Data/BabySRL.html)
states that selected parental utterances were parsed in a Penn Treebank-style
representation and annotated with PropBank-style verbal propositions and
surface argument spans. The audit confirms that the pinned CHAT artifact has
133 data files:

| Child | CHAT files |
| --- | ---: |
| Adam | 23 |
| Eve | 20 |
| Sarah | 90 |
| **Total** | **133** |

The underlying
[CHILDES Brown corpus](https://talkbank.org/childes/access/Eng-NA/Brown.html)
documents the source collection, required citation, participant histories, and
[DOI 10.21415/T5HK5G](https://doi.org/10.21415/T5HK5G).

## Acquisition and archive safety

| Field | Audited value |
| --- | --- |
| Corpus page | `https://talkbank.org/childes/access/Derived/BabySRL.html` |
| Archive URL | `https://talkbank.org/childes/access/Derived/0docs/BabySRL.zip` |
| Filename | `BabySRL.zip` |
| Retrieval date | 2026-08-09 |
| Compressed size | 3,151,030 bytes |
| Uncompressed size | 18,859,283 bytes |
| SHA-256 | `a2d8d38b0818910d05154cb62adcb05ef36b00f0aeae2690b7e1677fd5154604` |
| ZIP members | 140 |
| Supported `.srl.cha` members | 133 |
| CRC | Pass |
| Path-safety checks | Pass |

The upstream page does not publish a checksum. The SHA-256 above is therefore a
local acquisition pin rather than a claim of upstream cryptographic signing.
The audit rejects a size or digest mismatch, duplicate archive member, encrypted
member, symlink, absolute path, traversal path, backslash path, or malformed ZIP.
It reads supported CHAT members in memory and never extracts the archive.

Raw storage remains under ignored `data/raw/`. Prepared output, when eventually
approved, must also stay ignored. The archive, prepared JSONL, corpus excerpts,
and corpus-derived test cases are absent from Git.

## Population and denominator

The format documentation reports 15,148 parsed and annotated parental
utterances. The pinned archive contains 15,147 consecutive `%srl:` utterance
blocks. The audit records this **15,147 versus 15,148** discrepancy as
documentation/artifact drift; it does not fabricate, recover, or infer a
missing block.

CHAT encodes one physical row per surface token. After the tier marker, word,
and predicate-marker fields, each remaining field is a proposition's bracket
cell. An utterance can contain multiple proposition columns. Consequently, the
correct model-instance denominator is the sum of the maximum proposition-column
width observed in each block: **18,536**. Counting utterance blocks, predicate
markers, or only the first row's width would measure a different quantity.

| Aggregate counter | Value |
| --- | ---: |
| `%srl:` utterance blocks | 15,147 |
| First-row proposition-column total | 18,535 |
| Maximum per-block proposition-column total | 18,536 |
| Non-dash predicate-marker rows | 18,556 |

The one-column max-versus-first-row difference is retained rather than hidden,
and the 20 excess predicate-marker rows are not miscounted as propositions.

## Conversion procedure

For each CHAT document, the converter:

1. groups consecutive `%srl:` rows into an utterance block without using the
   surrounding transcript text;
2. derives the block's declared proposition width and validates every row
   against it;
3. decodes each proposition column as a balanced bracket stream over physical
   surface tokens;
4. normalizes the predeclared supported PropBank labels and rejects every
   unsupported continuation, reference, or ad hoc role;
5. resolves the column's relation span and predicate marker to one supplied
   surface predicate index;
6. creates exactly one complete BIO sequence per accepted proposition and
   validates the resulting word-level record; and
7. assigns exactly one terminal reason to a rejected proposition.

The supported source vocabulary is `V`, `C-V`, `A0`-`A5`, and `AM-*`.
Arguments and modifiers normalize to `ARG0`-`ARG5` and `ARGM-*`; unsupported
continuation, reference, or ad hoc labels fail closed. The unique marked head
receives `B-V`, while other overt relation pieces receive `B-C-V`/`I-C-V`.
Discontinuous argument annotations remain separate spans. The converter never
fills intervening words, converts an unsupported label to `O`, or keeps the
easy roles from a proposition whose other role failed. CHAT does not provide a
verified predicate sense, so accepted records retain the lemma with an
explicit `.XX` unknown-sense suffix.

## Conversion accounting

| Outcome | Proposition columns | Share |
| --- | ---: | ---: |
| Converted losslessly | 18,397 | 99.2501% |
| Rejected fail-closed | 139 | 0.7499% |
| **Declared** | **18,536** | **100.0000%** |

The five rejection reasons are mutually exclusive under the converter's fixed
validation order:

| Terminal reason | Count | Contract meaning |
| --- | ---: | --- |
| `row_width_mismatch` | 4 | Rows in the utterance do not agree on the declared proposition structure |
| `invalid_bracket_sequence` | 15 | A role column cannot be decoded as one valid balanced span sequence |
| `missing_relation_span` | 99 | The proposition has no usable base verbal relation span |
| `ambiguous_predicate_head` | 5 | The relation and marker evidence does not identify exactly one predicate word |
| `unsupported_role_label` | 16 | At least one source label is outside the predeclared lossless mapping |
| **Rejected** | **139** | Exact sum of all terminal failures |

The reconciliation invariant is exact:

```text
18,397 accepted + 139 rejected = 18,536 declared
```

At the 99% threshold, `ceil(18,536 * 0.99) = 18,351` accepted propositions are
required. The result's 46-proposition margin is materially safer than the
earlier exploratory decoder, whose broader rejection taxonomy sat only four
propositions above the gate. The five counts in this report are the frozen
regression targets for the final strict contract.

## Split freeze and duplicate controls

The audit fixes document assignment independently of conversion outcomes and
before any model result. For every child, it hashes the literal version salt,
child name, and source basename with SHA-256, sorts by digest with document ID
as a tie-breaker, assigns the first `floor(0.80 * n)` documents to train, the
next `floor(0.10 * n)` to development, and the remainder to test.

The exact assignment is committed as
[`babysrl_split_manifest.json`](../docs/datasets/babysrl_split_manifest.json),
whose canonical SHA-256 is
`73ecae9f81d1d2b9f8495b13b297da9c3d24e387630e71a38ef2420d4c9a5de7`.

```text
sha256("babysrl-v1\0" + child + "/" + source_basename)
```

| Child | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Adam | 18 | 2 | 3 | 23 |
| Eve | 16 | 2 | 2 | 20 |
| Sarah | 72 | 9 | 9 | 90 |
| **All children** | **106** | **13** | **14** | **133** |

The duplicate policy is applied only after that document split:

- detect exact sentence-token sequences that occur across split boundaries and
  exclude all propositions for all occurrences of each crossing sequence;
- retain every remaining training occurrence;
- within development and test separately, deduplicate only identical semantic
  fingerprints `(words, predicate_index, BIO tags)`, keeping
  the lexicographically first example ID; and
- fail on conflicting annotations for the same source sentence/predicate
  identity.

No document is moved to recover excluded data, and no evaluation outcome can
change the frozen allocator. The aggregate in-memory result is:

| Stage | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Accepted before duplicate policy | 15,160 | 1,682 | 1,555 | 18,397 |
| Cross-split sequence exclusions | 1,447 | 289 | 257 | 1,993 |
| Development/test semantic duplicates removed | 0 | 37 | 24 | 61 |
| **Final eligible examples** | **13,713** | **1,356** | **1,274** | **16,343** |

The cross-split exclusions cover 274 exact sentence-token sequences. Training
deduplication is deliberately disabled, so all 0 removals in that column are a
policy invariant. The prepared manifest digest will be recorded only after the
access hold is cleared and the ignored dataset is written.

## Scope of the pass

The structural annotation-fit pass establishes that at least 99% of the
artifact's declared verbal proposition columns can be represented by the fixed
one-label-per-word BIO target. It does not establish:

- permission to train or publish model weights;
- an official train/development/test split supplied by BabySRL;
- correctness of every accepted parse or role annotation;
- availability of PropBank sense IDs in the CHAT files;
- coverage of child speech, preposition predicates, or the complete Brown
  corpus; or
- benchmark comparability with OntoNotes or CoNLL SRL datasets.

The annotation documentation itself records known omissions, misparses, and
inconsistent treatment of some moved or split arguments. Those are corpus
quality limitations, not reasons to silently edit the source or relax the
evaluation contract.

The lineage also requires care. Connor et al.'s
[2010 ACL paper](https://aclanthology.org/P10-1101/) describes the original
child-directed-speech SRL work. Moon et al.'s
[2018 COLING paper](https://aclanthology.org/C18-1254/) describes a later
Adam-focused extension with verb and preposition senses and both adult and child
speech. The full 133-file pinned archive presents the parent-only verbal CHAT
regime documented by CogComp, so this audit does not attribute unobserved 2018
features to it.

## Access, training, and release disposition

The [TalkBank ground rules](https://talkbank.org/0share/rules.html) state that,
unless otherwise indicated, the data use CC BY-NC-SA 3.0; exclude incorporation
into commercial products including model systems; require non-storage
assurances for web processing; and impose ethics and confidentiality
obligations. Direct HTTP reachability is not treated as proof that registration
or those terms can be skipped.

| Activity | Disposition |
| --- | --- |
| Aggregate read-only audit | Completed technical inspection; not access authorization |
| Source-neutral code and synthetic tests | Allowed |
| User TalkBank registration and rules acceptance | Required; not yet recorded |
| Provisional prepared local dataset | HOLD pending registration and acceptance; must remain ignored |
| Authorized manual conversion sample | HOLD; runs privately over the fingerprinted prepared/raw pair |
| Model training | HOLD pending registration, acceptance, and a manual-review `pass` |
| Colab or other web processing | HOLD absent explicit no-storage assurance |
| Raw/prepared data publication | Prohibited by repository policy |
| Checkpoint redistribution | HOLD pending written rights clarification |
| Aggregate metrics after approved training | Planned, with privacy-safe reporting |

This conservative checkpoint policy reflects ambiguity around
model-incorporation, non-commercial scope, share-alike obligations, and
participant confidentiality. It is not a broader legal conclusion about every
possible research use.

## Reproduction boundary

After independently obtaining the exact archive under an approved TalkBank
account, place it at the ignored path `data/raw/BabySRL.zip`. The aggregate audit
reads the ZIP in place and prints no corpus text:

```bash
PYTHONPATH=src python -m semantic_action_extractor.srl.babysrl \
  data/raw/BabySRL.zip
```

Synthetic tests contain no BabySRL sentence, annotation, or excerpt. The audit
must reproduce the exact archive pin, 133-document inventory, 18,536-column
denominator, 18,397 accepted columns, 139 exclusive rejections, 99.2501%
coverage, and 106/13/14 document allocation before training can proceed.

## Conclusion

BabySRL resolves the MASC structural annotation-fit failure without changing
the semantic project's claim: it directly supplies overwhelmingly
representable surface argument spans conditioned on a supplied verbal
predicate. The technical data gate is passed, but the operational gate is
intentionally not. The next steps are TalkBank registration and recorded
acceptance of the current rules; only then may provisional ignored preparation
begin. The authorized manual workflow must compare that prepared dataset with
the pinned raw source and reach `pass` before model training. CourseWorks is not
part of that workflow.
