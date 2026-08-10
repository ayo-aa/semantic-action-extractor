# BabySRL feasibility and access gate

> **Superseded — fallback only (2026-08-10):** The no-registration EWT route
> is the selected primary source. The registration, preparation, manual-review,
> and training gates below are historical or conditional, not current user
> actions. Do not register for BabySRL unless this fallback is explicitly
> reactivated.

Status: **superseded as active path; preserved technical pass and conditional fallback controls**

Decision date: 2026-08-10

## Decision

BabySRL is a derived version of the CHILDES Brown corpus in which selected
adult utterances to Adam, Eve, and Sarah have hand-corrected Penn Treebank-style
parses and PropBank-style verbal semantic roles. Its CHAT distribution supplies
surface argument spans directly, so it fits the project's unchanged research
contract:

```text
(sentence words, supplied verbal predicate index) -> one gold PropBank BIO tag per word
```

The pinned archive passes the project's archive-integrity and 99% structural
annotation-fit gates. A strict, fail-closed audit converts 18,397 of 18,536
declared proposition columns, or **99.2501%**, without changing the target or
silently dropping a bad role from an otherwise accepted proposition.

That technical result is not authorization to train. CHILDES describes
transcript access as registration-required, and the
[TalkBank ground rules](https://talkbank.org/0share/rules.html) govern use even
though this particular ZIP currently responds to an anonymous HTTP request. The
project will not write prepared files until the user:

1. registers and signs in through
   [TalkBank/CHILDES](https://talkbank.org/childes/access.html); and
2. records acceptance of the current access conditions, ground rules, ethics
   obligations, and non-commercial limits in this decision record.

After those two access steps, the adapter may write a provisional prepared
dataset only to ignored local or approved Columbia storage. The private manual
workflow then verifies that complete prepared dataset against the pinned raw
conversion and reviews deterministic raw-cell/BIO pairs. Model training remains
blocked until the aggregate manual decision is `pass`.

No CourseWorks or Columbia sign-in is involved. Columbia authorization does not
replace the separate TalkBank registration and ground-rules step.

## Approval checkpoint

| Requirement | Status | Evidence required before training |
| --- | --- | --- |
| TalkBank/CHILDES registration | **HOLD** | User confirms registration; no credentials recorded |
| Current ground-rules review | **HOLD** | Acceptance date and the reviewed rules URL are recorded here |
| Provisional ignored preparation | **BLOCKED** | Registration and current-rules acceptance must be recorded first |
| Authorized manual conversion sample | **HOLD** | Prepared/raw identity and sampled mappings pass private review without publishing text |
| Local or approved Columbia training | **BLOCKED** | Manual aggregate is `pass`, the prepared fingerprint is frozen, and compute is suitable |
| Third-party web or cloud processing | **HOLD** | No-storage assurance and configuration evidence |
| Checkpoint redistribution | **HOLD** | Written clarification for the proposed release |

The project limits activity during the hold to the completed,
non-reconstructive aggregate feasibility analysis. That inspection is not
treated as access or training authorization. Raw transcripts, prepared
examples, corpus excerpts, and model weights are not publication artifacts.

## Acquisition pin

| Field | Recorded value |
| --- | --- |
| Official corpus page | [BabySRL Derived Corpus](https://talkbank.org/childes/access/Derived/BabySRL.html) |
| Artifact URL | `https://talkbank.org/childes/access/Derived/0docs/BabySRL.zip` |
| Local filename | `BabySRL.zip` |
| Retrieval date | 2026-08-09 |
| Size | 3,151,030 bytes |
| SHA-256 | `a2d8d38b0818910d05154cb62adcb05ef36b00f0aeae2690b7e1677fd5154604` |
| CHAT files | 133: Adam 23, Eve 20, Sarah 90 |
| Frozen document assignment | [`babysrl_split_manifest.json`](babysrl_split_manifest.json) |
| Assignment manifest SHA-256 | `73ecae9f81d1d2b9f8495b13b297da9c3d24e387630e71a38ef2420d4c9a5de7` |

The SHA-256 is a repository acquisition pin, not an upstream signature. The ZIP
is read in place after exact filename-independent size and digest checks, CRC
verification, duplicate-member checks, and rejection of encrypted, symlink, or
unsafe paths. No archive member is extracted during the audit.

The official BabySRL page confirms the 23/20/90 distribution. The detailed
[annotation and format description](https://cogcomp.seas.upenn.edu/Data/BabySRL.html)
identifies the source sections as Brown Adam 01-23, Eve 01-20, and Sarah 01-90,
and describes the parent-only verbal PropBank annotation represented in the
audited CHAT files. The underlying
[Brown corpus page](https://talkbank.org/childes/access/Eng-NA/Brown.html)
provides the source history, required citation, and
[DOI 10.21415/T5HK5G](https://doi.org/10.21415/T5HK5G).

## Gate results

| Gate | Result | Consequence |
| --- | --- | --- |
| G1 access and usage basis | **HOLD** | Registration and recorded ground-rules acceptance are still required |
| G2 archive integrity and structure | **PASS** | Pinned archive safely reads as 133 CHAT documents |
| G3 annotation fit | **PASS** | 18,397 / 18,536 proposition columns convert structurally: 99.2501% |
| G4 conversion verification | **STRUCTURAL PASS; MANUAL HOLD** | Counts reproduce; authorized sample remains required |
| G5 split policy | **FROZEN; preparation HOLD** | Split and duplicate rules are frozen |
| Training | **NOT STARTED** | Starts only after G1 is cleared |
| Checkpoint release | **HOLD** | Corpus terms require a separate release decision |

G3 required at least 99% conversion. With a denominator of 18,536, the minimum
passing count is 18,351. The audited result passes by 46 proposition columns.
This is a representability gate, not a model score and not an assertion that the
underlying linguistic annotations are error-free.

## Strict conversion contract

The adapter is fail-closed and applies the following rules:

1. Consecutive `%srl:` rows form one utterance block. Each physical row must
   provide the tier marker, surface token, predicate marker, and the block's
   role columns.
2. The denominator is the maximum declared role-column count in each block,
   summed across the archive. It is not the number of predicate-marker rows and
   not merely the first row's width.
3. Each role cell must contain one token anchor and a valid, balanced bracket
   sequence. A malformed column rejects the whole proposition.
4. Supported source labels are `V`, its explicit continuation `C-V`, core roles
   `A0`-`A5`, and `AM-*` modifiers. Core and modifier labels normalize to the
   repository's `ARG0`-`ARG5` and `ARGM-*` vocabulary. Other continuation,
   reference, or ad hoc labels are rejected; no unknown label becomes `O`.
5. The `V`/`C-V` relation pieces and predicate markers must resolve to exactly
   one supplied surface predicate index. The head receives `B-V`; other overt
   relation pieces receive `B-C-V`/`I-C-V`. A missing relation or ambiguous
   head rejects the proposition.
6. Every accepted proposition produces exactly one validated word-level BIO
   sequence. Surface discontinuities remain separate BIO spans; gaps are never
   filled. CHAT supplies the predicate lemma but not a verified sense, so the
   roleset metadata uses the explicit unknown-sense suffix `.XX`.
7. A proposition is either accepted in full or assigned exactly one terminal
   rejection reason. No difficult argument is deleted while the rest of its
   proposition is retained.

The five mutually exclusive audit outcomes are frozen regression targets:

| Rejection reason | Proposition columns |
| --- | ---: |
| `row_width_mismatch` | 4 |
| `invalid_bracket_sequence` | 15 |
| `missing_relation_span` | 99 |
| `ambiguous_predicate_head` | 5 |
| `unsupported_role_label` | 16 |
| **Rejected** | **139** |

The ordered validation contract assigns only one terminal reason to a rejected
column. Accepted 18,397 plus rejected 139 reconciles exactly to the 18,536-column
denominator.

## Document split and leakage policy

The split is fixed before any training result is observed. Documents are never
divided across partitions. Within each child, the allocator computes:

```text
sha256("babysrl-v1\0" + child + "/" + source_basename)
```

It sorts by `(digest, document_id)`, assigns `floor(0.80 * n)` documents to
train, `floor(0.10 * n)` to development, and the remainder to test.

The exact 133-document result is frozen in
[`babysrl_split_manifest.json`](babysrl_split_manifest.json). Its canonical
SHA-256 is
`73ecae9f81d1d2b9f8495b13b297da9c3d24e387630e71a38ef2420d4c9a5de7`.

| Child | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Adam | 18 | 2 | 3 | 23 |
| Eve | 16 | 2 | 2 | 20 |
| Sarah | 72 | 9 | 9 | 90 |
| **Total** | **106** | **13** | **14** | **133** |

Exact repetitions are handled after this file assignment, without moving a
document to improve a result:

- Hash each exact sentence-token sequence. If the same sequence occurs in more
  than one split, exclude every proposition attached to every occurrence from
  all affected splits.
- Keep all remaining training occurrences. Training frequency is part of the
  observed corpus and is not retrospectively altered.
- Within development and test separately, compute the exact semantic
  fingerprint `(words, predicate_index, BIO tags)`. Keep the
  lexicographically first example ID and remove later identical fingerprints.
- Reject conflicting annotations for one source sentence/predicate identity;
  do not resolve them by preference or majority vote.

This policy prevents exact text leakage into evaluation, prevents repeated
evaluation items from receiving extra weight, and preserves the predeclared
whole-document split.

The privacy-safe in-memory audit produces the following aggregate accounting;
no prepared file is written during this calculation:

| Stage | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Accepted before duplicate policy | 15,160 | 1,682 | 1,555 | 18,397 |
| Cross-split sequence exclusions | 1,447 | 289 | 257 | 1,993 |
| Within-split semantic duplicates removed | 0 | 37 | 24 | 61 |
| **Final eligible examples** | **13,713** | **1,356** | **1,274** | **16,343** |

The 1,993 exclusions belong to 274 exact sentence-token sequences crossing a
split boundary. The zero in the training deduplication row is required by
policy, not an accidental absence of repeated training examples.

## Rights and publication boundary

TalkBank states that its data are governed by CC BY-NC-SA 3.0 unless otherwise
indicated, excludes incorporation into commercial products including model
systems, restricts web processing to services that assure non-storage, and
requires confidentiality and ethics protections. The repository therefore
uses the following conservative boundary:

Repository activity retained during the access hold:

- source-neutral adapter code and wholly invented test fixtures;
- this decision record and non-reconstructive aggregate counts;
- the completed read-only, non-reconstructive feasibility verification.

Allowed only after registration and recorded acceptance:

- local preparation into an ignored directory;
- non-commercial training on a local machine or an approved Columbia system
  that does not circulate the data;
- aggregate evaluation and error-category counts that reveal no corpus text.

Not approved for publication:

- raw or prepared transcript data;
- corpus-derived examples, quotations, or fixtures;
- trained checkpoints, pending written clarification of model-incorporation,
  share-alike, privacy, and redistribution implications;
- uploading data to Colab or another web service without an explicit no-storage
  assurance that satisfies TalkBank's rules.

## Source and version notes

The audited artifact is identified by its byte-level pin, not by an inferred
release label. The format page documents 15,148 parsed and annotated parental
utterances; the pinned CHAT archive contains 15,147 `%srl:` utterance blocks.
The one-block difference is recorded as source-documentation drift and is not
silently synthesized or repaired.

The 2010 paper,
[Starting from Scratch in Semantic Role Labeling](https://aclanthology.org/P10-1101/),
describes the original child-directed-speech SRL resource. The
[2018 adult-child SRL paper](https://aclanthology.org/C18-1254/) describes a
later Adam-focused augmentation covering adult and child speech and
prepositions as well as verbs. The full 133-file archive audited here exhibits
the parent-only verbal CHAT regime described by the format page; this project
does not infer that all 2018 augmentation features are present.

## Next checkpoint

Training is still required to complete the portfolio experiment, including the
predicate-signal model and its no-signal ablation. The next action is the
TalkBank registration and acceptance checkpoint. Once the user confirms it,
the project may write provisional ignored prepared splits, create and complete
the private raw-versus-BIO review, freeze the prepared fingerprint after a
`pass`, and then start the controlled research training run. The completed
invented-data runtime rehearsal does not satisfy this gate. No CourseWorks
access is needed at any stage.
