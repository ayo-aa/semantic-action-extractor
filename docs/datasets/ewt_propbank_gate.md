# EWT PropBank feasibility gate

Status: **selected no-registration source; technical gate passed for private preparation, with checkpoint publication still under separate review**

Decision date: 2026-08-10

## Decision

Use the public PropBank English Web Treebank (EWT) gold skeletons together with the words in the pinned Universal Dependencies English EWT r2.2 release for the fixed milestone:

```text
(sentence words, supplied verbal predicate index) -> one gold PropBank BIO tag per word
```

This route requires no account, registration, CourseWorks login, LDC download, or user-provided corpus file. Both pinned sources are publicly readable Git repositories:

- [PropBank release at `4abade0b53ce4a181e1d98b3518101c1a44d395a`](https://github.com/propbank/propbank-release/tree/4abade0b53ce4a181e1d98b3518101c1a44d395a), specifically [`data/google/ewt`](https://github.com/propbank/propbank-release/tree/4abade0b53ce4a181e1d98b3518101c1a44d395a/data/google/ewt)
- [UD English EWT r2.2 at `6e064999a75b9c941c515ce1be98352e6f9831e0`](https://github.com/UniversalDependencies/UD_English-EWT/tree/6e064999a75b9c941c515ce1be98352e6f9831e0)

The completed [alignment audit](../../reports/ewt_propbank_audit.md) and
[private source review](../../reports/ewt_private_source_review.md) support the
technical decision. This is a validated, inferred cross-release join. It is
**not** the LDC-based mapping prescribed by PropBank. PropBank's pinned
[README](https://github.com/propbank/propbank-release/blob/4abade0b53ce4a181e1d98b3518101c1a44d395a/README.md)
says its `[WORD]` skeletons are normally completed by acquiring LDC2012T13 and
running the provided mapping script. This project instead joins the public
skeletons to the public UD release, validates the shared structure, and fails
closed on any token-width mismatch.

## Gate result

| Check | Result | Decision |
| --- | ---: | --- |
| Shared documents | 1,145 | Pass; every PropBank EWT skeleton document joins a UD EWT document |
| PropBank skeleton sentences | 16,579 | Pass; every skeleton sentence has a corresponding document/sentence position |
| Structurally accepted verbal predicates | 38,639 | Pass; the skeleton columns fit the fixed span representation |
| Token-width mismatches | 1 train sentence / 4 predicates | Exclude the entire sentence and its four predicate instances |
| Word-aligned verbal predicates | 38,635 | Pass; 99.9896% of structurally accepted verbal predicates remain |
| Cross-split duplicate-text groups | 82 | Exclude every affected predicate instance from the post-leakage population |
| Identical-input label conflicts | 31 examples in 14 groups | Exclude every example in each conflicting group |
| Repeated evaluation semantics | 42 examples | Deduplicate exact semantics within development and test |
| Final prepared eligibility | 38,486 examples | Freeze before any model outcome is inspected |
| `max_length=128` overlength | 62 train examples | Keep separate from preparation counts; exclude at the modeling boundary |
| Modeled population | 38,424 examples | 31,039 train / 3,775 development / 3,610 test |

Predicate eligibility and `predicate_index` are defined by the word row where the unique primary `(V...)` span begins. The lemma/roleset metadata row is not a reliable anchor because a discontinuous predicate can place that metadata on a `(C-V*)` row. The real-data adapter audit finds 13 columns where those two rows differ, 810 columns with multi-token primary `V` spans, and 51,084 total primary-`V` span tokens across all predicate columns. A preliminary aggregation used the metadata row and reported 38,646 verbal predicates; the format audit corrected that classification before adapter implementation, dataset preparation, or training. Every count below uses the primary-`V` rule.

The private review inspected all 13 / 13 metadata/primary-anchor divergences,
the one 36-versus-35 width mismatch, and 30 deterministically selected distinct
aligned verbal records. All 13 divergences support the primary-`V` rule; only
one is verbal under the primary row's XPOS. All 30 / 30 sampled records had
bounded, balanced same-width span placement, with no systematic or off-by-one
positional drift observed. The review publishes no corpus text or identifiers
and remains supporting evidence rather than proof against the hidden LDC words.

Model-facing roles use the repository's source-neutral normalization. For
example, the numbered-core feature in source `ARG1-DSP` does not create a
separate class: the model role is `ARG1`. Continuation/reference prefixes remain
attached to the normalized role, modifiers canonicalize to `ARGM-*`, and
unsupported roles fail closed rather than becoming `O`.

Non-gating join diagnostics further record 16,429 / 16,578 exact sentence-XPOS
sequences, 254,370 / 254,528 exact token-XPOS positions (99.9379%), 38,142 /
38,635 exact metadata-lemma comparisons, and 37,442 / 38,635 exact
roleset-lemma comparisons. These diagnostics describe annotation-release
differences; they do not override the fail-closed positional gates.

The official split counts and conservative duplicate exclusions are:

| Split | Aligned before duplicate exclusion | Excluded as cross-split duplicate text | Retained after cross-split filter |
| --- | ---: | ---: | ---: |
| Train | 31,174 | 44 | 31,130 |
| Development | 3,806 | 19 | 3,787 |
| Test | 3,655 | 13 | 3,642 |
| **Total** | **38,635** | **76** | **38,559** |

### Identical-input conflicts and evaluation deduplication

The post-leakage records receive two additional controls before they are eligible prepared examples:

- An identical-input conflict is a group with the same `(words, predicate_index)` but more than one distinct gold tag sequence. Every example in a conflicting group is excluded: 29 train examples across 13 groups and two development examples in one group. There are no test conflicts.
- An identical-target duplicate at the same source document, sentence, and
  predicate index would be collapsed deterministically; the pinned sources
  contain zero such duplicates.
- Nonconflicting frequency across distinct training source identities is
  retained.
- Within development and test, exact repeated semantics with the same
  `(words, predicate_index, tags)` across distinct source identities are
  deduplicated. This removes 10 development examples and 32 test examples so
  repeated evaluation items cannot receive extra weight.

| Split | After cross-split filter | Conflicting-input examples excluded | Repeated eval semantics removed | Final prepared eligibility |
| --- | ---: | ---: | ---: | ---: |
| Train | 31,130 | 29 | 0 | 31,101 |
| Development | 3,787 | 2 | 10 | 3,775 |
| Test | 3,642 | 0 | 32 | 3,610 |
| **Total** | **38,559** | **31** | **42** | **38,486** |

These policies and counts are frozen before training or inspection of model outcomes. The one token-width mismatch is in the training split; no partial predicate or argument is silently retained from that sentence.

### Model-length and label preflight

Prepared eligibility is recorded independently from model capacity. With the configured `max_length=128`, 62 otherwise eligible training examples are overlength; development and test have none. Those 62 examples are omitted at the modeling boundary:

| Split | Prepared eligible | Overlength at `max_length=128` | Modeled |
| --- | ---: | ---: | ---: |
| Train | 31,101 | 62 | 31,039 |
| Development | 3,775 | 0 | 3,775 |
| Test | 3,610 | 0 | 3,610 |
| **Total** | **38,486** | **62** | **38,424** |

The `google-bert/bert-base-uncased` tokenizer pinned at revision
`86b5e0934494bd15c9632b12f734a8a67f723594` builds 111 train-derived labels,
including `O` and continuation closure. The modeled development and test
populations contain no label absent from that training vocabulary.

The frozen [UP 1.0 English EWT documentation](https://github.com/UniversalPropositions/UP-1.0/blob/master/UP_English-EWT/README.org) independently records the same release-boundary shape: 28 development and 15 test sentences have no PropBank counterpart, and exactly one training sentence has a different token count. UP 1.0 itself remains unsuitable as the training target because it projects arguments to dependency heads rather than preserving the full gold spans required by this milestone. Its documentation is corroborating evidence for the join, not the selected labels.

## Rights and repository boundary

Public availability is not the same as blanket redistribution permission. The pinned [UD EWT README](https://github.com/UniversalDependencies/UD_English-EWT/blob/6e064999a75b9c941c515ce1be98352e6f9831e0/README.md) and [license file](https://github.com/UniversalDependencies/UD_English-EWT/blob/6e064999a75b9c941c515ce1be98352e6f9831e0/LICENSE.txt) license the annotations and database rights under CC BY-SA 4.0, while expressly noting that the underlying texts come from multiple sources and may carry separate copyrights. PropBank's pinned repository has its own [CC BY-SA 4.0 license](https://github.com/propbank/propbank-release/blob/4abade0b53ce4a181e1d98b3518101c1a44d395a/LICENSE).

The project therefore adopts this conservative publication boundary:

- raw source files, reconstructed text, and prepared examples stay ignored and private;
- corpus-derived examples are not committed;
- trained weights stay ignored and private pending a separate checkpoint-release review;
- source-neutral code (including wholly synthetic fixtures) and non-reconstructive aggregate metrics may be published with attribution.

This boundary permits the portfolio experiment to be completed without claiming that the underlying EWT words or a checkpoint trained on them can be redistributed.

## What remains

No dataset credential or file is required from the user. The pinned adapter,
fail-closed controls, synthetic tests, and quantified private review (13 / 13
anchor divergences, one width mismatch, and 30 / 30 deterministic aligned
records) are complete. The remaining work is execution work:

1. run the first ignored preparation and require every frozen gate count to match;
2. freeze the preparation manifest, source commits, data fingerprint, and
   train-derived label vocabulary;
3. freeze the paired configurations, train, and evaluate on the retained private
   splits;
4. publish code and aggregate results, while keeping data and weights private
   unless the later weights review clears release.

## Reconciliation with the rejected routes

[MASC remains rejected](masc_propbank_gate.md). Its optimistic exact-span recovery ceiling was below the predeclared 99% requirement, and its item-level rights/join work remained unresolved. That failure is not repaired by the EWT evidence and is not being reopened.

BabySRL remains a fallback only. Although it is a plausible direct-span source, it requires registration and additional review. The validated EWT route removes that dependency while preserving the fixed gold-span objective.
