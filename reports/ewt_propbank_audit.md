# EWT PropBank cross-release alignment audit

Status: **technical pass for a validated inferred join; not legal clearance and not PropBank's prescribed LDC mapping**

Audit date: 2026-08-10

## Purpose

This audit asks whether the public PropBank EWT gold skeletons can be paired deterministically with the public words in UD English EWT r2.2 to create the project's fixed span-SRL examples without registration or an LDC corpus download.

The answer is yes for private preparation and training, subject to the exclusions and publication boundary below. The audit does not assert that this inferred pairing is an official PropBank conversion, and it does not grant rights in the underlying source text.

## Pinned inputs

| Input | Pinned revision | Relevant official documentation |
| --- | --- | --- |
| PropBank release | [`4abade0b53ce4a181e1d98b3518101c1a44d395a`](https://github.com/propbank/propbank-release/tree/4abade0b53ce4a181e1d98b3518101c1a44d395a) | [README](https://github.com/propbank/propbank-release/blob/4abade0b53ce4a181e1d98b3518101c1a44d395a/README.md), [EWT skeleton directory](https://github.com/propbank/propbank-release/tree/4abade0b53ce4a181e1d98b3518101c1a44d395a/data/google/ewt), [license](https://github.com/propbank/propbank-release/blob/4abade0b53ce4a181e1d98b3518101c1a44d395a/LICENSE) |
| UD English EWT r2.2 | [`6e064999a75b9c941c515ce1be98352e6f9831e0`](https://github.com/UniversalDependencies/UD_English-EWT/tree/6e064999a75b9c941c515ce1be98352e6f9831e0) | [README](https://github.com/UniversalDependencies/UD_English-EWT/blob/6e064999a75b9c941c515ce1be98352e6f9831e0/README.md), [license](https://github.com/UniversalDependencies/UD_English-EWT/blob/6e064999a75b9c941c515ce1be98352e6f9831e0/LICENSE.txt) |

Both revisions are anonymously readable. No account, registration, CourseWorks session, LDC credential, or manual data upload is needed to fetch them.

## Method

The audit was read-only and retained only aggregate results. It did not prepare a dataset or train a model. It:

1. normalized each PropBank EWT skeleton path to the corresponding UD `newdoc` identifier;
2. joined documents by that normalized identity and sentences by their zero-based order within each document;
3. compared the number of ordinary word rows, excluding CoNLL-U multiword-token and empty-node rows;
4. parsed each gold-skeleton role column as a balanced, non-overlapping span sequence;
5. located the unique primary `(V...)` span opening in each predicate column, used that word row as `predicate_index`, and retained the predicate only when that row has a verbal Penn tag (`VB`, `VBD`, `VBG`, `VBN`, `VBP`, or `VBZ`);
6. assigned PropBank's official EWT document split from the pinned [train](https://github.com/propbank/propbank-release/blob/4abade0b53ce4a181e1d98b3518101c1a44d395a/docs/evaluation/ewt.train.txt), [development](https://github.com/propbank/propbank-release/blob/4abade0b53ce4a181e1d98b3518101c1a44d395a/docs/evaluation/ewt.dev.txt), and [test](https://github.com/propbank/propbank-release/blob/4abade0b53ce4a181e1d98b3518101c1a44d395a/docs/evaluation/ewt.test.txt) lists and rejected a whole sentence if its token width differed;
7. grouped aligned sentences by the exact UD word sequence and conservatively marked every member of a group appearing in more than one split for exclusion.

This validation does not compare the hidden LDC words to UD words: PropBank's public `.gold_skel` rows replace words with `[WORD]`. PropBank's [official instructions](https://github.com/propbank/propbank-release/blob/4abade0b53ce4a181e1d98b3518101c1a44d395a/README.md) prescribe acquiring the corresponding LDC treebank and running `map_all_to_conll.py`. The audited method is instead an inferred cross-release join supported by identifiers, sentence structure, token widths, POS evidence, split agreement, and an independent UP 1.0 record of the known exceptions.

### Predicate-anchor correction

The lemma/roleset metadata row is not necessarily the primary predicate anchor. In a discontinuous predicate, the metadata can occur on a `(C-V*)` continuation row. The production adapter's real-data audit finds 13 predicate columns where the metadata row differs from the primary `V` start. It also finds 810 columns with multi-token primary `V` spans and 51,084 primary-`V` span tokens across all predicate columns. A preliminary aggregation classified verbal eligibility from the metadata row and produced the superseded total of 38,646 verbal predicates. The deeper format audit corrected the rule before dataset preparation or training: `predicate_index` and verbal eligibility now come from the unique primary `(V...)` span start. All results below use that corrected rule.

### Role normalization

The adapter applies the repository's source-neutral PropBank role normalizer
before emitting BIO labels. Numbered-core feature suffixes are metadata rather
than separate model classes: for example, source `ARG1-DSP` becomes model role
`ARG1`. A source continuation or reference prefix remains attached to the
normalized core role, and modifier labels canonicalize to `ARGM-*`. Unsupported
roles fail closed; none becomes `O`. The 111-label model vocabulary is therefore
the normalized, train-derived inventory with continuation closure, not the raw
set of distinct source strings.

## Results

### Inventory and structural conversion

| Measure | Count |
| --- | ---: |
| Shared PropBank/UD documents | 1,145 |
| PropBank skeleton sentences | 16,579 |
| PropBank skeleton token rows | 254,564 |
| All predicate columns | 50,262 |
| Nonverbal predicate columns excluded | 11,623 |
| Verbal predicate columns structurally accepted | 38,639 |
| Predicate columns with metadata/primary-anchor divergence | 13 |
| Predicate columns with multi-token primary `V` spans | 810 |
| Primary-`V` span tokens across all predicate columns | 51,084 |
| Malformed skeleton rows | 0 |
| Duplicate skeleton token rows | 0 |
| Sentences with inconsistent role-column widths | 0 |

Every one of the 1,145 PropBank EWT skeleton documents was found in the pinned UD release. Every one of their 16,579 sentence positions was present. The UD release has 29 additional documents containing 43 sentences with no PropBank skeleton; they are outside the training population.

### Word-width alignment

Of 38,639 structurally accepted verbal predicate instances, 38,635 are in sentences whose PropBank skeleton width equals the corresponding UD word-row width. This is 99.9896% alignment coverage.

Exactly one training sentence differs: the PropBank skeleton has 36 rows and UD
has 35 word rows. The whole sentence is excluded; none of its four verbal
predicates or arguments is partially repaired.

The official [UP 1.0 English EWT README](https://github.com/UniversalPropositions/UP-1.0/blob/master/UP_English-EWT/README.org) independently reports 28 development and 15 test sentences marked `no-up`, plus exactly one training sentence marked `diff-number-tokens`. That reproduces the audit's 43 UD-only sentences and single width mismatch. The same README explains that UP mapped the PropBank data through LDC2012T13 and then projected constituent arguments to dependency heads. This makes it useful corroborating evidence for source correspondence, but UP's head-only output is not used as this project's gold-span target.

### Non-gating correspondence diagnostics

The adapter records POS and predicate-lemma correspondence as diagnostics, not
hard gates. Equal token width fixes the positional model boundary; differences
between annotation releases may remain without implying an offset.

| Diagnostic | Exact matches | Differences | Exact rate |
| --- | ---: | ---: | ---: |
| Sentence-level XPOS sequence, among 16,578 equal-width sentences | 16,429 | 149 | 99.1012% |
| Token-level XPOS, among 254,528 equal-width token positions | 254,370 | 158 | 99.9379% |
| Metadata lemma versus UD lemma at the primary-`V` start, among 38,635 predicates | 38,142 | 493 | 98.7240% |
| Roleset lemma versus UD lemma at the primary-`V` start, among 38,635 predicates | 37,442 | 1,193 | 96.9121% |

Lemma comparisons use Unicode casefolding at the primary-`V` start; the roleset
lemma is the text before its final period. The tracked
[private source review](ewt_private_source_review.md) targeted these exceptions
without publishing text or identifiers. It inspected all 13 metadata/primary
anchor divergences, the one width mismatch, and a deterministic 30-record
aligned-verbal sample.

### Official splits and leakage control

After the one width-mismatched sentence is removed, the official document splits contain:

| Split | Aligned predicates |
| --- | ---: |
| Train | 31,174 |
| Development | 3,806 |
| Test | 3,655 |
| **Total** | **38,635** |

An exact-word-sequence check found 82 duplicate-text groups whose members cross official split boundaries. The conservative policy excludes all verbal predicate instances attached to every sentence in those groups:

| Split | Duplicate-group predicates excluded | Retained after cross-split filter |
| --- | ---: | ---: |
| Train | 44 | 31,130 |
| Development | 19 | 3,787 |
| Test | 13 | 3,642 |
| **Total** | **76** | **38,559** |

### Identical-input conflicts and repeated evaluation semantics

The next audit key is the exact model input `(words, predicate_index)`. If one such input has more than one distinct gold `tags` sequence, the input is label-ambiguous for this deterministic task. The fail-closed policy excludes every example in each conflicting group, rather than choosing a majority label or allowing an input to appear with contradictory targets.

The corrected, cross-split-filtered records contain:

- 29 train examples across 13 conflicting groups;
- two development examples in one conflicting group;
- no test conflict groups.

All 31 conflicting examples are excluded. After that exclusion, an exact
duplicate column with the same source document, sentence, predicate index, and
target would be collapsed deterministically so the persisted semantic identity
remains unique; the pinned sources contain zero such duplicates. Frequency
across distinct training source identities is preserved. Evaluation is
different: within each of development and test, exact repeated
`(words, predicate_index, tags)` semantics across distinct source identities
are counted once. This removes 10 development examples and 32 test examples.

| Split | After cross-split filter | Conflicts excluded | Exact eval repetitions removed | Prepared eligible |
| --- | ---: | ---: | ---: | ---: |
| Train | 31,130 | 29 | 0 | 31,101 |
| Development | 3,787 | 2 | 10 | 3,775 |
| Test | 3,642 | 0 | 32 | 3,610 |
| **Total** | **38,559** | **31** | **42** | **38,486** |

This makes 38,559 the post-cross-split count and 38,486 the final prepared-eligibility count. The distinction is intentional. These rules and counts are frozen before any model outcome is inspected. The implemented adapter reproduces them; any private preparation must do so again or stop.

### `max_length=128` and label-coverage preflight

Model capacity is audited after the preparation controls so it does not silently change the prepared-corpus accounting. At the configured `max_length=128`, 62 eligible training examples are overlength. No eligible development or test examples are overlength.

| Split | Prepared eligible | Overlength | Modeled examples |
| --- | ---: | ---: | ---: |
| Train | 31,101 | 62 | 31,039 |
| Development | 3,775 | 0 | 3,775 |
| Test | 3,610 | 0 | 3,610 |
| **Total** | **38,486** | **62** | **38,424** |

After the length filter, the `google-bert/bert-base-uncased` tokenizer pinned at
revision `86b5e0934494bd15c9632b12f734a8a67f723594` builds 111 train-derived
labels, including `O` and continuation closure. Every label present in
development or test is also present in that modeled training inventory.

## Interpretation and limitations

The audit establishes a strong technical fit for this exact pair of revisions: complete coverage of PropBank skeleton documents and sentence positions, one independently documented token-width exception, structurally valid verbal span columns, official split agreement, explicit leakage and ambiguity exclusions, evaluation-semantic deduplication, and a model-length and label-coverage preflight.

It does not prove word-for-word identity with LDC2012T13 because the public skeleton intentionally omits the words. Therefore:

- the adapter must remain pinned to these two commits and fail closed on structural or width drift;
- the completed [private source review](ewt_private_source_review.md) inspected
  all 13 anchor divergences, the one width mismatch, and 30 deterministically
  selected aligned verbal records; it retains only aggregate evidence and must
  be repeated if a source pin, conversion rule, or selection policy changes;
- the experiment must be described as an inferred cross-release reconstruction, not an official PropBank/LDC conversion;
- raw or prepared corpus examples must never be committed as tests or documentation.

## Rights and publication classification

The PropBank repository is distributed under [CC BY-SA 4.0](https://github.com/propbank/propbank-release/blob/4abade0b53ce4a181e1d98b3518101c1a44d395a/LICENSE). The pinned UD EWT [README](https://github.com/UniversalDependencies/UD_English-EWT/blob/6e064999a75b9c941c515ce1be98352e6f9831e0/README.md) and [license](https://github.com/UniversalDependencies/UD_English-EWT/blob/6e064999a75b9c941c515ce1be98352e6f9831e0/LICENSE.txt) license the annotations and database rights under CC BY-SA 4.0 while noting that the underlying texts come from multiple sources and may have separate copyrights.

The resulting policy is conservative:

| Artifact | Repository/publication class |
| --- | --- |
| Adapter, validation code, synthetic fixtures | Publishable with attribution |
| Non-reconstructive aggregate counts and metrics | Publishable with attribution |
| Raw UD/PropBank files and reconstructed words | Private and ignored |
| Prepared train/development/test records | Private and ignored |
| Trained weights | Private and ignored pending a separate weights review |

This is a project publication policy, not a legal opinion.

## Gate conclusion

The no-registration EWT route passes the technical source-selection gate and supersedes BabySRL as the primary path. It preserves the fixed exact-span objective and leaves no corpus-access task for the user. The adapter and quantified private source review are complete; ignored preparation, training, and evaluation may proceed under the controls above. Public checkpoint release remains a later, independent decision.

[MASC remains rejected](../docs/datasets/masc_propbank_gate.md): its diagnostic exact-span ceiling fell below the fixed 99% threshold and its unresolved lineage/join gaps are unrelated to this EWT result.
