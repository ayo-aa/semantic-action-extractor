# QANom data card

## Status

| Field | Status |
| --- | --- |
| Release verification | Completed on 2026-08-06 |
| Dataset revision | Archive distributed by QANom revision `2bce70e8a39b40157ba97f38e1a8ae7619b30162` |
| Adapter | Implemented and fixture-tested; all three verified splits processed successfully |
| Repository data | None |
| Redistribution decision | Raw and processed records remain outside Git; checkpoint publication remains unresolved |

## Purpose

QANom extends QA-SRL from verbal predicates to event-expressing nouns such as `approval`, `construction`, and `cancellation`. It supplies both the contextual decision that a candidate noun expresses an event and the question-answer structures that connect a positive predicate to its participants and circumstances.

Primary references:

- Ayal Klein and colleagues. [QANom: Question-Answer driven SRL for Nominalizations](https://aclanthology.org/2020.coling-main.274/). COLING 2020.
- [QANom reference implementation and data](https://github.com/kleinay/QANom)

## Verified release identity

| Artifact | Verified source | Verification value |
| --- | --- | --- |
| `qanom_dataset.zip` | QANom repository revision `2bce70e8a39b40157ba97f38e1a8ae7619b30162` | SHA-256 `165c699ba0f8f9e4d093f7b879836e2a169f4fdf8d309fbeafc4f2d03e515cf5` |
| Reference code | `https://github.com/kleinay/QANom` | Git revision `2bce70e8a39b40157ba97f38e1a8ae7619b30162` |

The preparation command must verify the archive checksum before extraction. A changed archive is treated as a different release even when its filename remains the same.

## Release-computed statistics

The following values were computed from the verified CSV files. A candidate is the unique combination of `qasrl_id` and `target_idx`. A question row has a non-empty question field.

| Split | CSV rows | Sentences | Candidate nouns | Positive eventive nominals | Nonempty question rows | Retained distinct questions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 30,644 | 7,114 | 23,060 | 9,226 | 15,895 | 15,871 |
| Development | 7,660 | 1,557 | 4,661 | 2,616 | 5,577 | 5,577 |
| Test | 7,023 | 1,517 | 4,461 | 2,401 | 4,886 | 4,886 |
| **Total** | **45,327** | **10,188** | **32,182** | **14,243** | **26,358** | **26,334** |

The 24-row difference in training comes from exact duplicate question-answer rows. The adapter counts each duplicate and retains one canonical question rather than treating repeated bytes as additional supervision. The training sentences were sampled from the QA-SRL Bank 2.0 training split. Development and test sentences come from the QA-SRL Gold Standard source material. This overlap is useful for paired verbal–nominal analysis but creates a cross-task leakage risk if split identities are discarded.

## Data representation

The archive contains `annot.train.csv`, `annot.dev.csv`, and `annot.test.csv`. All three contain 22 shared semantic fields. Development adds `source_assign_id`; test instead adds an accidental saved-index column named `Unnamed: 0`. The adapter validates each parsed header by name rather than assuming one uniform column count.

The implemented adapter preserves:

- `qasrl_id`, sentence text, target token index, and candidate key;
- the eventive-nominal decision stored upstream as `is_verbal`;
- the related verbal form;
- the natural-language question and seven structured question slots;
- passive and negated question properties;
- all answer spans, including multiple spans separated upstream by `~!~`;
- worker provenance and the official split.

Internally, the ambiguous upstream field `is_verbal` is renamed `is_eventive_nominal`. It means that the candidate noun carries eventive or verbal semantics in context; it does not mean that the token itself is grammatically a verb. Token ranges use inclusive-start, exclusive-end boundaries.

Full-release processing records 472 case-normalized noun fields in training, 141 in development, and 114 in test. Development contains 57 retained eventivity/question conflicts and one blank answer string that is reconstructed from its verified token range. Test contains 1,141 nonempty values in the accidental `Unnamed: 0` column; those release-artifact values are discarded while their field and count remain in the manifest.

## Research split policy

- Preserve the official train, development, and test splits.
- Use development data for threshold selection and model selection only.
- Keep the official test split frozen until the nominal protocol is locked.
- Maintain QA-SRL sentence and document identifiers during joint training and leakage checks.
- Report candidate detection separately from argument extraction with a supplied positive predicate.
- Derive unseen-lemma analysis without moving official test examples into training.

The fixed cross-task comparison finds no direct source-ID or document-ID collision across roles and no development-to-test exact-text overlap. It does find copied evaluation text under different training IDs. The conservative document-level policy excludes 28 of 7,114 QANom training sentences across two affected documents; those documents are part of the same seven-document quarantine applied to QA-SRL training.

## Reference scorer behavior

At revision `2bce70e8a39b40157ba97f38e1a8ae7619b30162`, the reference implementation:

- greedily aligns argument spans in descending token intersection-over-union order;
- declares a match only when overlap is strictly greater than `0.3`;
- treats questions as equivalent when they map to the same coarse QA-SRL role;
- inner-joins gold and predicted predicate identifiers before evaluation;
- omits argument and role counts when the eventive-nominal decision is wrong.

The last two behaviors can hide the downstream impact of missed, spurious, or misclassified predicates. The project therefore implements two named modes:

1. **`qanom-reference-v1`:** reproduces the checked-in inner join, strict `> 0.3` boundary, descending-IoU greedy matching keyed by span value, coarse question-role equivalence, grouped-role alignment, and omission of argument and role counts when eventivity differs.
2. **`primary-end-to-end-v1`:** evaluates the union of gold and predicted candidates, uses inclusive `0.5` overlap and optimal one-to-one matching, and charges candidate and predicate failures to downstream argument precision and recall.

Exact token-span and exact character-offset metrics are reported alongside the overlap score. Every result records the scorer name, revision, threshold, matching algorithm, predicate source, and question-equivalence rule.

## Rights and publication boundary

The reference repository contains an MIT license and distributes the dataset archive, but the archive itself contains only the CSV files and a dataset README. It does not provide a separate license resolving the annotation and source-text rights for every Wikipedia and Wikinews sentence.

The repository will therefore publish:

- download and checksum-verification code;
- adapters and newly authored synthetic fixtures;
- preparation manifests, aggregate statistics, and source-lineage records;
- measured aggregate results when their use is permitted.

It will not publish raw or processed QANom records. A separate review must resolve the dataset, base-model, and derived-weight terms before any trained checkpoint is released.
