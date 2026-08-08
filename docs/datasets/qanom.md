# QANom data card

## Status

**Release audit:** Completed on 2026-08-06
**Dataset revision:** Archive distributed by QANom repository revision `2bce70e8a39b40157ba97f38e1a8ae7619b30162`
**Adapter:** Not implemented
**Repository data:** None
**Redistribution decision:** Raw and processed records remain outside Git; checkpoint publication remains unresolved

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

| Split | CSV rows | Sentences | Candidate nouns | Positive eventive nominals | Question rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 30,644 | 7,114 | 23,060 | 9,226 | 15,895 |
| Development | 7,660 | 1,557 | 4,661 | 2,616 | 5,577 |
| Test | 7,023 | 1,517 | 4,461 | 2,401 | 4,886 |
| **Total** | **45,327** | **10,188** | **32,182** | **14,243** | **26,358** |

The training sentences were sampled from the QA-SRL Bank 2.0 training split. Development and test sentences come from the QA-SRL Gold Standard source material. This overlap is useful for paired verbal–nominal analysis but creates a cross-task leakage risk if split identities are discarded.

## Data representation

The archive contains `annot.train.csv`, `annot.dev.csv`, and `annot.test.csv`. The actual files have 22 fields, so the adapter must follow the parsed header rather than assume that older prose documentation lists every column.

The adapter must preserve:

- `qasrl_id`, sentence text, target token index, and candidate key;
- the eventive-nominal decision stored upstream as `is_verbal`;
- the related verbal form;
- the natural-language question and seven structured question slots;
- passive and negated question properties;
- all answer spans, including multiple spans separated upstream by `~!~`;
- worker provenance and the official split.

Internally, the ambiguous upstream field `is_verbal` is renamed `is_eventive_nominal`. It means that the candidate noun carries eventive or verbal semantics in context; it does not mean that the token itself is grammatically a verb. Token ranges use inclusive-start, exclusive-end boundaries.

## Research split policy

- Preserve the official train, development, and test splits.
- Use development data for threshold selection and model selection only.
- Keep the official test split frozen until the nominal protocol is locked.
- Maintain QA-SRL sentence and document identifiers during joint training and leakage checks.
- Report candidate detection separately from argument extraction with a supplied positive predicate.
- Derive unseen-lemma analysis without moving official test examples into training.

## Scoring audit

At revision `2bce70e8a39b40157ba97f38e1a8ae7619b30162`, the reference implementation:

- greedily aligns argument spans in descending token intersection-over-union order;
- declares a match only when overlap is strictly greater than `0.3`;
- treats questions as equivalent when they map to the same coarse QA-SRL role;
- inner-joins gold and predicted predicate identifiers before evaluation;
- omits argument and role counts when the eventive-nominal decision is wrong.

The last two behaviors can hide the downstream impact of missed, spurious, or misclassified predicates. E1 must therefore provide two named modes:

1. **QANom reference:** reproduce the checked-in behavior for comparison with prior work, including the strict `> 0.3` boundary.
2. **End-to-end:** evaluate the union of gold and predicted candidates and charge predicate-detection failures to downstream argument precision and recall.

Exact token-span and exact character-offset metrics are reported alongside the overlap score. Every result records the scorer name, revision, threshold, matching algorithm, predicate source, and question-equivalence rule.

## Rights and publication boundary

The reference repository contains an MIT license and distributes the dataset archive, but the archive itself contains only the CSV files and a dataset README. It does not provide a separate license resolving the annotation and source-text rights for every Wikipedia and Wikinews sentence.

The repository will therefore publish:

- download and checksum-verification code;
- adapters and newly authored synthetic fixtures;
- preparation manifests, aggregate statistics, and provenance records;
- measured aggregate results when their use is permitted.

It will not publish raw or processed QANom records. A separate review must resolve the dataset, base-model, and derived-weight terms before any trained checkpoint is released.
