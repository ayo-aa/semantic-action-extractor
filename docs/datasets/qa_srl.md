# QA-SRL verbal data card

## Status

| Field | Status |
| --- | --- |
| Release verification | Completed on 2026-08-06 |
| Training release | QA-SRL Bank 2.1, the official minor-fix release of Bank 2.0 |
| Primary evaluation release | QA-SRL Gold Standard |
| Adapter | Implemented and fixture-tested; every verified release file processed successfully |
| Repository data | None |
| Redistribution decision | Raw and processed records remain outside Git; checkpoint publication remains unresolved |

## Purpose

QA-SRL represents a verbal predicate’s semantic arguments with constrained natural-language questions and source-grounded answer spans. The Bank 2.0 paper reports more than 250,000 question-answer pairs for more than 64,000 sentences across Wikipedia, Wikinews, and TQA material. Bank 2.1 retains that task while making minor corrections to question-slot definitions.

The Gold Standard release provides higher-coverage development and test annotations for Wikipedia and Wikinews sentences. It is the primary verbal evaluation source because its collection protocol was designed specifically to address missing-role coverage in the original corpus.

Primary references:

- Nicholas FitzGerald, Julian Michael, Luheng He, and Luke Zettlemoyer. [Large-Scale QA-SRL Parsing](https://aclanthology.org/P18-1191/). ACL 2018.
- Paul Roit and colleagues. [Controlled Crowdsourcing for High-Quality QA-SRL Annotation](https://aclanthology.org/2020.acl-main.626/). ACL 2020.
- [QA-SRL tools and data repository](https://github.com/julianmichael/qasrl)
- [QA-SRL Gold Standard repository](https://github.com/plroit/qasrl-gs)

## Verified release identity

| Artifact | Verified source | Verification value |
| --- | --- | --- |
| QA-SRL Bank 2.1 archive | `https://qasrl.org/data/qasrl-v2_1.tar` | SHA-256 `407c4f5e554fbfcc446c2dbfee093312a6f443c53901bda0d0ce64ae3e7301cf` |
| QA-SRL Gold Standard archive | `https://qasrl.org/data/qasrl-gs.tar` | SHA-256 `2fdbe4274141842cdb22efc04b6330e01455e843dd1a3d6c779bd799e7110eda` |
| QA-SRL tools repository | `https://github.com/julianmichael/qasrl` | Git revision `16ab49490f2df837ce7069bb1344f09e730d9a4a` |
| Gold Standard repository | `https://github.com/plroit/qasrl-gs` | Git revision `f7c64ae9b6fe48ff3910c3e59850a12ec278bf83` |

The preparation command must verify the archive checksum before extraction. A changed upstream archive is a new data revision even when its URL is unchanged.

## Release-computed statistics

These counts were computed directly from the verified Bank 2.1 archive. A *question label* is one question entry in the JSONL release; it should not be confused with a separately adjudicated semantic fact or with the paper’s aggregate count.

| Layer | Split | Sentences | Predicates | Question labels | Labels with at least one valid judgment |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Train | 44,477 | 95,258 | 215,427 | 215,427 |
| Original | Development | 9,078 | 17,577 | 38,487 | 38,487 |
| Original | Test | 10,453 | 20,603 | 45,387 | 45,387 |
| Expanded | Train | 44,477 | 95,258 | 293,624 | 283,530 |
| Expanded | Development | 9,078 | 17,577 | 52,370 | 50,516 |
| Dense | Development | 2,614 | 5,886 | 33,967 | 32,350 |
| Dense | Test | 2,591 | 5,844 | 31,100 | 29,669 |

The verified Gold Standard JSONL archive contains:

| Split | Sentences | Predicates | Question labels | Sources |
| --- | ---: | ---: | ---: | --- |
| Development | 1,000 | 2,448 | 7,183 | 500 Wikipedia; 500 Wikinews |
| Test | 999 | 2,450 | 7,097 | 500 Wikipedia; 499 Wikinews |

The full adapter scan reproduces these sentence, predicate, and question counts. It also finds one judgment in the Bank 2.1 dense development file that is marked valid but contains no answer span. The canonical representation retains and counts that upstream anomaly without inventing an answer or silently dropping the judgment.

## Data representation

Bank 2.1 uses gzipped JSON Lines with one sentence object per line. The implemented adapter preserves:

- `sentenceId` and the pretokenized `sentenceTokens`;
- the predicate token index and inflected forms;
- the seven structured question slots and voice, tense, aspect, and negation fields;
- question provenance and every answer judgment;
- alternative token spans using inclusive-start, exclusive-end boundaries;
- the original layer and split.

Multiple answer judgments are annotation evidence, not independent training examples. The implemented `valid-judgment-union-v1` view retains a question when at least one judgment marks it valid, unions distinct answer alternatives from valid judgments, counts exact duplicate alternatives once, and records valid judgments with no answer without creating an artificial span. Raw judgments remain in canonical annotation JSONL.

## Research split policy

- Use verified Bank 2.1 expanded training annotations as the default verbal training source.
- Use the expanded development split for pipeline diagnostics only.
- Use the Gold Standard development split for model selection and scorer regression.
- Keep the Gold Standard test split frozen until the verbal protocol is locked.
- Report the dense Bank 2.1 splits as a secondary coverage analysis, not as a substitute for the Gold Standard.
- Derive unseen-lemma and held-out-domain analyses without moving official test examples into training.

QANom development and test examples were sampled from QA-SRL Gold Standard source sentences. The implemented joint-split check preserves this intended same-role overlap. It finds no development-to-test identity overlap, but it identifies copied evaluation text under different training IDs. The fixed document-level quarantine excludes 120 QA-SRL expanded-training sentences across seven documents while leaving Gold Standard development and test unchanged.

## Reference scorer behavior

The Gold Standard reference code at revision `f7c64ae9b6fe48ff3910c3e59850a12ec278bf83` specifies token-span intersection-over-union of at least `0.5` and maximum-weight one-to-one argument matching. Its current `evaluate.py` imports a `get_paraphrase_score` function that is absent from the checked-in `paraphrases.py`, so the labeled scorer is not directly runnable from that revision.

The project therefore implements two explicitly named modes:

1. **`qasrl-gs-compatible-v1`:** preserves inclusive `0.5` token overlap, maximum one-to-one matching, the Gold Standard predicate scope, and a frozen five-field question-equivalence contract. It is not described as byte-for-byte official because the pinned reference revision is not directly runnable.
2. **`primary-end-to-end-v1`:** scores the union of gold and predicted predicate identifiers, uses maximum-cardinality then maximum-IoU matching, and makes missed and spurious predicates contribute downstream false negatives and false positives.

Exact token-span and exact character-offset scores are reported alongside overlap-based scores. Every result records the scorer name, version, threshold, matching algorithm, and role-equivalence rule.

## Rights and publication boundary

The two code repositories contain MIT licenses, but neither downloaded data archive contains its own license or source-text terms. The papers identify Wikipedia, Wikinews, and TQA sources without establishing one uniform redistribution grant for all sentences and annotations.

The repository will therefore publish:

- download and checksum-verification code;
- adapters and newly authored synthetic fixtures;
- preparation manifests, aggregate statistics, and source-lineage records;
- measured aggregate results when their use is permitted.

It will not publish raw or processed QA-SRL records. A separate review must resolve the dataset, base-model, and derived-weight terms before any trained checkpoint is released.
