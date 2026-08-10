# EWT private source review

Status: **PASS as aggregate supporting evidence for the pinned inferred join; no corpus text or identifiers published**

Review date: 2026-08-10

## Scope and non-claim

This review privately inspected the exact source revisions selected by the
project:

- PropBank release `4abade0b53ce4a181e1d98b3518101c1a44d395a`; and
- UD English EWT r2.2 `6e064999a75b9c941c515ce1be98352e6f9831e0`.

It provides targeted supporting evidence that the public PropBank skeleton
positions and public UD word rows are being joined consistently. It is not a
comparison against the hidden LDC2012T13 words and cannot prove their
word-for-word identity. No raw text, token sequence, document ID, sentence ID,
example ID, roleset, source path, or review item is included in this report.

No prepared dataset was written, no training ran, and no model result was
produced during the review.

## Exhaustive exception review

The reviewer inspected every aggregate exception that could directly challenge
the predicate anchor or sentence width:

| Review population | Inspected | Outcome |
| --- | ---: | --- |
| Metadata-row versus primary-`V`-start divergences | 13 / 13 | All support using the unique primary `V` span start as `predicate_index`; only one is verbal under the primary row's XPOS |
| PropBank/UD token-width mismatches | 1 / 1 | The 36-versus-35 sentence remains excluded atomically with all four verbal predicates |

The 13/13 result is the direct reason the corrected adapter anchors eligibility
and `predicate_index` at the primary `V` start rather than at the lemma/roleset
metadata row. A discontinuous predicate may carry metadata on a `C-V` row.

## Deterministic aligned-record review

The review selected aligned verbal records by SHA-256 order without using model
outcomes. The versioned selection contract is:

| Field | Frozen value |
| --- | --- |
| Schema | `ewt-private-source-review-selection/v1` |
| Selection commitment SHA-256 | `08bb65aa8ceb88d7d40d5d8b572409c440bf2503c46b041b15133da880127509` |
| Distinct records reviewed | 30 |

Candidate strata were selected in this order and deduplicated by stable record
identity so overlapping strata could not inflate the denominator:

1. four general aligned verbal records from each split: 12 candidates;
2. the one aligned verbal metadata/primary-anchor divergence;
3. six additional records with multi-token primary `V` spans;
4. six records from equal-width sentences whose PropBank and UD XPOS sequences
   differ; and
5. six records where neither the metadata lemma nor the roleset lemma exactly
   matches the UD lemma at the aligned primary-`V` position.

The candidate categories overlap once; stable overlap removal produces the
committed 30 distinct records.

## Outcome

All 30 / 30 reviewed records had bounded, balanced span placement at the same
token width. The review found no systematic positional shift and no recurring
left- or right-off-by-one pattern.

The targeted lexical and POS discrepancies were localized cases of
normalization, orthography, contraction or nonstandard form, or annotation
differences. They did not form a consistent positional shift in the inspected
records.

## Interpretation

Together, the exhaustive exception checks and deterministic 30-record review
support the pinned adapter's fail-closed inferred join. They do not eliminate
the fundamental limitation that the public PropBank skeleton contains `[WORD]`
placeholders rather than the hidden LDC words. The source pins, conversion rule,
or selection schema changing would require a new private review and a new
commitment hash.

Only this aggregate report may be tracked. Any private review material remains
outside Git and must not be published.
