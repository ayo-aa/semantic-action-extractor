# Error analysis

No corpus-level model error analysis has been performed. The neural categories
below are preregistered; they are not observed findings. No corpus excerpt,
paraphrase, or derived example is included.

## EWT preparation accounting

The current corpus-backed counts describe completed private preparation and
frozen pre-outcome controls, not model predictions:

| Stage or exclusion | Examples |
| --- | ---: |
| All PropBank predicate columns | 50,262 |
| Nonverbal predicate columns excluded | 11,623 |
| Structurally valid verbal predicates | 38,639 |
| Token-width mismatch excluded | 4 |
| Word-aligned predicates | 38,635 |
| Cross-split duplicate-text examples excluded | 76 |
| Identical-input conflicting examples excluded | 31 |
| Repeated development/test semantics removed | 42 |
| Prepared examples written to ignored storage | 38,486 |
| Train examples over `max_length=128` | 62 |
| Modeled examples | 38,424 |

The prepared-eligible split is 31,101 train, 3,775 development, and 3,610 test;
the modeled split is 31,039/3,775/3,610. Nonconflicting train frequency is
preserved, while exact repeated semantics receive no extra evaluation weight.
The pinned tokenizer preflight builds 111 train-derived labels, including `O`
and continuation closure, with no development or test label outside train. The
prepared fingerprint is
`2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b`;
split and provenance hashes are recorded in the aggregate-only
[preparation report](ewt_preparation.md).

The [aggregate-only private source review](ewt_private_source_review.md)
inspected all 13 predicate-anchor divergences, the one token-width mismatch,
and 30/30 deterministically selected aligned verbal records. It requires no
account, registration, or user manual review. This is a validated inferred
PropBank-skeleton/UD-word join, not the official LDC mapping. Preparation and
one real-data optimizer-step preflight are complete; no paired training run or
model error result exists yet.

## Rule-baseline taxonomy

Future rule-baseline analysis will count:

- missing predicate vocabulary;
- verb/noun/adjective ambiguity;
- passive voice;
- coordination and embedded clauses;
- missing or overextended actor span;
- missing or overextended patient span;
- ambiguous qualifier relation;
- negation or modality not represented;
- pronoun/coreference failure; and
- sentence-boundary failure.

These action-record categories are not PropBank-role errors.

## Supplied-predicate neural taxonomy

Future neural analysis will count mutually intelligible categories while
retaining the exact primary metric:

- correct boundary with the wrong PropBank role;
- missed gold argument;
- spurious predicted argument;
- left, right, or both-boundary error;
- coordination or discontinuity error;
- attachment or subordinate-clause error;
- punctuation-boundary error;
- rare role or rare predicate;
- sentence-length or WordPiece-fragmentation effect;
- overlength exclusion before collation;
- malformed predicted BIO transition and deterministic repair;
- missing `B-V` at the supplied predicate;
- spurious `V`/`C-V` away from the supplied predicate; and
- predicted argument span overlapping a gold predicate word.

Errors will also be stratified by role and support. First-subword collapse is a
fixed decoding policy, so any associated failures must be reported rather than
changed after inspecting test predictions.

## Raw-text pipeline taxonomy

If the rule proposer is connected to the neural model, analysis will separately
count:

- missed annotated predicate candidate;
- spurious predicate candidate;
- correct candidate with incorrect arguments; and
- duplicate or conflicting frames emitted for one source span.

Raw-text failures must never be merged into supplied-predicate role metrics.

## Reporting protocol

For every future analysis, report the exact data and configuration fingerprints,
seed and variant, checkpoint-selection rule, denominator, counts, role support,
overlength and BIO-repair totals, and whether the category was assigned
automatically or by private review. Keep preparation, rule-baseline,
supplied-predicate, and raw-text tables separate.

Private review may inspect source context under the governing source terms,
but public reports must use counts, taxonomies, and independently invented
illustrations only. Blank findings must remain blank; historical course output
and synthetic predictions are not substitutes for an executed experiment.
