# Error analysis

No corpus-level model error analysis has been performed. The neural categories
below are preregistered; they are not observed findings. No corpus excerpt,
paraphrase, or derived example is included.

## BabySRL conversion accounting

The only current corpus-backed failure counts describe structural conversion,
not model predictions:

| Terminal conversion reason | Proposition columns |
| --- | ---: |
| `row_width_mismatch` | 4 |
| `invalid_bracket_sequence` | 15 |
| `missing_relation_span` | 99 |
| `ambiguous_predicate_head` | 5 |
| `unsupported_role_label` | 16 |
| **Rejected** | **139** |

Accepted 18,397 plus rejected 139 reconciles exactly to 18,536 declared
propositions, for 99.2501% structural coverage. A proposition with any failed
role is rejected in full; the adapter does not keep easy roles and relabel the
failure `O`.

The authorized manual conversion sample remains on hold pending TalkBank
registration and recorded acceptance of the current ground rules. Accordingly,
the structural pass is not yet a human validation claim.

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
automatically or by authorized review. Keep preparation, rule-baseline,
supplied-predicate, and raw-text tables separate.

Authorized review may inspect source context under the governing access rules,
but public reports must use counts, taxonomies, and independently invented
illustrations only. Blank findings must remain blank; historical course output
and synthetic predictions are not substitutes for an executed experiment.
