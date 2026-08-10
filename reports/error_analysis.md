# Error analysis

No corpus-level error analysis has been performed in the restored scope. The categories below are a preregistered taxonomy, not observed findings. Future reports should count reproducible categories and use only synthetic or redistributable examples.

## Rule-baseline taxonomy

- missing predicate vocabulary;
- verb/noun/adjective ambiguity;
- passive voice;
- coordination and embedded clauses;
- missing or overextended actor span;
- missing or overextended patient span;
- ambiguous qualifier relation;
- negation or modality not represented;
- pronoun/coreference failure;
- sentence-boundary failure;

## Planned supplied-predicate SRL taxonomy

- wrong PropBank role with correct boundary;
- missed or spurious argument;
- boundary error, including coordination and attachment;
- WordPiece alignment or prediction-collapse error;
- malformed BIO prediction and deterministic repair;
- truncation or dropped example;
- rare predicate or rare role;
- sentence-length and fragmentation effects.

## Public-data preparation audit

Before model analysis, report source-policy exclusions and adapter failures separately:

- document absent from the provenance-reviewed allowlist or denied by a source-family hold;
- missing or ambiguous document/sentence/parse join;
- unrecognized PropBank record dialect, pointer operator, role, or link type;
- invalid terminal/height address or predicate mismatch;
- unresolved empty argument or ambiguous trace chain;
- discontinuity, overlap, or conflict that cannot fit one gold BIO label per word;
- nonverbal predicate exclusion;
- duplicate pointer repair recorded by the adapter.

These counts measure dataset conversion coverage, not model quality. Reject the whole predicate instance when one gold role cannot be represented, and never reinterpret a failed role as gold `O`.

For each future evaluation, record counts, denominators, gold-span conversion policy, representative authorized examples, severity, and the planned response. Keep data-preparation, rule-baseline, supplied-predicate neural, and raw-text pipeline failures in separate tables.
