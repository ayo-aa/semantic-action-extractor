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

For each future evaluation, record counts, denominators, annotation view, representative authorized examples, severity, and the planned response. Keep rule-baseline, supplied-predicate neural, and raw-text pipeline failures in separate tables.
