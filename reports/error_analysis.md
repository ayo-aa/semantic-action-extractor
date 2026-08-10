# Error analysis

This report tracks reproducible error categories rather than isolated anecdotes.

## Predicate detection

- missed eligible candidate during raw-text candidate generation;
- spurious candidate before classification;
- missed verbal predicate;
- spurious verbal predicate;
- missed nominal predicate;
- incorrect nominal eventivity decision;
- incorrect predicate boundary or lemma;
- incorrect related verbal form for a nominal predicate;
- duplicate or overlapping predicate frame.

## Argument extraction

- missing argument;
- spurious argument;
- underextended or overextended answer span;
- wrong or missing QA-SRL question slot;
- invalid realized QA-SRL role question;
- incorrect grouping of multiple answers under one question;
- duplicate or overlapping answer span;
- long-distance argument;
- passive-voice construction;
- coordination or embedded clause;
- implicit argument incorrectly invented or omitted.

## Mention qualifiers and interpretation boundaries

- missed supported qualifier kind;
- spurious qualifier kind without source support;
- correct qualifier kind with missing, underextended, overextended, or otherwise incorrect evidence boundaries;
- qualifier attached to the wrong predicate or given the wrong predicate scope;
- an assessed empty qualifier set confused with an unassessed predicate;
- co-occurring qualifier kinds incorrectly collapsed or made mutually exclusive;
- real-world occurrence, truth, completion, assignment, commitment, ownership, due date, or execution improperly inferred from a mention qualifier;
- assignment or action-item status otherwise incorrectly inferred;
- coreference or cross-sentence identity failure;
- ambiguous prepositional cue.

## Data and generalization

- unseen predicate lemma;
- held-out source domain;
- operational-style distribution shift;
- duplicated or leaked document;
- annotation disagreement or ambiguous reference.

## Data preparation

- release header or field-schema drift;
- invalid token range, source offset, or answer text;
- exact duplicate question-answer row;
- valid upstream judgment without an answer span;
- retained eventivity/question conflict;
- missing or inconsistent source or document identity;
- cross-role document, source, exact-text, lineage, or predicate-family overlap;
- disagreement introduced by the named consolidation rule;
- tied nominal eventivity judgments.

## Neural and serialization failures

- word-to-subword alignment error;
- predicate marker or feature misalignment;
- malformed BIO or span decoding;
- generative answer not found in the source;
- malformed or incomplete generated QA set;
- repeated or order-sensitive generated QA;
- predicate or argument lost through truncation;
- mention qualifier or qualifier evidence lost through truncation;
- source-offset mismatch;
- qualifier evidence that does not map exactly to the source;
- invalid schema or duplicate serialized frame;
- end-to-end failure caused by predicate detection rather than argument extraction.

## Calibration failures

- confident but incorrect predicate classification;
- confident but incorrect answer span;
- correct answer span with an overconfident wrong role question;
- underconfident correct prediction lost at a chosen coverage threshold;
- calibration drift between verbal and nominal predicates;
- calibration drift on held-out predicate families or source domains.

For each evaluation, record the count, denominator, representative redistributable examples, affected model and configuration, likely cause, and planned response. Attribute candidate-generation, predicate-classification, mention-qualifier kind, qualifier-evidence, argument-span, question-slot, and serialization errors to separate stages; report gold-predicate and complete-pipeline failures separately. Keep linguistic qualifier errors separate from unsupported claims about real-world event status.
