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

## Interpretation boundaries

- negation not represented;
- modality or hypothetical language not represented;
- assignment or action-item status incorrectly inferred;
- coreference or cross-sentence identity failure;
- ambiguous prepositional cue.

## Data and generalization

- unseen predicate lemma;
- held-out source domain;
- operational-style distribution shift;
- duplicated or leaked document;
- annotation disagreement or ambiguous reference.

## Neural and serialization failures

- word-to-subword alignment error;
- predicate marker or feature misalignment;
- malformed BIO or span decoding;
- generative answer not found in the source;
- malformed or incomplete generated QA set;
- repeated or order-sensitive generated QA;
- predicate or argument lost through truncation;
- source-offset mismatch;
- invalid schema or duplicate serialized frame;
- end-to-end failure caused by predicate detection rather than argument extraction.

## Calibration failures

- confident but incorrect predicate classification;
- confident but incorrect answer span;
- correct answer span with an overconfident wrong role question;
- underconfident correct prediction removed by selective review;
- calibration drift between verbal and nominal predicates;
- calibration drift on held-out predicate families or source domains.

For each evaluation, record the count, denominator, representative redistributable examples, affected model and configuration, likely cause, and planned response. Attribute candidate-generation, predicate-classification, argument-span, question-slot, and serialization errors to separate stages; report gold-predicate and complete-pipeline failures separately.
