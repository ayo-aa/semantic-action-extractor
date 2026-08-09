# Model card: `rule-based-v1`

## Summary

`rule-based-v1` is a dependency-free English software baseline for source-grounded action extraction. It is a deterministic program, not a trained statistical model.

The baseline finds configured verbal predicates and returns nearby text as surface arguments. A surface argument records observable position or a preposition; it does not claim that the span is a semantic actor, patient, recipient, time, or location.

The term *action* means a linguistic action or event mention. The baseline does not detect assignments, commitments, due dates, or action-item status.

## Version

- Extractor ID: `rule-based-v1`
- Project version: `0.3.0`
- Schema version: `0.2.0`
- Training data: none
- Runtime dependencies: Python standard library only

## Intended uses

- inspecting the versioned predicate–argument schema;
- prototyping source-grounded text workflows with human review;
- generating candidates for annotation;
- providing a reproducible lower-bound comparator for trained models;
- verifying the CLI and API before a neural checkpoint exists.

## Out-of-scope uses

- treating surface positions as verified semantic roles;
- treating the heuristic score as a calibrated probability;
- extracting nominal predicates;
- determining whether an action is planned, requested, completed, negated, or hypothetical;
- autonomous decisions about employment, credit, health, safety, legal status, or another consequential domain;
- unsupervised ingestion of sensitive text without a separate privacy review;
- high-recall event extraction or production automation without human validation;
- multilingual extraction.

## Input and output

Input is one Unicode string. Output is an `ExtractionResult` containing zero or more `ActionFrame` objects. Offsets are zero-based, end-exclusive Unicode code-point indices, equivalent to Python string indices. They are not UTF-8 byte offsets or JavaScript UTF-16 code-unit offsets. The response schema verifies that each predicate, argument, and cue points to the exact source substring.

Every action contains:

- one verbal predicate span, lemma, and type;
- zero or more surface arguments;
- a sentence index;
- a heuristic completeness score and score type;
- the extractor identifier.

The role `before_predicate` means only that the span appears before the predicate in the current rule parse. The role `after_predicate` means only that the span appears after it and before the first recognized preposition. A role such as `to`, `on`, or `by` preserves the observed preposition and its value.

## Score

The score is a transparent completeness heuristic, not a prediction of correctness. Every detected predicate begins at `0.35`; left context adds `0.20`, direct right context adds `0.20`, and at least one prepositional argument adds `0.10`. Scores therefore range from `0.35` to `0.85` for emitted frames.

The rule baseline does not populate the optional predicate- or argument-confidence fields used by future neural systems. Its completeness score has not been calibrated and must not be interpreted as an empirical probability.

## Evaluation

The current automated suite verifies:

- schema and score validation;
- exact predicate, argument, and cue offsets;
- the role-neutral passive-voice representation;
- Unicode source spans;
- common regular and irregular verb lemmas;
- multiple-sentence extraction;
- coordinated-context inheritance without crossing hard clause boundaries;
- negation warnings;
- normalized custom verbs and score filtering;
- JSON CLI behavior.

No corpus-level quality benchmark has run. Unit tests establish expected software behavior but do not establish extraction accuracy, robustness, or business readiness.

## Known limitations

- Detects only configured verbs and conservative inflections.
- Does not detect nominal predicates such as `approval` or `cancellation`.
- Uses token position and prepositions instead of syntax or learned semantics.
- Performs best on short, active, declarative clauses.
- Handles coordination and embedded clauses weakly.
- Does not reliably represent passive voice, negation, modality, coreference, implicit arguments, or predicate senses.
- Preserves ambiguous prepositions instead of assigning semantic roles.
- Uses punctuation-based sentence boundaries that can fail on abbreviations.
- Can confuse the same surface form across noun, adjective, and verb uses.
- Does not expose calibrated uncertainty or an abstention policy.

## Observed failure examples

- In `The contract was emailed by Maya`, the baseline reports `The contract` as `before_predicate` and `Maya` under the surface cue `by`. It does not infer that Maya is the semantic sender.
- In `Maya did not approve the refund`, the baseline identifies `approve` and emits a negation warning but does not encode negative polarity.
- In multi-clause or embedded constructions, a surface argument can extend beyond the correct semantic boundary.

## Risk mitigation

- Preserve exact source spans so a reviewer can inspect every field.
- Label all baseline roles as surface observations.
- Label the score as heuristic completeness.
- Emit warnings for unsupported negation instead of silently claiming polarity.
- Require human review for consequential workflows.
- Evaluate representative authorized domain data before deployment.
- Give every future trained extractor its own model card, data lineage, evaluation, and checkpoint license.
