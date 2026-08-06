# Model card: `rule-based-v0`

## Summary

`rule-based-v0` is a dependency-free English baseline that converts short sentences into action frames. It is a deterministic program, not a trained statistical model.

It finds predicates from a documented verb vocabulary plus conservative inflection rules, then treats nearby text as an actor, patient, or prepositional qualifier. Every extracted span retains its exact location in the source text.

## Version

- Extractor ID: `rule-based-v0`
- Project version: `0.1.0`
- Training data: none
- Runtime dependencies: Python standard library only

## Intended uses

- inspecting the action-frame schema;
- prototyping operational-text workflows;
- generating candidates for human annotation;
- serving as a reproducible lower-bound baseline for trained models;
- demonstrating the CLI and API before a neural checkpoint is available.

## Out-of-scope uses

- autonomous decisions about people, employment, credit, health, safety, or legal status;
- treating heuristic confidence as a calibrated probability;
- unsupervised ingestion of sensitive text without a separate privacy review;
- high-recall event extraction or production automation without human validation;
- multilingual extraction.

## Input and output

Input is a UTF-8 string. Output is an `ExtractionResult` containing zero or more `ActionFrame` objects. Character positions are zero-based and end-exclusive.

The baseline emits prepositional qualifiers as a surface relation such as `to` or `on`; it does not pretend to know whether `on Tuesday` is temporal while `on the table` is locative.

## Confidence

Confidence is a transparent ranking heuristic based on which fields were found and whether the predicate is in the known vocabulary. It has not been calibrated and must not be interpreted as empirical correctness probability.

## Evaluation

Milestone one verifies:

- schema validation;
- exact character offsets;
- common regular and irregular verb lemmas;
- multiple-sentence extraction;
- configuration loading;
- JSON CLI behavior.

No corpus-level quality benchmark has been run. A results table belongs in `reports/results.md` only after an authorized, documented evaluation set exists.

## Known limitations

- English-centric tokenization and verb rules;
- best on short, active, declarative clauses;
- incomplete verb vocabulary;
- no syntactic parser or learned semantics;
- weak handling of coordination and embedded clauses;
- no reliable passive-voice, negation, modality, or coreference representation;
- no implicit arguments or predicate-sense disambiguation;
- ambiguous prepositions are preserved rather than semantically classified;
- punctuation-based sentence splitting does not handle every abbreviation;
- the same surface form can be a noun, adjective, or verb, so false positives remain possible.

## Risk mitigation

- keep source spans so users can audit every field;
- label the system and confidence as heuristic;
- return warnings instead of silently claiming no action exists;
- use human review for consequential workflows;
- evaluate domain-specific errors before deployment;
- replace or augment the baseline with a documented trained model only when its data and model rights are clear.

## Future neural model

A future model will directly reengineer the BERT SRL prototype behind the same schema. Its model card must separately document authorized training data, architecture, hyperparameters, hardware, benchmark protocol, subgroup/domain behavior, and checkpoint license. Results from the earlier restricted-data notebook are not results for this package.
