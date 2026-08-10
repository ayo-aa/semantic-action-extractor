# Operational-style challenge-set protocol

Protocol version: `candidate-pilot-v1`

Status: candidate note authoring and the normalized workbook conversion
contract are implemented and tested; human annotation is pending. This protocol
does not complete E1 and does not authorize model selection on challenge data.

## Purpose

The challenge set tests whether the extractor can find source-grounded verbal
and nominal event mentions in short notes resembling support and operations
work. The primary product story is a human-reviewed case timeline: one note is
processed at a time, event records retain exact evidence, and a reviewer accepts,
edits, or rejects every suggestion.

The target is a linguistic event mention, not a claim that an event occurred.
Negated, possible, necessary, planned, future, hypothetical, conditional,
questioned, and reported mentions remain eligible. Qualifiers describe how the
source presents the mention; they do not establish truth, occurrence,
completion, assignment, commitment, ownership, due dates, or execution.

## Pilot design

The pilot contains 20 newly authored synthetic records and is excluded from all
reported challenge results. The records were accepted for candidate-pilot
publication under Ayo Adetayo's explicit protocol-level delegation on
2026-08-09; no direct note-level review occurred. This delegation is not an
independent annotation, adjudication, or freeze decision. The notes contain no
copied tickets, private correspondence, customer data, school material, or
identifying personal information.

Four records are planned for each scenario:

1. customer-support case notes;
2. IT incident and change notes;
3. procurement, billing, and approval notes;
4. fulfillment and logistics updates; and
5. project, meeting, and administrative updates.

The pilot must cover verbal and nominal predicates, passive voice, coordination,
multiple predicates, negation, modality, reporting, temporal circumstances,
long-distance arguments, and non-eventive nominal candidates. Tags may overlap.

## Annotation passes without a second annotator

A human annotator completes pass 1, then repeats the same pilot after a 7–14 day
washout. Pass 2 uses a shuffled record order and is completed without viewing
pass 1 or any model output. Differences are used only to revise this guide and
report intra-annotator repeatability. Each pass uses a separate workbook with
one annotator/pass identity. The implemented validator and converter preserve
the normalized candidate, qualifier-evidence, question, and answer-span tables
without relying on worksheet row order. Both per-pass workbooks keep the Review
Log inactive; after they have validated and converted, differences are recorded
in the Review Log of a separate comparison copy that is not converter input.

This procedure is not independent annotation, inter-annotator agreement, or
adjudication. The pilot cannot be called gold, final, or frozen.

## Annotation representation

Every retained event mention records:

- an exact verbal or nominal predicate span;
- lemma and, for nominals, a related verbal form when supported;
- QA-SRL role questions and exact answer spans;
- zero or more typed `mention_qualifiers`, each linked to exact cue text; and
- scenario, phenomenon, ambiguity, source-rights, and lineage metadata.

The qualifier kinds are `negated`, `possible`, `necessary`, `planned`, `future`,
`hypothetical`, `conditional`, `questioned`, and `reported`. Qualifiers are
predicate-local and can co-occur. `None` means qualifier assessment was not
performed; an empty list means assessment was performed and no supported cue
was found.

The workbook records one candidate per `Annotation` row. Independent
`Qualifier Evidence`, `Questions`, and `Answer Spans` tables use explicit
candidate, question, alternative, and span-order identifiers. Exact character
spans are checked against the pinned note text and mapped to token spans with the
frozen `pilot-source-tokenizer-v1` contract. A completed pass converts directly
to an evaluation bundle marked as single-annotator, guide-development-only, and
not for model selection.

If an annotation decision excludes any candidate, the fixed
`challenge-record-quarantine-v1` policy removes the entire source record from
the scorer-ready corpus. This prevents an ignored candidate from becoming a
prediction-only false positive while retaining source and decision metadata for
audit. The same source IDs must be absent from prediction bundles.
Prediction bundles must also declare the matching Authoring fingerprint and
tokenizer version before scoring.

## Evaluation contracts

The challenge report will lead with raw-text predicate detection and
`primary-end-to-end-v1` labeled relation F1. It will also report verbal and
nominal slices, exact token and character spans, and the separately versioned
mention-qualifier diagnostics:

- qualifier-kind precision, recall, and F1; and
- exact grounded qualifier-evidence precision, recall, and F1.

Reference-compatible QA-SRL and QANom modes remain unchanged and do not score
mention qualifiers. Displayed source spans must have 100% offset validity.

## Split and freeze rules

The provisional scored design is 50 development records and 150 sealed test
records, with no challenge-set training split. Documents, exact or derived
texts, template families, and predicate families remain within one split. Test
text and labels remain hidden from model and threshold selection.

No second annotator is available for this pilot. It therefore remains a
guide-development exercise excluded from model selection and reported results,
even after the repeat pass. A future scored challenge set would require a
separate independent annotation effort covering 100% of retained development
and test records without exposure to Ayo's labels or model predictions.
Pre-resolution agreement and every disagreement resolution would need to be
recorded, followed by regenerated text, annotation, exclusion, split, and
SHA-256 fingerprints. Only that independently reviewed artifact could be named
`frozen-v1`.

## Stop conditions

- Revise the guide and repeat the pilot when repeated labels reveal a systematic
  policy ambiguity.
- Exclude any source with uncertain release rights or privacy risk.
- Exclude unresolved qualifier-scope cases rather than inventing factual status.
- Retire and replace a test set if it is inspected for iterative model or
  threshold tuning.
- Call all synthetic results operational-style, never operational-domain
  performance.
