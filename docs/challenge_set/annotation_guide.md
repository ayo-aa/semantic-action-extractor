# Pilot annotation guide

Guide version: `candidate-pilot-v1`

## Core decision

Annotate event mentions, not real-world status. A predicate remains eligible
when it is negated, possible, planned, hypothetical, conditional, questioned,
or attributed to someone else. The annotation says only what the note mentions.

## Record-level workflow

1. Read the complete note without model output.
2. Mark every verbal and nominal predicate candidate.
3. Decide whether each nominal candidate is eventive in this context.
4. Record the exact predicate span, lemma, predicate type, and related verbal
   form when supported.
5. Add predicate-local mention qualifiers and their exact cue spans.
6. Write QA-SRL role questions and mark every exact answer span supported by the
   note.
7. Record ambiguity or exclude the item when a guide-based decision is not
   defensible.

QA-SRL is an internal role notation. It is used because questions preserve
semantic distinctions such as actor, recipient, object, time, and location
without forcing them into a small product schema. It is not a customer-facing
question-answering feature.

## Predicate eligibility

- Verbal predicates: include verbs that evoke an event mention.
- Nominal predicates: include nouns that evoke an event in the sentence, such as
  `approval` in `Priya's approval of the refund`.
- Hard negatives: retain plausible nominal candidates that are non-eventive in
  context, such as `plan` in `The plan is on the desk`.
- Do not infer implicit events that lack a source-grounded predicate span.

Use the shortest complete predicate span. Offsets are zero-based, end-exclusive
Unicode code-point indices.

## Mention qualifiers

Each qualifier has one kind and one or more exact evidence spans:

| Kind | Use when the event mention is framed by | Typical cues |
| --- | --- | --- |
| `negated` | event-scoped negation | `not`, `never`, `no` |
| `possible` | explicit possibility | `may`, `might`, `could` |
| `necessary` | explicit necessity or obligation | `must`, `required to` |
| `planned` | explicit intention or plan | `plans to`, `intends to` |
| `future` | explicit future marking | `will`, `shall` |
| `hypothetical` | a hypothetical frame | `would`, `suppose` |
| `conditional` | an explicit condition | `if`, `unless` |
| `questioned` | the event proposition is questioned | `whether`, `?` |
| `reported` | the event is attributed or reported | `reported`, `said`, `according to` |

Qualifiers are attached separately to each predicate. In `Jordan reported that
Maya might not approve the refund`, `approve` can carry `reported`, `possible`,
and `negated`, each with its own cue evidence.

Do not use these labels to say that the event happened. An empty qualifier list
means no supported cue was annotated, not that the event is true or complete.

The workbook records qualifier assessment explicitly. Set
`qualifier_assessed` to `false` and add no `Qualifier Evidence` rows when the
assessment was not performed. Set it to `true` with no evidence rows when
assessment found no supported cue, or to `true` and add one or more evidence
rows when cues were found. Non-eventive nominal candidates use `false`.

When scope is unresolved—for example, `Maya does not have to approve`—mark the
item ambiguous and exclude it. Record quarantine then removes the complete
source from the guide-development evaluation corpus.

## Role questions and answers

Preserve all seven QA-SRL slots: `wh`, `aux`, `subj`, `verb`, `obj`, `prep`, and
`obj2`. Use `_` for an unused slot, but every pilot question requires a supported
question-word slot and a non-placeholder abstract verb slot. Its surface form
must end in `?`. Every answer must copy exact source text and retain its token
and character boundaries. Coordinated or discontinuous answers remain grouped
rather than being flattened into unrelated arguments.

Examples:

- `Maya did not approve the refund.` Predicate: `approve`; qualifier:
  `negated` with cue `not`; roles include who approved and what was approved.
- `After Priya's approval of the refund, Jordan emailed the customer.`
  Predicates: nominal `approval` and verbal `emailed`; annotate their roles
  independently.
- `The plan is on the desk.` Candidate: nominal `plan`; eventivity: false; no
  event roles or mention qualifiers.

## Two-pass pilot procedure

- Pass 1: annotate all 20 records and save the completed file.
- Wait 7–14 days.
- Pass 2: use the separately shuffled file without opening pass 1.
- Compare only after pass 2 is complete.
- Keep both per-pass conversion files unchanged. Record every difference and
  resulting guide change in the Review Log of a separate comparison copy after
  both pass files have validated and converted.

The comparison is intra-annotator repeatability only. Do not call it independent
agreement or adjudication.

## Workbook contract

The checked-in workbook is the accepted, blank annotation template. Its source
fingerprint pins the 20 Authoring rows so changed text, identifiers, or release
metadata cannot silently pass as the accepted pilot. Four normalized tables keep
independent one-to-many relationships separate:

- `Annotation` contains one predicate candidate per row, including its stable
  candidate ID, exact predicate span, type, eventivity decision, qualifier
  assessment state, inclusion decision, and rationale.
- `Qualifier Evidence` contains zero or more ordered cue spans for one candidate
  and qualifier kind.
- `Questions` contains one stable question ID, its surface form, all seven
  QA-SRL slots, and explicit passive and negated flags.
- `Answer Spans` contains one contiguous answer span per row. `alternative_id`
  groups spans into one answer, and zero-based `span_order` preserves grouped or
  discontinuous evidence without flattening it.

One workbook represents exactly one annotator and one pass. Character offsets
remain zero-based and end-exclusive. The validator checks each copied span
against the unchanged note, derives token spans with
`pilot-source-tokenizer-v1`, and rejects partial-token spans, broken foreign
keys, mixed annotator/pass rows, duplicate semantic questions, incomplete
eventive candidates, formulas in annotation data, and structural workbook
drift. The blank template validates as ready for annotation but cannot be
converted.

Conversion requires at least one candidate for every Authoring record, a
completed qualifier decision for every included eventive candidate, and at
least one answered role question for every included eventive candidate. It
produces a versioned evaluation bundle marked `single-annotator`,
`guide-development-only`, and `not_for_model_selection`. If any candidate is
excluded, `challenge-record-quarantine-v1` omits that entire source record from
the scorer-ready corpus while preserving its source and candidate audit
metadata. Prediction bundles must declare the same source fingerprint and
tokenizer version, and bundles containing quarantined source IDs are rejected.

The converter handles one pass only and requires its Review Log to remain
inactive. It does not compare passes, consume the separate comparison log,
adjudicate disagreements, or create gold or frozen data. Human pass 1 and the
later clean repeat pass remain pending.
