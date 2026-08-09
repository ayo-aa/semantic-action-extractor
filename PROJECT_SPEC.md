# Project specification

## Objective

Semantic Action Extractor converts unstructured English text into source-grounded event records for structured information extraction. Each record identifies a verbal or nominal predicate, the participants or circumstances connected to it, and the exact supporting text without inventing unsupported `actor` or `patient` labels. QA-SRL questions serve as the supervised research representation rather than a user-facing question-answering interface.

The target pipeline is:

```text
text
  -> verbal and nominal candidate generation
  -> predicate or eventivity classification
  -> one selected predicate in context
  -> predicate-conditioned argument-span detection
  -> seven-slot QA-SRL role prediction for research evaluation
  -> grounded event record + grouped spans + exact source offsets + confidence
  -> optional separately evaluated product-role projection
```

The output can support human-reviewed search, case timelines, structured-field suggestions, annotation, and research analysis. The project does not equate semantic events with work assignments.

## Terminology and task boundary

- A **predicate** is the word or phrase that expresses an action or event.
- A **verbal predicate** expresses the event as a verb, such as `approved`.
- A **nominal predicate** expresses the event as a noun, such as `approval`.
- A nominal predicate’s **related verbal form** is the verb used to express its QA-SRL questions; for `approval`, the nominal lemma is `approval` and the related verbal form is `approve`.
- An **argument** is a participant or circumstance linked to the predicate.
- A **role question** is the internal QA-SRL label that describes that link, such as `who approved something?`; it is not a question that an end user must supply.
- A **source-grounded span** copies text from the input and records zero-based, end-exclusive Unicode code-point offsets.

The project performs source-grounded event and information extraction using an open predicate–argument representation. It is not a typed event ontology, intent classifier, action-item detector, assignment tracker, or system for determining whether work is planned or complete.

## Prior result and research gap

[QASem Parsing](https://aclanthology.org/2022.emnlp-main.528/) already demonstrated that one text-to-text model can generate verbal and nominal QA-based semantic representations. It compared separate and joint QA-SRL/QANom training, predicate-marker variants, output linearization, and cross-domain transfer. This project therefore does not claim that unified verbal and nominal parsing is new.

The practical task is structured event and information extraction from raw text. QA-SRL and QANom are used as supervision because their questions provide interpretable role labels and their answers provide exact source spans. QASem’s generative design can produce fluent question-answer strings, but generated answers must later be aligned back to the source. A structured encoder can instead select answer spans from the input by construction and predict the constrained question slots separately. The open question is whether that constraint improves grounding, reliability, efficiency, and transfer without sacrificing semantic coverage.

## Research question

> Under matched data and compute, how does a source-constrained event extractor trained with QA-SRL and QANom supervision compare with a generative QASem parser on labeled extraction quality, exact source grounding, calibration, efficiency, and transfer to unseen predicates and operational-style text—and which predicate-conditioning method is most robust?

## Hypotheses

- **H1 — grounding:** A source-constrained encoder makes returned evidence source-valid by construction while remaining competitive with a generative parser on labeled QA-pair F1.
- **H2 — predicate signal:** Explicit predicate conditioning outperforms an otherwise identical encoder that receives no target-predicate signal.
- **H3 — conditioning method:** Boundary markers or learned predicate features outperform repurposed BERT token-type embeddings on unseen-lemma transfer; marker-based conditioning also transfers to an encoder without segment embeddings.
- **H4 — joint learning:** Balanced joint verbal and nominal training improves nominal labeled F1 while keeping the paired verbal change within a one-point non-inferiority margin.
- **H5 — hidden generalization gap:** Random in-domain results overstate performance on held-out predicate families and excluded source domains.
- **H6 — calibrated confidence:** Development-set calibration improves confidence–coverage behavior without changing extraction accuracy.
- **H7 — systems tradeoff:** The structured encoder reduces decoding latency and malformed-output handling relative to autoregressive generation.

## Two schemas with different purposes

### Public inference schema

The versioned `ActionFrame` response remains intentionally compact. The research interface may expose QA-SRL role labels, while a separately evaluated serving adapter can project them into simpler fields. The response contains:

- a source-grounded predicate span, lemma, and `verbal` or `nominal` type;
- an optional related verbal form;
- zero or more source-grounded arguments;
- a role and role scheme for each argument;
- an optional group identifier that binds multiple answer spans to one role question;
- a required typed frame score plus optional typed predicate and argument confidence;
- an optional source-grounded cue, plus a sentence index and extractor identifier.

The public schema supports three role schemes:

| Role scheme | Meaning | Example |
| --- | --- | --- |
| `surface` | The rule baseline reports observable position or a preposition without semantic interpretation. | `before_predicate`, `to` |
| `qa_srl` | A trained system returns a dataset-compatible natural-language role question. | `who approved something?` |
| `coarse` | A separately evaluated adapter derives a simplified product-facing role. | `time`, `location` |

### Research annotation schema

Dataset preparation uses a separate evidence-preserving canonical representation. It preserves release and split identity, source and document identifiers, token boundaries, predicate candidates, verbal inflection paradigms, negative nominal-eventivity decisions, all seven question slots, question and answer source information, the grammatical fields supplied by each release, grouped or discontinuous answers, alternative spans, and multiple annotation judgments. Documented normalizations convert empty slots to `_`, reconstruct canonical text from release tokens, deduplicate byte-identical QANom rows while counting them, and discard release-only index values while retaining their counts.

This separation prevents training evidence from being flattened merely to fit a simple inference response. Adapters convert verified dataset records into the research schema; model predictions convert into the public schema only at the serving boundary.

## Data plan

### QA-SRL verbal data

QA-SRL Bank 2.1 supplies verbal predicates, constrained role questions, and answer spans for training. QA-SRL Gold Standard provides the primary verbal development and test evaluation.

Primary references:

- [Large-Scale QA-SRL Parsing](https://aclanthology.org/P18-1191/)
- [Controlled Crowdsourcing for High-Quality QA-SRL Annotation](https://aclanthology.org/2020.acl-main.626/)
- [QA-SRL tools and data repository](https://github.com/julianmichael/qasrl)
- [QA-SRL Gold Standard repository](https://github.com/plroit/qasrl-gs)

### QANom

QANom supplies lexical nominal candidates, contextual eventivity decisions, related verbal forms, role questions, and answer spans. It does not by itself represent unrestricted discovery of every possible eventive noun from raw text.

Primary references:

- [QANom: Question-Answer driven SRL for Nominalizations](https://aclanthology.org/2020.coling-main.274/)
- [QANom reference repository](https://github.com/kleinay/QANom)

### Operational-style challenge set

A small, frozen challenge set will test constructions resembling support and operations notes that the research corpora may not represent. It will contain newly authored or explicitly licensed text. Unless it contains representative real operational text, results will be described as **operational-style** performance rather than proof of operational-domain performance.

Annotation guidance, adjudication, exclusions, and a held-out test split will be fixed before model development uses the set.

### Data controls

Each adapter records the release, source terms, official split, document identifiers, checksums, preprocessing revision, excluded records, and derived-artifact rights. QA-SRL and QANom source identifiers remain intact because their development and test material overlaps. The selected protocol freezes development and test first, hashes adapter-canonicalized text exactly, and quarantines every training record from a document that shares either a document ID or exact text with evaluation. The verified releases quarantine 120 QA-SRL and 28 QANom training sentences while leaving evaluation unchanged.

Restricted OntoNotes-derived course data, outputs, and checkpoints remain outside the public project.

## System architecture

### Candidate generation and predicate classification

The complete pipeline separates three decisions:

1. **Candidate generation:** identify eligible verbal tokens and lexically related nominal candidates in raw text.
2. **Predicate classification:** decide whether a verbal candidate is in task scope and whether a nominal candidate is eventive in context.
3. **Argument extraction:** given one positive predicate, recover its question-answer relations.

Candidate-generator recall, predicate-classifier F1, supplied-predicate argument F1, and complete-pipeline F1 are reported separately. A supplied predicate is a component diagnostic, not evidence that the raw-text system solved predicate discovery. Verbal eligibility rules explicitly address auxiliaries, light verbs, and stative constructions.

### Source-constrained encoder parser

For one predicate in context, the model performs:

1. contextual encoding with an explicit target-predicate signal;
2. answer-span proposal or BIO-style span detection;
3. pooling of each predicate–span pair;
4. constrained prediction of the seven QA-SRL question slots;
5. deterministic realization of the role question;
6. grouping of multiple answer spans that share one question through an explicit group identifier;
7. token-to-character offset reconstruction and schema validation.

The implementation owns token-to-subword alignment, predicate encoding, span decoding, question-slot constraints, source reconstruction, confidence calculation, checkpointing, and evaluation.

The main BERT-family encoder supports the predecessor notebook’s token-type predicate indicator. A matched RoBERTa-family portability run tests marker and learned-feature conditioning without relying on segment embeddings.

### Generative comparison system

A reproduced T5-small QASem parser is a required baseline. It receives the same predicate-level examples and generates the complete QA set. Its answers are aligned back to the source with a declared procedure, and invalid, duplicate, or ungrounded generations remain measurable failures rather than silently repaired outputs.

The report presents two comparisons. A published-style reproduction follows the documented QASem training recipe closely enough to check prior results. A controlled head-to-head comparison then uses the same prepared examples, splits, model-selection rule, and declared training-FLOP allowance. Parameter counts and achieved training compute are reported rather than assumed equal.

### Product-facing adapter

The canonical research output preserves QA-SRL role questions. An optional downstream adapter can derive simpler product-facing fields such as `who`, `what`, `when`, and `where` while retaining source offsets and confidence. The adapter receives separate evaluation because this mapping can lose or distort semantic information; its output is not treated as interchangeable with the research annotation.

## Experiment plan

### E0: Software and rule baseline

- Preserve `rule-based-v1` as a deterministic lower bound and interface test.
- After E1 supplies compatible evaluation records, measure predicate detection, unlabeled answer overlap, exact spans, offset validity, latency, and error categories.
- Do not report labeled QA-SRL F1 for its surface roles.
- Treat its score as a completeness heuristic, not a probability.

### E1: Annotation, scorer, and challenge-set layer

- Implement versioned QA-SRL and QANom adapters into the evidence-preserving research schema.
- Preserve official splits, predicate types, verbal inflections, related verbal forms, raw question slots, grammatical fields, question and answer provenance, grouped answers, alternative judgments, and document identifiers.
- Reproduce reference-compatible metrics with tested fixtures.
- Add a corrected end-to-end scorer over the union of gold and predicted predicates.
- Define the project’s primary labeled scorer and exact predicate-matching rule.
- Construct, annotate, adjudicate, and freeze the operational-style challenge set before model selection uses it.
- Add document, sentence, predicate-family, and derived-example leakage checks.

The adapters, manifests, canonical readers, full-release validation, fixed training quarantine, named consolidation, evaluation bundles, and three scorer contracts are implemented. The operational-style challenge set remains to be authored, independently annotated, adjudicated, and frozen, so E1 remains incomplete.

### E2: QASem generative reproduction

- Reproduce a T5-small predicate-level parser on the verified QA-SRL and QANom preparation.
- Match published input/output conventions closely enough to establish a credible baseline.
- Record generated-output validity, source-alignment failures, quality, compute, and latency.
- Preserve a published-style run separately from the fixed-training-FLOP comparison with the structured encoder.
- Run at least three seeds from versioned configurations.

### E3: Structured verbal parser

- Train a predicate-conditioned BERT-family encoder for verbal QA-SRL.
- First reproduce answer-span detection with a supplied predicate.
- Add the seven-slot question head and deterministic question realization.
- Compare a frozen encoder, full fine-tuning, and the rule/software lower bound where their metrics are compatible.
- Verify grouped answers, WordPiece alignment, truncation, and exact source reconstruction.

### E3.5: Verbal raw-text vertical slice

- Connect a transparent, high-recall verbal candidate layer to the trained E3 parser.
- Emit versioned, evidence-linked `ActionFrame` records through the existing CLI or a reproducible batch path.
- Report candidate recall, predicate precision/recall/F1, supplied-predicate extraction, raw-text end-to-end extraction, and p50/p95 inference latency separately.
- Keep the claim bounded to verbal predicates; nominal eventivity and the full joint pipeline remain E5 and E6 work.

### E4: Predicate-conditioning study

Hold the encoder, data, optimizer, training tokens, model-selection rule, and evaluation constant while comparing:

1. no predicate signal;
2. a BERT token-type predicate indicator;
3. explicit predicate boundary markers;
4. learned predicate-feature embeddings.

Repeat the marker and learned-feature finalists on a matched RoBERTa-family encoder. Prior QASem marker experiments are treated as related evidence; the token-type, learned-feature, structured-decoder, and portability comparisons are the extension.

### E5: Separate and joint verbal/nominal training

Compare:

1. verbal-only training;
2. nominal-only training;
3. joint training at the natural data ratio;
4. balanced joint training.

The primary comparison fixes optimizer steps or training tokens, reports the exact sampling ratio, gives single-task systems equivalent compute, and uses paired seeds. Verbal and nominal metrics remain separate. Cross-task evaluation is labeled **zero-shot transfer**, not ordinary in-domain performance.

The declared verbal non-inferiority margin is one labeled-F1 point. Joint training is not called beneficial if a nominal gain hides a larger verbal regression.

### E6: Complete predicate-to-argument pipeline

- Extend the E3.5 verbal candidate layer into the full verbal and nominal raw-text candidate pipeline.
- Train or adapt verbal predicate and nominal eventivity classifiers.
- Evaluate supplied-candidate classification separately from candidate generation.
- Compare supplied-predicate extraction with the complete pipeline.
- Attribute missed predicates, spurious predicates, duplicate frames, and lost downstream arguments to their originating stage.

### E7: Generalization, calibration, and efficiency

- Evaluate naturally unseen predicate families in the official test sets.
- Run a controlled family-holdout experiment that groups verbal inflections and related nominal forms, holds out only families with sufficient examples, removes all members from training, and freezes the split before tuning.
- Run domain transfer with training-set size matched to the smallest compared source domain.
- Evaluate the frozen operational-style challenge set.
- Calibrate predicate and argument scores on development data only.
- Report CPU and GPU latency, throughput, peak memory, checkpoint size, preprocessing time, and decoding failures.

## Metrics

### Primary extraction metric

The primary research metric is labeled QA-pair F1 using maximum one-to-one matching, answer-token intersection-over-union of at least `0.5`, and a frozen question-equivalence contract. The scorer name, version, threshold, matching algorithm, predicate source, and equivalence rule accompany every result. Candidate, predicate, and raw-text complete-pipeline metrics remain separate so this component metric is not presented as user-facing system quality.

`qasrl-gs-compatible-v1` and `qanom-reference-v1` are also reported for comparison with prior work, even when their thresholds or treatment of missing predicates differ from the project scorer. `valid-judgment-union-v1` is the named gold-consolidation rule, and every score records it together with predicate source, reference revision, predicate identity, threshold, matching algorithm, and question-equivalence rule.

### Predicate pipeline

- candidate-generator recall;
- supplied-candidate predicate/eventivity precision, recall, F1, and AUROC when appropriate;
- raw-text predicate precision, recall, and F1;
- verbal and nominal results separately;
- unseen-family and held-out-domain breakdowns.

### Argument and question extraction

- labeled and unlabeled QA-pair precision, recall, and F1;
- exact answer-span precision, recall, and F1;
- seven-slot question accuracy and realized-question validity;
- grouped-answer accuracy;
- supplied-predicate and detected-predicate results;
- exact source-offset validity, duplicate rate, and malformed-output rate.

### Calibration and confidence–coverage behavior

- predicate ECE, Brier score, and negative log-likelihood, where correctness means an exactly matched eligible predicate;
- argument ECE and Brier score, where correctness means a matched answer and equivalent role question;
- risk–coverage curves using development-calibrated probabilities;
- calibration reported separately by confidence target rather than from one ambiguous frame score.

### Systems measurements

- p50 and p95 latency at batch one and a declared throughput batch;
- examples or predicates per second;
- peak CPU/GPU memory;
- checkpoint size and parameter count;
- preprocessing and source-alignment overhead;
- mean and sample standard deviation across at least three paired seeds.

Token accuracy is diagnostic only because non-argument tokens dominate the sequence. Dataset test sets remain unavailable for iterative model selection.

## Evaluation slices and split construction

The standard analysis includes:

- verbal versus nominal predicates;
- frequent versus naturally unseen predicate families;
- controlled held-out families, grouping inflections and verbal–nominal relatives;
- source domain with size-matched training comparisons;
- sentence length and predicate–argument distance;
- passive voice, coordination, negation, and modality;
- explicit versus implicit arguments;
- subword fragmentation and truncation;
- candidate-generation, predicate-classification, and argument-model error attribution;
- exact-offset, duplicate, and malformed-generation failures.

The controlled lemma-family split records its minimum family frequency, selection seed, complete family list, and fingerprint. No inflection or related nominal form from a held-out family may remain in training.

## Reproducibility requirements

Every result bundle records:

- Git commit and clean-worktree status;
- dataset release, split, checksums, and preprocessing fingerprint;
- model and tokenizer revisions;
- resolved configuration, seed, and training-token or optimizer-step allowance;
- sampling ratios and model-selection rule;
- hardware and software profile;
- training history and checkpoint identity;
- evaluation code revision and complete per-slice metrics.

Aggregation rejects incomplete seed sets, mixed data revisions, unequal primary-compute allowances, incompatible scorer settings, and result bundles without required source and experiment identity.

## Current foundation definition of done

- The task boundary distinguishes semantic predicates from action items and status inference.
- The public inference schema represents role-neutral verbal and nominal actions with exact source spans.
- The research annotation schema preserves verified supervision without flattening alternatives or judgments.
- The rule baseline, schemas, configuration loader, and CLI have automated regression tests.
- Documentation distinguishes prior results, current software, planned experiments, and measured conclusions.
- QA-SRL and QANom cards record the selected releases, computed statistics, and scorer limitations.
- Restricted course data, outputs, and checkpoints remain absent.
- CI tests every supported Python version.

## Research-project definition of done

- Both adapters reproduce verified statistics and official splits.
- Reference-compatible and corrected scorers pass fixture and regression tests.
- The QASem reproduction and structured encoder train from versioned configurations across at least three seeds.
- The structured model predicts answer spans and all seven question slots.
- A bounded verbal raw-text vertical slice emits evidence-linked records and reports stage-specific errors and latency before the broader conditioning and joint-training studies conclude.
- Conditioning and separate-versus-joint studies use matched data, compute, and paired seeds.
- Candidate generation, predicate classification, supplied-predicate extraction, and complete-pipeline results remain separate.
- Results include exact grounding, calibration, generalization, efficiency, error analysis, limitations, and negative findings.
- Any released checkpoint has verified redistribution rights, a model card, checksums, and an inference example.
- The CLI and Python API can select the rule or released neural backend through the versioned public schema.

## Non-goals

- Publishing the predecessor notebook, restricted OntoNotes-derived course files, or a checkpoint trained from them.
- Pretraining a foundation model from random initialization.
- Re-claiming unified QA-SRL/QANom parsing, joint learning, or predicate markers as first demonstrations.
- Claiming that semantic actions are assignments, commitments, action items, or completed work.
- Inferring implicit participants, resolving cross-document identities, or autonomously executing workflows in the core study.
- Claiming production readiness from research corpora, operational-style examples, synthetic data, or unit tests.
