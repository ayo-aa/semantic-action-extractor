# Project specification

## Product problem

Teams routinely receive unstructured sentences that imply work: a person sent a document, a service escalated an incident, or a customer requested a refund. The project converts that text into traceable action frames while preserving exact source spans.

The product contract is intentionally narrower than general language understanding:

```text
text -> [actor, predicate, patient, qualifiers, source offsets]
```

The first users are:

- product and operations teams prototyping work-item extraction;
- analysts creating or auditing labeled examples;
- ML engineers comparing a trained model with an inspectable baseline;
- researchers studying predicate-aware semantic role labeling.

## Research question

How much do predicate-aware contextual encoders improve action-frame extraction over transparent rules, and which gains survive unseen predicates, domain shift, and realistic latency constraints?

## Scope

This repository directly refactors and reengineers a prior BERT semantic-role-labeling notebook into a reusable system. The intended progression is:

1. define a stable, source-grounded action schema and runnable rule baseline;
2. port the neural data/model/evaluation concepts into modular, newly reviewed code;
3. train only on authorized data;
4. compare predicate-conditioning strategies under controlled experiments;
5. expose the best justified model through the same API and CLI.

The project does not claim to identify every event, resolve entities, infer intent, or train a foundation model from scratch.

## Hypotheses

- **H1 — contextual model:** A fine-tuned contextual encoder will outperform the rule baseline on labeled span F1, especially when arguments are distant from the predicate.
- **H2 — predicate signal:** Explicit predicate conditioning will outperform an otherwise identical encoder with no predicate signal.
- **H3 — encoding choice:** A dedicated predicate representation or marker-token strategy will be at least as accurate as repurposing BERT token-type embeddings and will transfer to encoders without segment embeddings.
- **H4 — generalization gap:** All learned systems will lose materially more performance on held-out predicates or domains than random in-domain splits reveal.
- **H5 — operational tradeoff:** The deterministic baseline will remain useful when explainability, zero setup, or CPU latency matters more than maximum recall.

## Experiment plan

### E0: rule baseline

- Run `rule-based-v0` on synthetic, redistributable examples.
- Measure schema validity, span-offset correctness, latency, and hand-labeled precision/recall once an authorized evaluation set exists.
- Record error types rather than presenting unit tests as a quality benchmark.

### E1: reengineered neural reproduction

- Implement word-to-subword label alignment, a dataset adapter, predicate-aware token classification, BIO decoding, and span metrics as package modules.
- Compare a majority/O baseline, a frozen encoder, and full fine-tuning.
- Use at least three seeds and report mean, standard deviation, hardware, runtime, and package/model revisions.

### E2: predicate-conditioning ablation

Hold data, splits, optimizer, budget, and encoder constant while comparing:

1. no predicate signal;
2. token-type predicate indicator;
3. predicate boundary marker tokens;
4. learned predicate-feature embeddings.

### E3: robustness and generalization

- Create leak-checked held-out-predicate and, when available, held-out-domain splits.
- Break down span F1 by role, predicate frequency, sentence length, and subword fragmentation.
- Test truncation policy explicitly, including cases where the predicate falls outside a naive fixed window.

### E4: efficiency

- Report batch and single-example p50/p95 latency, throughput, peak memory, model size, and hardware.
- Compare full precision with any justified quantized or exported version.

## Metrics

Primary model-quality metric:

- micro-averaged exact labeled-span precision, recall, and F1 using a documented, tested boundary convention and an official dataset scorer when one is required.

Secondary metrics:

- per-role F1;
- token accuracy, reported only as a diagnostic;
- malformed-BIO rate;
- truncation/drop rate;
- latency, throughput, memory, and artifact size;
- performance mean and spread across seeds.

## First-milestone definition of done

- A new user can install the package and extract an action from text without downloading a model or dataset.
- The CLI accepts positional text, standard input, or a file and emits valid JSON.
- Every returned span maps exactly back to the input text.
- The schema, rule baseline, configuration loader, and CLI have automated tests.
- Documentation clearly separates current behavior from future neural claims.
- Restricted course data, assignment prose, starter code, figures, and checkpoints are absent.
- Provenance, data use, model limitations, and license scope are explicit.
- CI runs the tests on supported Python versions.

## Research-project definition of done

- An authorized dataset can be prepared from documented instructions without entering Git history.
- Training and evaluation are reproducible from versioned configurations.
- All baselines and ablations use fixed splits and comparable budgets.
- Results include multiple seeds, error analysis, compute accounting, and limitations.
- The public API remains compatible across the rule and neural extractors.
- A released checkpoint has a completed model card and a verified redistribution right.

## Non-goals for this milestone

- publishing the original notebook;
- distributing OntoNotes, PropBank-derived course files, or a checkpoint trained from them;
- claiming benchmark performance from the rule baseline;
- resolving pronouns, modality, negation, passive voice, or implicit arguments;
- production deployment or compliance certification.
