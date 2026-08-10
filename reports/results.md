# Results

No corpus-level extraction result is available in the restored scope.

The repository currently provides software and feasibility evidence only: the dependency-free rule baseline runs; synthetic unit tests exercise the schema, source grounding, PropBank/PTB pointer conversion, predicate-conditioned alignment, BIO handling, model-construction boundary, and generic exact labeled-span scorer; and a read-only aggregate audit rejects MASC for the fixed span milestone. None of this measures model accuracy, validates a real BERT runtime, or constitutes a public-data reproduction.

## Current evidence

| Component | Evidence | Status |
|---|---|---|
| `rule-based-v0` | API, CLI, configuration, and synthetic behavior tests | Implemented; no corpus quality or latency result |
| `bert-srl-token-type` primitives | Synthetic pointer conversion, alignment, BIO, scoring, and injected-model tests | Partially implemented; no approved-corpus adapter, training run, checkpoint, or corpus result |
| MASC feasibility | Read-only rights, archive/join, and aggregate annotation-fit audit | Rejected: G2 hold and G3 optimistic ceiling below 99% |
| Supplied-predicate public evaluation | Dataset, split, scorer, seeds, and hardware | Blocked on replacement-source selection and pipeline implementation |
| Raw-text pipeline evaluation | Candidate detection plus downstream role extraction | Pending; must remain separate from supplied-predicate evaluation |

Future quality tables must identify the exact dataset release and rights status, split manifest, gold-span conversion policy and coverage, scorer, role filters, predicate source, seeds, hardware, configuration, and repository revision.
