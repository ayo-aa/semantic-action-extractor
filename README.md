# Semantic Action Extractor

Turn short operational text into inspectable action records.

**Input**

```text
Maya emailed the signed contract to Jordan on Tuesday.
```

**Output (abridged; see `examples/sample_output.json` for the full schema)**

```json
{
  "actions": [
    {
      "actor": {"text": "Maya", "start": 0, "end": 4},
      "predicate": {"text": "emailed", "start": 5, "end": 12},
      "predicate_lemma": "email",
      "patient": {"text": "the signed contract", "start": 13, "end": 32},
      "qualifiers": [
        {"relation": "to", "value": {"text": "Jordan", "start": 36, "end": 42}},
        {"relation": "on", "value": {"text": "Tuesday", "start": 46, "end": 53}}
      ]
    }
  ]
}
```

The first milestone is useful to product teams turning requests into structured work items, analysts bootstrapping annotation workflows, and ML engineers who need a transparent baseline before training a semantic-role model. It is deliberately small: a deterministic English rule baseline with exact source offsets, a stable JSON schema, a CLI, and tests.

## Run it

Requires Python 3.11 or newer.

```bash
python -m pip install -e .
semantic-action-extractor --pretty "Maya emailed the signed contract to Jordan on Tuesday."
```

It also accepts standard input or a UTF-8 text file:

```bash
echo "The support team escalated the incident to Priya." | semantic-action-extractor --pretty
semantic-action-extractor --input-file examples/sample_input.txt --pretty
```

Python API:

```python
from semantic_action_extractor import RuleBasedExtractor

result = RuleBasedExtractor().extract(
    "The support team escalated the incident to Priya."
)
print(result.to_dict())
```

## What the baseline returns

Each action contains:

- an optional actor;
- a predicate and a conservative lemma;
- an optional patient/object;
- prepositional qualifiers represented without overclaiming their semantic role;
- exact, zero-based, end-exclusive character offsets;
- a heuristic confidence score and extractor version.

The baseline is not a substitute for semantic role labeling. It works best on short, active, declarative English sentences and makes no claim of benchmark quality. See [MODEL_CARD.md](MODEL_CARD.md) for limitations.

## Configure it

Pass a TOML file to add domain verbs or set a confidence threshold:

```bash
semantic-action-extractor \
  --config configs/baseline.toml \
  --pretty \
  "The operator triaged the alert."
```

## Project direction

This repository directly reengineers an earlier BERT semantic-role-labeling prototype into a reusable product and research codebase. The prototype established the task, WordPiece alignment problem, predicate conditioning, fine-tuning path, and span evaluation direction. This milestone creates a clean interface and an executable baseline; subsequent milestones will add an authorized-data neural implementation and controlled experiments.

- [PROJECT_SPEC.md](PROJECT_SPEC.md): research question, hypotheses, experiment plan, and definition of done
- [PROVENANCE.md](PROVENANCE.md): what comes from the prototype and what is newly written
- [DATA_USAGE.md](DATA_USAGE.md): data boundaries and release checklist
- [MODEL_CARD.md](MODEL_CARD.md): current baseline behavior and limitations

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m semantic_action_extractor --pretty \
  "Maya emailed the signed contract to Jordan on Tuesday."
```

## License

New code in this repository is MIT licensed. That license does not grant rights to any dataset, checkpoint, assignment material, or other third-party artifact. None of those artifacts are included here.
