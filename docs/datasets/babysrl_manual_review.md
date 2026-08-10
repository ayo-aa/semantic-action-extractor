# BabySRL private manual-review workflow

Status: **implemented and synthetically tested; not run on BabySRL**

The BabySRL structural audit establishes aggregate conversion coverage, but a
manual check must verify that selected raw CHAT bracket cells were converted to
the intended BIO tags. Looking only at prepared words and labels would test
linguistic plausibility, not conversion fidelity. This workflow therefore
places the exact selected raw `%srl` fields and prepared tags side by side in a
private, ignored review package.

No CourseWorks or Columbia authentication is part of this process.

## Authorization boundary

The tool fails before reading the raw archive or prepared dataset unless the
operator explicitly confirms all of the following:

- TalkBank registration and sign-in are complete;
- the current [CHILDES access conditions](https://talkbank.org/childes/access.html)
  have been reviewed; and
- the current [TalkBank ground rules](https://talkbank.org/0share/rules.html)
  have been accepted, with the acceptance date supplied as `YYYY-MM-DD`.

The review package records those booleans, the acceptance date, and the two
official URLs. It records no username, password, email address, or other
credential. Command-line flags are an explicit attestation; they must not be
used until the user has actually completed those steps.

## Inputs and identity check

Package creation requires both:

- the pinned `data/raw/BabySRL.zip` archive; and
- the canonical prepared dataset under an ignored child of `data/processed/`.

Before selecting a sample, the tool revalidates the official archive size and
SHA-256 through the BabySRL adapter, reproduces the complete final conversion,
and requires every prepared example to match that raw-derived conversion
exactly. IDs, split, words, predicate index, BIO tags, and roleset metadata must
all match. A missing, extra, or altered prepared record aborts the review.

The official archive pin is:

| Field | Value |
| --- | --- |
| Size | 3,151,030 bytes |
| SHA-256 | `a2d8d38b0818910d05154cb62adcb05ef36b00f0aeae2690b7e1677fd5154604` |

## Deterministic sampling

Examples are stratified by three non-text dimensions:

1. dataset split: train, development, or test;
2. Brown child collection: Adam, Eve, or Sarah; and
3. annotation shape: continued relation, discontinuous argument, multiword
   span, or single-token spans.

Within each observed stratum, the tool ranks examples using SHA-256 over the
versioned selection policy, prepared-data fingerprint, stratum, and example ID.
It takes the first `per_stratum` examples, with a permitted range of 1 to 20 and
a default of 2. Repeated runs over the same prepared fingerprint produce the
same review sample and sample fingerprint.

The opaque review ID is a SHA-256 value derived from the prepared-data
fingerprint and source example ID. The separate decision form uses only that
opaque ID; it contains no corpus words, labels, roleset, or source path.

## Private package contents

All files are written with private permissions under a new child of the
explicitly ignored `data/review/` directory:

| File | Contains corpus material | Purpose |
| --- | --- | --- |
| `review_items.jsonl` | **Yes** | Raw fields beside each prepared tag |
| `review_decisions.jsonl` | No | Opaque review IDs, structured decision, and reason codes |
| `review_manifest.json` | No | Archive/data/sample fingerprints, strata counts, policy, and access attestation |
| `aggregate_decision.json` | No | Final aggregate counts and pass/hold/fail disposition |

The tool never prints review items, words, raw cells, prepared tags, rolesets,
or source IDs to stdout or logs. Successful package creation prints only a
corpus-free receipt. Finalization prints only the aggregate decision record.

Path validation is fail-closed:

- raw data must be the ignored `data/raw/BabySRL.zip` file;
- prepared data must be in an ignored child of `data/processed/`;
- review output must be a new child of `data/review/`;
- Git must confirm every data-bearing path is ignored and contains no tracked
  target; and
- resolved paths must remain inside their allowed directory, preventing a
  symlink escape.

The workflow does not write to an arbitrary external path or tracked report
directory. Publishing an aggregate later is a separate, deliberate action.

## Create the package

Run only after the access attestations are true:

```bash
semantic-action-review-babysrl --repository-root . create \
  data/raw/BabySRL.zip \
  data/processed/babysrl \
  data/review/babysrl-manual-v1 \
  --per-stratum 2 \
  --confirm-talkbank-registration \
  --confirm-talkbank-ground-rules \
  --rules-accepted-on YYYY-MM-DD
```

The destination must not already exist. Package creation is atomic: an
incomplete temporary directory is removed if validation or writing fails.

## Perform the review

Open `review_items.jsonl` only on the authorized local or approved Columbia
device. For each item, compare the raw predicate marker and bracket cell with
the prepared predicate index and BIO tag on the same row. Then edit the
corresponding line in `review_decisions.jsonl`.

Allowed decisions are:

- `"approve"`, with an empty `reason_codes` list;
- `"reject"`, with at least one structured reason code;
- `"uncertain"`, with at least one structured reason code; or
- `null`, with no reasons, for a pending item.

Allowed reason codes are:

- `predicate_head_error`;
- `argument_boundary_error`;
- `role_label_error`;
- `bio_encoding_error`;
- `source_or_display_uncertain`; and
- `other_structured_issue`.

Free-text notes are intentionally unsupported because they could repeat corpus
content into a later aggregate artifact. If investigation requires notes, keep
them outside the repository under the same approved storage controls.

## Finalize the aggregate decision

Use the same acceptance date recorded during creation:

```bash
semantic-action-review-babysrl --repository-root . finalize \
  data/review/babysrl-manual-v1 \
  --confirm-talkbank-registration \
  --confirm-talkbank-ground-rules \
  --rules-accepted-on YYYY-MM-DD
```

Finalization verifies the item-file digest, sample fingerprint, manifest
schemas, exact decision-ID order, structured reason codes, and matching access
attestation. It then writes `aggregate_decision.json` without item IDs or
corpus content.

The fixed approval rule is deliberately conservative:

| Condition | Aggregate status |
| --- | --- |
| Every sampled item is approved | `pass` |
| At least one item is rejected | `fail` |
| No rejection, but at least one item is uncertain or pending | `hold` |

A `pass` is necessary before the BabySRL manual gate can be cleared. A `hold`
or `fail` cannot authorize training. Changing the sample size, strata,
selection policy, or acceptance rule after reviewing outcomes creates a new
review protocol rather than modifying this one.

## Synthetic verification

The automated tests build wholly invented CHAT archives and prepared datasets.
They verify authorization ordering, archive/prepared identity, deterministic
stratification, exact raw-versus-prepared display, Git path protections,
private permissions, tamper detection, structured decisions, aggregate-only
output, and the conservative approval rule. They contain no BabySRL excerpt or
transformed corpus example.

```bash
python -m unittest tests.test_srl_manual_review -v
```
