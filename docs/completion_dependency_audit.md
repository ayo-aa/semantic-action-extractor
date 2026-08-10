# Completion dependency audit

Status: **private EWT preparation and actual-data MPS preflight complete;
six-run research study awaits reliable power**

Audit date: 2026-08-10

## Completion boundary

The required finish is the bounded supplied-predicate span-SRL study already
defined by the project:

1. prepare a source-grounded public-data experiment without changing the
   PropBank word-span target;
2. train the predicate-signal model and its matched no-signal ablation for
   three fixed seeds;
3. report development, test, per-role, error, overlength, repair, runtime,
   memory, and artifact-size aggregates;
4. keep raw-text predicate discovery separate from supplied-predicate scores;
5. publish verified code, configurations, aggregate evidence, citations, and a
   tagged repository release.

A hosted neural service, public research checkpoint, portfolio website edit,
and resume edit are separate release choices. They are not required to finish
the controlled research project.

## Selected data route

The selected route joins two directly downloadable official repositories:

- the `.gold_skel` span columns in the
  [official PropBank release](https://github.com/propbank/propbank-release),
  pinned at commit
  `4abade0b53ce4a181e1d98b3518101c1a44d395a`; and
- surface words and official document splits from
  [UD English EWT](https://github.com/UniversalDependencies/UD_English-EWT),
  pinned to release `r2.2` at commit
  `6e064999a75b9c941c515ce1be98352e6f9831e0`.

Neither download requires an account, registration, API key, paid license,
CourseWorks, or a Columbia sign-in. The join is a project inference rather than
the LDC-based reconstruction prescribed by the PropBank README, so it is gated
by exact document, split, sentence, token-width, bracket, and leakage checks.
The official
[Universal Proposition Bank EWT notes](https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT)
independently describe the same source alignment and identify 43 sentences
without PropBank data and exactly one sentence with a token-count mismatch.

The independent read-only audit found:

| Quantity | Result |
| --- | ---: |
| Shared annotated documents | 1,145 |
| PropBank skeleton sentences | 16,579 |
| Structurally valid verbal predicate columns | 38,639 (0 classification failures) |
| Token-width mismatch exclusions | 4 examples in 1 train sentence |
| Aligned verbal examples | 38,635 |
| Cross-split exact-text exclusions | 44 train / 19 development / 13 test |
| Conflicting identical-input exclusions | 29 train / 2 development / 0 test |
| Repeated evaluation-semantic exclusions | 0 train / 10 development / 32 test |
| Maximum-length-128 exclusions | 62 train / 0 development / 0 test |
| Expected modeled examples | 31,039 train / 3,775 development / 3,610 test |

The tracked production adapter reproduces these counts. The ignored private
preparation is complete at fingerprint
`2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b`.
Its aggregate-only split and provenance hashes are recorded in the
[preparation report](../reports/ewt_preparation.md).

The [aggregate-only private source review](../reports/ewt_private_source_review.md)
also inspected all 13 metadata-versus-primary predicate-anchor divergences,
the one token-width mismatch, and 30/30 deterministically selected aligned
verbal records. It retains no corpus text or identifiers. This is supporting
evidence for the inferred join, not proof that it matches hidden LDC words.

## Completed preparation and fit preflight

Adapter implementation commit
`9b7c94ec9be4a9b56c3cd7df3cb9a83b34b87f42` wrote 31,101 train, 3,775
development, and 3,610 test examples to ignored private storage. The 111-label
prepared dataset has fingerprint
`2eb2f0e20bfa5e3521faba9521b329a0c43dcc63eb523a359e79337c04b66e1b`;
31,039/3,775/3,610 examples remain after the fixed length-128 policy.

The paired configuration digests are
`9bf8cd7c7a839bd9bfb6b39fde616f7e6f42d2ef163ea5f47b7afeeee1120bdc`
for predicate signal and
`19ff94ff8bf839ee2fd5ebdab8ffed24b1ad7b243cc413a2ba687262f8f9d866`
for no predicate signal. The predicate variant passed one actual-data MPS
batch at size 32 and longest retained sequence 118, completing forward,
backward, gradient clipping, AdamW, and scheduler step in 3.0069 seconds with
3,211,741,952 allocated bytes. This is fit evidence, not six-run training,
evaluation, or a real benchmark.

## Rights and publication boundary

The PropBank annotation repository is CC BY-SA 4.0. UD English EWT applies CC
BY-SA 4.0 to its annotations and database rights, while its README separately
notes that the underlying texts come from several sources and may retain their
original copyrights. The conservative project boundary is therefore:

- acquire the public repositories directly and pin their commits;
- keep raw and prepared text outside Git;
- publish acquisition/conversion code, hashes, exclusions, citations, and
  non-reconstructive aggregate results;
- keep trained research checkpoints private and ignored unless a later,
  explicit redistribution review clears them; and
- do not present the repository's MIT code license as a license for data or
  learned artifacts.

This boundary completes the portfolio study without claiming that the public
repository redistributes either source corpus or a corpus-derived model.

## Full dependency matrix

| Area | What completion needs | Current state | User contribution |
| --- | --- | --- | --- |
| Dataset account | Direct official GitHub downloads | No registration or login required | None |
| Dataset payment | PropBank and UD repository access | Free | None |
| Course materials | Independent implementation and public data only | CourseWorks and Columbia course data are excluded | None |
| Cloud notebook | Local Apple MPS execution | Colab is unnecessary | None |
| Columbia compute/storage | Local ignored storage is sufficient | Not required | None |
| Manual source review | Automated join plus bounded public-source checks | Completed: all 13 predicate-anchor divergences, the token-width mismatch, and 30/30 deterministic aligned records; aggregate-only report tracked | None |
| Python environment | Python 3.12, PyTorch, Transformers, test tooling | Installed and rehearsed | None |
| Base model and tokenizer | `google-bert/bert-base-uncased` at exact revision `86b5e0934494bd15c9632b12f734a8a67f723594` | Cached locally; public Apache-2.0 model page | No Hugging Face account/token |
| Data acquisition | Two pinned repository snapshots | Both source checkouts are fetched locally and their revisions are verified | None |
| Data preparation | Adapter, audit, canonical splits, fingerprint | Complete in ignored `data/processed/ewt-span-srl-v1`: 31,101/3,775/3,610 examples; fingerprint `2eb2f0e2…b66e1b` | None |
| Experiment configs | Two matched variants, seeds 13/17/23, two epochs | Frozen in `configs/ewt_predicate_signal.toml` and `configs/ewt_no_predicate_signal.toml` with recorded digests | None |
| Compute | Six fine-tuning runs on MPS | Actual prepared-data predicate preflight passed one full batch-32 optimizer step; latest power checks still show discharge while AC is attached | Before the unattended run, connect a charger that actually delivers power and keep the lid open |
| Runtime | About 11,640 optimizer steps before any interruption | One longest-retained-sequence batch completed in 3.0069 seconds; conservative full-run window remains 6–12 hours | No active supervision |
| Disk | Raw data, prepared splits, and approximately 2.4–3.1 GiB of checkpoints | Point-in-time check on 2026-08-10: about 9.7 GiB free after the reproducible 2.4 GiB synthetic checkpoint output was removed; tight but workable if monitored | None unless more local files are added meanwhile |
| Failure recovery | Atomic result/checkpoint boundaries | Independently reviewed and verified: exact-identity resume, completed pair validation, canonical complete journal, bounded residue recovery, unknown-artifact rejection, and per-output nonblocking lock | None |
| Evaluation | Exact span, role, token, predicate, repair, and overlength metrics | Implemented; real results pending | None |
| Error analysis | Aggregate automatic taxonomy; no corpus text published | Must be generated after training | None for the required bounded analysis |
| Systems evidence | Latency, throughput, memory, and checkpoint size | Real benchmark path rehearsed | None |
| Research outcome | Honest paired result, including a null or negative ablation | No minimum accuracy or positive-result requirement | None |
| Public checkpoint | Not needed for the bounded study | Default is private/no release | User input only if public weights later become a requirement |
| Live neural deployment | Not needed for the bounded study | Out of current release scope | User input and a rights/hosting decision only if later requested |
| GitHub | Push, CI, PR merge, and tag | Existing branch/PR and authenticated paths available | Browser reauthentication only if GitHub rejects the saved session |
| Portfolio/resume | Repository release is sufficient for project completion | No canonical website; multiple resume files exist | Name a target only if a separate resume/site edit is desired |

## Verified interruption recovery

The training recovery boundary is complete. If an execution is interrupted,
Codex reruns the same `semantic-action-train-srl` command with only `--resume`
added. The partial run must match the same output, provenance, paired configs,
prepared dataset and fingerprint, exact Git revision, and runtime identity.

Before any completed work is reused, the command validates every result with
its checkpoint and validates completed predicate/no-predicate seed pairs. It
retains a canonical journal even after successful publication, recovers only
exact writer-owned interrupted atomic-write residue, next-checkpoint staging,
and checkpoint-tombstone cleanup, and rejects lookalike or unknown artifacts.
A per-output nonblocking lock prevents concurrent writers. This durability was
verified with injected failures. EWT preparation and a separate one-step
real-data preflight are now complete, but no paired research run or result was
produced.

## Exact user checkpoints

Only one physical checkpoint is expected for the required path: before the
final unattended training window, make sure a working charger actually delivers
power and leave the lid open with no restart scheduled. The latest check reports
AC attached while the battery is nevertheless discharging, so the power
delivery issue must be corrected before training. The app may also show routine
one-click approvals for MPS execution or GitHub authentication.

There is no TalkBank registration, terms attestation, private BabySRL review,
CourseWorks login, Columbia account action, Colab session, Hugging Face token,
paid service, hand-authored configuration, manual metric calculation, or
manual Git work on the selected path.

If public model weights or a hosted neural demo are later added to the goal,
that creates a new checkpoint: explicit rights review plus a hosting decision.
The default completion plan publishes code and aggregate evidence while keeping
the checkpoints private.
