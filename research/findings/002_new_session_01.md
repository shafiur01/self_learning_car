# Experiment 002: first new-session evaluation

> Git-tracked records: [configuration, results and provenance](../results/002_new_session/README.md). Original run/data links below require the local private archives.

Date: 2026-10-07 (America/Chicago). This is an unfavorable but useful
single-user development finding. It is not a final benchmark or robot trial.

**Later status:** at the user's request, the entire 90-clip session has been
copied into training for a separate eight-command v2. Original files and these
pre-retraining measurements remain intact. It is no longer held-out evaluation
for v2 weights trained on it. See `../COMMANDS_V2_WORKFLOW.md`; no v2 recognition
result has been produced yet.

## What was frozen and what changed?

Training study: `20261008T005037258990Z`. Evaluate all 20 saved models:
30/60/90/201 original training labels, five seeds 11/22/33/44/55 each.
No retraining, new checkpoint selection, ensemble, or gate tuning.

The user recorded **90 new clips**, exactly 15 per label, in validation session
`20261008T005936-27dde0e5`. All original development recordings match their
frozen hashes. New clips have distinct raw contents and no exact development
duplicates; none cross training sessions. All 90 pass existing tracking checks.
Tracking quality does not establish correct gesture meanings or human labels.

The current inventory is 334 training clips in three sessions and 162 validation
clips in two sessions. Keep the old 72 and new 90 validation clips separately
identifiable. Test payloads were excluded from this evaluation.

## Results

All means are over five models evaluated on the **same 90 clips**. Run SD is
model/subset variability, not an interval across independent people/sessions.

| Training labels | Old-session accuracy | New-session accuracy ± run SD | New macro F1 | Macro F1 difference from 201 labels |
| ---: | ---: | ---: | ---: | ---: |
| 30 (5/class) | 78.06% | 58.00% ± 4.87 pp | 0.5558 | −0.1383 |
| 60 (10/class) | 85.28% | 60.44% ± 3.00 pp | 0.5840 | −0.1102 |
| 90 (15/class) | 90.56% | 67.56% ± 2.14 pp | 0.6657 | −0.0285 |
| 201 (all original) | 93.33% | 70.67% ± 2.30 pp | 0.6942 | Reference |

The all-original reference drops **22.67 percentage points** between these two
sessions. The validation sets have different class counts and recording
conditions, so this is a descriptive comparison, not a controlled causal effect.
Every condition has a substantial drop. Fifteen/class remains behind the
all-original reference on average; similarity of the gap across sessions is not
proof of equivalence or actual annotation savings.

| Seed | 30 labels | 60 labels | 90 labels | 201 labels |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 53.33% | 55.56% | 67.78% | 67.78% |
| 22 | 64.44% | 60.00% | 66.67% | 71.11% |
| 33 | 61.11% | 63.33% | 71.11% | 72.22% |
| 44 | 53.33% | 62.22% | 65.56% | 73.33% |
| 55 | 57.78% | 61.11% | 66.67% | 68.89% |

## Where the models fail

Recall is the fraction of recordings with a given true label recognized as that
label. These are means over five models, each with 15 new clips/class.

| Training labels | FORWARD | BACKWARD | LEFT | RIGHT | STOP | NO_COMMAND |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 32.00% | 80.00% | 84.00% | 86.67% | 46.67% | 18.67% |
| 60 | 29.33% | 81.33% | 86.67% | 86.67% | 58.67% | 20.00% |
| 90 | 52.00% | 80.00% | 86.67% | 89.33% | 68.00% | 29.33% |
| 201 | 52.00% | 81.33% | 90.67% | 93.33% | 82.67% | 24.00% |

All-original FORWARD errors go to BACKWARD: mean 7.2 of 15 clips per model.
NO_COMMAND often becomes BACKWARD, with mean 6.8 such errors per model.
Different gesture styles, contextual intent ambiguity, recording changes, or
limited training diversity are possible explanations, not established causes.
Stored `.npz` files contain landmarks rather than RGB video; passing numerical
checks cannot retrospectively verify what instruction the person intended.
The user has been asked to describe how these three labels were performed.

**User clarification received after evaluation:** angles varied slightly and
FORWARD/BACKWARD used creative styles somewhat different from training. This
is consistent with a change in input styles, not proof that angles alone caused
the errors. The user also noted fewer NO_COMMAND/STOP training examples.
Original counts are 16/29 for those classes, while FORWARD has 61; however,
the small-budget conditions use equal class counts and still fail to transfer.
No labels or predictions changed. See `../SESSION_DIVERSITY_PLAN.md` for the
next matched-count training-mixture comparison.

| Training labels | NO_COMMAND → motion mean | STOP → motion mean count / 15 | STOP recall | Mean NLL |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 80.00% | 5.0 | 46.67% | 2.6728 |
| 60 | 80.00% | 4.6 | 58.67% | 3.2300 |
| 90 | 70.67% | 4.2 | 68.00% | 2.7253 |
| 201 | 76.00% | 2.4 | 82.67% | 2.4797 |

All-original NO_COMMAND-to-motion counts by seed: 11/12/10/12/12 out of 15.
STOP-to-motion counts: 4/2/2/2/2 out of 15. A STOP prediction of NO_COMMAND also
lowers STOP recall, but is distinct from a motion classification.

**Exploratory confidence inspection after the primary evaluation:** among
all-original STOP/NO_COMMAND-to-motion predictions, 50 model–clip predictions
score ≥0.8 and 41 score ≥0.95; the maximum softmax score rounds to 1.0. These
are repeated predictions across five models on the same clips, not 50 or 41
independent events. Neither a score cutoff nor this clipped-recording evaluation
reproduces the live margin/three-window gate. No physical motion or continuous
false-command rate was measured. Nevertheless, confidently wrong raw predictions
are a reason to audit pseudo-labels rather than assume confidence establishes
correctness. No confidence threshold was selected from these results.

## Interpretation and next decisions

The current supervised baseline does not transfer reliably to this new sitting,
even for the same user. It cannot serve as evidence of broadly reliable robot
commands. The earlier high validation score alone was insufficient to show
cross-session generalization. This experiment does not establish which change
will fix the problem, nor whether semi-supervised learning helps.

1. Clarify FORWARD/BACKWARD/NO_COMMAND meanings and gesture styles. Preserve
   existing recordings and labels; do not relabel examples just to fit predictions.
2. Retain this entire session as development validation. Use it to diagnose
   candidate changes, acknowledging that it is now inspected. Do not add its
   clips to training and keep claiming evaluation on the same session is held out.
3. Collect the remaining independent validation sittings after clarifying meanings.
   Further training diversity must come from separate, explicitly training sessions.
4. Freeze a matched-budget supervised/SSL comparison before fitting new models.
   Treat automatic pseudo-labels as uncertain hypotheses; existing high-confidence
   errors make unreviewed admission a concern. Controller/audit experiments remain
   proposed, not implemented.
5. Reserve genuinely new people/sessions for final evaluation after choices freeze.

## Files, reproducibility, and verification

Evaluation directory:
[`20261008T013739665513Z`](../experiments/runs/20261008T005037258990Z/session_evaluations/20261008T013739665513Z/).

- [report](../experiments/runs/20261008T005037258990Z/session_evaluations/20261008T013739665513Z/report.md),
  [summary](../experiments/runs/20261008T005037258990Z/session_evaluations/20261008T013739665513Z/summary.json),
  [all trials](../experiments/runs/20261008T005037258990Z/session_evaluations/20261008T013739665513Z/trials.json).
- `evaluation_plan.json` fixes all candidates and checkpoint hashes before
  inference. `manifest.json` fixes input identities/hashes/session provenance.
- `new_validation_features.npz` stores exactly the new evaluation arrays;
  `source/` and `environment.json` preserve executable sources and input hashes.
- Each condition/seed directory retains its metrics, session confusion matrix,
  and all 90 predictions with class scores. Original model weights and training
  reports are preserved; no application checkpoint was activated.

Five new integrity checks passed before real evaluation: changed/missing old
recordings, session/source/split overlap, raw duplication, incomplete/misaligned
cohorts, and paired aggregation. Runtime verification reproduced the original
72-clip score vectors from all 20 checkpoints, checked checkpoint provenance,
and confirmed all old/new development files and checkpoints were unchanged.
These verify evaluation software and preservation, not recognition reliability.

To run the same frozen candidates against the current new-session cohort:

```bash
python -m unittest research.experiments.test_evaluate_sessions -v
python -m research.experiments.evaluate_sessions
```

Each invocation saves a new evaluation rather than overwriting old results.
Adding later validation sessions changes the evaluation cohort; their membership
and per-session results are explicit in its new manifest. Adding training files
causes this strict comparison to stop and requires an explicit new protocol.
