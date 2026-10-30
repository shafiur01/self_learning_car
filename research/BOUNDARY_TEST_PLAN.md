# Experiment 005 — fresh, frozen single-user clip evaluation

Internal protocol written before new test collection, 2026-10-08. This is not
a public preregistration or the final unseen-person benchmark for a paper.

## Question and fixed comparison

Does the routine-versus-targeted pattern from experiment 004 carry over to two
new sittings? The inspected monitoring session favored routine examples. Keep
that unfavorable targeted result; a new test cannot erase it.

Freeze all 15 checkpoints from
`experiments/runs/boundary_data/20261008T072040903427Z`: reference, routine and
targeted, seeds 11/22/33/44/55. No additional training. Reference has 472 training
clips; each added condition has 512. Routine versus targeted is label/count and
training-budget matched; reference versus either addition has different data.

Primary: compute macro F1 on the pooled 80 new clips for every model; subtract
routine F1 from targeted F1 within each seed, then average the five differences.
Macro F1 gives each of the eight classes equal weight. Also preserve every
candidate's accuracy, NLL, class precision/recall/F1, confusion matrix,
NO_COMMAND-to-motion and NO_COMMAND-to-DOWN errors, STOP recall/motion errors,
FORWARD/BACKWARD confusions, and each sitting separately. Do not select a seed,
ensemble, gate threshold, subset of favorable clips, or alternate primary after
seeing results. Seeds share clips; seed SD is not independent-person uncertainty.

## New human collection

Same current user, two genuinely new physical sittings, 40 clips each: five per
each of FORWARD/BACKWARD/LEFT/RIGHT/STOP/NO_COMMAND/UP/DOWN. Close the first
recording window and take a break before sitting 2. A new software session ID
alone does not demonstrate a new physical sitting. Resume a paused recorder only
within that same sitting. Keep camera/hand consistent within a sitting; record
the hand with optional `--hand right` or `--hand left`.

The schedule interleaves all eight labels five times; the second sitting reverses
label order. Prompts include comfortable ordinary variants, vertical pointing
and whole-hand motion, natural relaxed/resting/repositioning NO_COMMAND, and one
no-hand NO_COMMAND clip per sitting. These are a specified small style sample,
not an estimate of the frequency of gestures in real deployment. Perform honestly
and naturally; do not deliberately make the test easier or adversarially harder.
Follow the existing meanings; do not redefine DOWN to include ordinary resting.
Only the currently prompted key saves. SPACE freezes/replays the preceding
roughly two seconds; check the actual replay, save if it matches, or x to repeat.
Label without model predictions. No RGB video is written; existing landmark
coordinates/times/visibility and provenance are saved.

`boundary_test.py freeze` verifies the original study and all weights, archives
source and fixes the rules/time/hash. `record` guides collection in a separate
test folder and never loads a model. `status` checks metadata/progress only;
it does not score or inspect landmark payloads. Both sittings must be complete
before a single joint `evaluate` call. Do not evaluate sitting 1 early.

The evaluator checks metadata first, then creates an access ledger before any
test landmark read. Any subsequent failure leaves the pool marked inspected;
it cannot be represented as an untouched retest. Exact raw duplicates of known
development inputs or another test clip fail after access is recorded. Historical
session IDs are banned, including earlier inspected tests. This protects honest
local provenance; saved metadata does not independently prove human intention,
participant identity or physical independence.

Retain/report tracking failures, including legitimate no-hand NO_COMMAND clips;
do not filter these out to raise accuracy. The existing app prevents saving a
poorly tracked movement clip: repeat it and account for the extra collection
effort. Camera attempt intervals include replay/preparation/pauses, not just
labeling; count the 80 evaluation labels separately from training label budgets.
Unsaved repeats and offline time are not fully timed by this recorder.

## Commands in order

The agent prepares and freezes the study before requesting human recording.
From the project terminal:

```sh
.venv/bin/python -m research.experiments.boundary_test record --sitting 1
```

After 40/40, press q. Begin a new sitting after a break:

```sh
.venv/bin/python -m research.experiments.boundary_test record --sitting 2
```

After both are complete the agent checks progress, then evaluates all 15 together.
Do not run training, label corrections, or a prediction demo on this pool first.

## What this establishes and what comes later

This evaluates fresh-session clip classification by fixed supervised models of
one user. It does not test continuous command gating, motors, drones, humanoids,
medical robots or a robot emergency stop. UP/DOWN remain screen-only labels.
It does not establish label savings, a new algorithm, semi-supervised learning,
state of the art or publication readiness.

The larger goal remains cost-measured label-efficient learning for robotics:
strong supervised budget baselines; a separate unlabeled training pool;
standard SSL versus audited SSL at equal human/compute budgets; optional
self-supervised pretraining and stage-order ablations; unseen-participant/public
benchmarks; and measured robot/controller consequences. Novelty and usefulness
must be supported by prior-work review and experiments. Preserve these 80 clips
as inspected evaluation afterward; future algorithm choices need development
data and another genuinely untouched benchmark after those choices freeze.
