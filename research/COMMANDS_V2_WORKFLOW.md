# Eight-command upgrade: collection, retraining, validation, final test

**2026-10-12 clarification:** Intended STOP is an open palm; this older workflow's
closed-fist description was an incorrect assumption. Its archived sources and
results remain historical. For the replacement UP/new STOP contract use
[the v3 workflow](COMMANDS_V3_WORKFLOW.md), pending exact UP and STOP-history
answers. Do not use the old prompts to collect the new meaning.

Updated: 2026-10-08 (America/Chicago). **Training and second-session validation completed.**
This is a supervised system upgrade. It does not implement or validate the
proposed semi-supervised robotics research algorithm.

**Current result:** experiment005 jointly evaluated all15 frozen experiment004
candidates on80 fresh clips. Reference/routine/targeted accuracy79/84.5/83%,
macro F1 .7692/.8320/.8171; expected targeted advantage absent. UP recall remains
32/26/26%; NONE improves34→86/86%. Read [finding005](findings/005_boundary_test.md)
and [Lesson8](LESSON_08.md). These test data are now inspected; no repeated test
invocation or app activation. The [frozen test plan](BOUNDARY_TEST_PLAN.md) records
the completed protocol; original single-condition commands below are historical,
not a second way to evaluate this same pool. Next diagnose UP/STOP on development
recordings before any further collection or fitting.

**Latest progress:** 472 training clips and 88 validation clips in two sittings.
All five models were trained using the first 47-clip validation sitting for
checkpoint selection; mean accuracy **85.11%**, macro F1 **0.8532**. The second
sitting has 41 clips and was evaluated without retraining/reselection: mean
accuracy **86.34%**, macro F1 **0.8577**. See
[first result](findings/003_commands_v2.md),
[second-session result](findings/003_commands_v2_session_02.md), and
[Lesson5](LESSON_05.md). These are historical development results; joint freeze
status is given above. No autoactivation.

All five models confidently predicted DOWN on one naturally resting NO_COMMAND
hand. The user confirms the distinction from deliberately pointing downward;
preserve the label and earlier deliberate whole-hand DOWN forms. The controlled
training comparison is now prepared in [experiment004](BOUNDARY_DATA_PLAN.md).
Next use its guided recorder for sitting1 (40 training clips), then sitting2
(40 more), preserving routine/targeted groups and all collection costs. Keep both
inspected validation sittings outside gradient updates. No extra validation
collection is required merely to repeat the completed second sitting.

Update: [experiment004](findings/004_boundary_data.md) completed both new training
sittings and the matched routine/targeted comparison. Monitoring means88.29%/
85.37% accuracy, macroF1 .8788/.8497; targeted advantage absent in this pilot.
All candidates remain saved and the camera unchanged. The joint freeze is now
complete, and its new test has now completed too. Read [Lesson6](LESSON_06.md) for the controls
and limits of the training comparison.

## Initial preparation inventory

The user authorized using the 90 recently evaluated clips for training and
adding UP/DOWN. Both vertical labels support deliberate pointing and deliberate
whole-hand movement, including varied styles. The existing six meanings remain.
UP/DOWN refer to the vertical directions in the camera preview.

The new dataset lives in `research/experiments/data/commands_v2/`. All original
recordings, six-class weights, and prior results remain intact. The v2 copies
have a new schema/label contract and retain original file hashes, metadata,
session identities, and previous split roles.

| Label | Training copies currently available |
| --- | ---: |
| FORWARD | 114 |
| BACKWARD | 75 |
| LEFT | 68 |
| RIGHT | 59 |
| STOP | 57 |
| NO_COMMAND | 51 |
| UP | 0 |
| DOWN | 0 |
| Total | 424 |

Fresh v2 validation and test currently contain zero clips. The 424 are 334
legacy training examples, including 133 historical corrections, plus the 90
former validation recordings. Their old performance remains a valid historical
measurement of the old frozen models. After learning from them, recognition on
those same clips cannot be reported as held-out generalization.

The old 72 validation clips and 52 historical test clips are not imported into
v2 evaluation. New evaluation needs genuinely new sessions. V2 is an eight-class
task, so its overall accuracy cannot be compared directly with an old six-class
number as evidence of a controlled algorithm improvement.

## Step 1 — record UP and DOWN training examples

From the project folder:

```bash
.venv/bin/python -m research.experiments.commands_v2 record --split train
```

First sitting: **10 UP + 10 DOWN clips**. For each label, record five pointing
examples and five whole-hand-motion examples. Vary reasonable angles while
keeping directional meaning clear. Perform each clip for about two seconds,
SPACE to freeze/review, **u** to save UP, **d** to save DOWN, x to discard, q to
finish. Click the camera window so it receives key presses.

For a broader initial pilot, repeat 10 UP + 10 DOWN in a separate sitting,
giving 20 per new class, 40 additional training labels and 464 training clips
overall. These are diagnostic collection counts, not a statistically justified
publication size or a guarantee of robust recognition.

Optional provenance: add `--hand right` or `--hand left` and
`--notes "describe angles and gesture styles"` to the command. Do not load a
model while collecting evaluation clips. Ordinary reaching/repositioning should
not receive a command label merely because a hand moved vertically; conversely,
indistinguishable intentional/nonintentional motions need observable context
that the current landmark-only classifier does not provide.

## Step 2 — collect fresh validation for all eight labels

Restart the recorder in a separate sitting:

```bash
.venv/bin/python -m research.experiments.commands_v2 record --split validation
```

Initial pilot: two separate sittings of **five/class/sitting**, 40 clips each,
80 fresh validation clips total. Restart the app between sittings. Each sitting
contains FORWARD, BACKWARD, LEFT, RIGHT, STOP, NO_COMMAND, UP, DOWN. Use
representative variation for every label, both vertical styles, and ordinary
visible-hand non-command movement. Keep STOP as the existing held closed fist.

Save keys are **f/b/l/r/s/n/u/d**. Show legitimate styles you expect the interface
to support. Do not make validation easier just to improve the reported score.
Check the inventory at any time:

```bash
.venv/bin/python -m research.experiments.commands_v2 status
```

## Step 3 — train and study validation errors

The initial run and second-session check are complete. Rerunning training creates
another study and would use all current validation clips for checkpoint selection;
they would then no longer be independent of that selection. Fix the next
comparison before fitting instead of rerunning simply to improve a score.

The separate evaluator (already executed) is:

```bash
.venv/bin/python -m research.experiments.evaluate_v2_sessions
```

It writes separate results for the original study's saved candidates. Repeating
it on these same clips does not supply new independent evidence.

After UP/DOWN training and fresh validation are available:

```bash
.venv/bin/python -m research.experiments.commands_v2 train
```

The runner requires at least five train and five fresh validation clips per
class; every validation sitting must include all eight classes. It rejects
legacy validation sessions, corrections used as validation, exact duplicated
development contents, and sessions crossing splits. Existing imports are
checked against their migration hashes.

All five seeds 11/22/33/44/55 start from random weights. The network retains
the existing 67-feature input and temporal layers, with an eight-score final
layer: 48,520 parameters, compared with the old 48,390. Adam lr 0.001,
class-weighted cross entropy, 560 updates, batches of 32 sampled with replacement,
no augmentation; validation at steps 20,...,560 selects the lowest-loss weights
for each seed. Adding labels does not give the classifier knowledge of their
meaning without examples. Old six-class weights are incompatible with this
eight-class loader; old applications still accept their original weights.

Each training run gets a new directory under `research/experiments/runs/commands_v2/`.
Keep every seed's weights, histories, predictions and session metrics. Overall
accuracy, macro F1, per-class recalls, NLL, STOP errors, and NO_COMMAND-to-motion
errors all matter. UP/DOWN are explicitly included among movement classes in
those metrics. Raw clip errors do not measure execution of a robot or the live
three-window gate.

The runner does not read test arrays/labels or open the camera during training.
It does not silently activate any candidate. Review validation, change methods
only through explicit new studies, and retain unfavorable results.

## Step 4 — freeze choices, then collect final test

Once the method/choices are fixed, use the actual run directory printed by
training. `RUN_PATH` below is a placeholder, not a literal folder name:

```bash
.venv/bin/python -m research.experiments.commands_v2 freeze --run RUN_PATH
```

Freeze records all candidate checkpoint hashes, input/source hashes, and the
evaluation policy: every seed, raw class predictions, no selected winning seed,
ensemble, or test-tuned threshold. This is a research declaration, not a model
activation or a guarantee of success.

**Only after freezing**, record genuinely new test sittings:

```bash
.venv/bin/python -m research.experiments.commands_v2 record --split test
```

Initial single-user pilot suggestion: two sittings of five/class (80 clips).
Finish all eight labels in each sitting before evaluation. A publication claim
about unseen people requires separate participants, not only more recordings
from the same person; those splits and sizes remain to be finalized.

```bash
.venv/bin/python -m research.experiments.commands_v2 test --run RUN_PATH
```

The evaluator rejects test recordings collected before freeze, legacy or
development sessions, exact duplicate development/test contents, and previously
inspected test sessions. Access is recorded in the dataset's `test_access.json`.
After payload access, failures also retire that pool; incomplete collection is
rejected at metadata preflight so it can be finished before accessing payloads.
All candidates are evaluated together and results preserved in `final_test/`.
Do not repeatedly tune against those results and call the same recordings fresh
test data. New model decisions then need another genuinely new final pool.

## Robot meanings are a separate mapping

The classifier supplies the perceived intent label UP/DOWN. A drone controller
might map them to altitude changes; a humanoid may map DOWN to sitting. Those
actions are different policies and need controller-specific context and tests.
No controller, drone, humanoid, physical safety evaluation, or motor connection
was implemented. The prototype displays labels only.

## Files and verification

| File | Purpose |
| --- | --- |
| `experiments/commands_v2.py` | Versioned data migration, eight-class model, recording adapter, training, freeze/test commands |
| `experiments/test_commands_v2.py` | Synthetic checks of migration, recording, class mappings and evaluation separation |
| `experiments/evaluate_v2_sessions.py` | Saved-weight inference on fresh validation sessions; preserves original training source/results |
| `experiments/test_evaluate_v2_sessions.py` | Checks fresh-session eligibility, artifact integrity, separate aggregation and no fitting |
| `data/commands_v2/contract.json` | Frozen label meanings and feature/schema compatibility |
| `data/commands_v2/migration.json` | Exact original/derived files, hashes, retired sessions and former roles |
| `data/commands_v2/preparation_status.json` | Whether copying completed or failed |
| Each run's `manifest.json` / feature archive | Exact training and validation inputs |
| Each run's `environment.json` / `source/` | Versions, source/input hashes and actual code used |
| Each run's `seed_N/model.pt` / prediction/history files | Every learned candidate and its development outcomes |
| Each run's `freeze.json` / `final_test/` | Prespecified candidates and their fresh final evaluation |

Nine new checks and 14 existing pipeline checks passed. Camera checks used
controlled fixtures rather than opening the physical camera. Synthetic training
verified eight-class checkpoint reload and saved predictions; it is not a real
recognition result. All 424 actual copies preserve exact landmark/timestamp/
visibility arrays and original hashes, with no exact raw development duplicates.
Initial training preflight correctly refused UP/DOWN counts of zero. After
new collection, all five real v2 checkpoints were trained; their saved prediction
vectors/input hashes/provenance are verified in the study's `verification.json`.
No new dependencies were added.
