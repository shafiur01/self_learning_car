# Lesson 02 — how much labeled information does the model need?

An example has two parts: its hand-motion features (the input), and the human's
command label (the target). Supervised learning changes the model's weights to
make its output agree with those targets. A **label budget** counts the distinct
human-labeled recordings supplied to training, not how often they are repeated.

## What experiment 002 tests

Give the same temporal model 5, 10, or 15 original recordings per gesture:
30, 60, or 90 total training labels. Compare these to all 201 original recordings
in the current snapshot. The original pool has only 16 NO_COMMAND clips, so a
balanced 20-per-class condition is unavailable without new original recordings.

For each seed, the five-example set is part of the ten-example set, which is
part of the fifteen-example set. Larger budgets add examples instead of swapping
the entire dataset. Five seeds change both the sampled subset and the initial
weights; their spread includes both kinds of variation.

The 133 correction recordings stay preserved and excluded from this primary
comparison. They were chosen with earlier models' help. Treating them as freely
available labels would conceal the earlier acquisition and training effort.

## Why count training updates instead of epochs?

An epoch visits a training dataset once. An optimizer update uses a batch of
examples to calculate an error, differentiate it, and change the weights. With
different dataset sizes, equal epochs produce different numbers of updates.

Every trial here receives 560 Adam updates with batches of 32. Sampling with
replacement means a recording can appear more than once in a batch. That is
intentional: even the 30-label condition can have a full batch, and every trial
processes 17,920 example draws. The distinct information remains 30 labels.
Repeated examples can help optimization, but cannot supply missing styles or
new people. Smaller datasets can also be memorized rather than generalized.

The loss is class-weighted cross entropy, as in the existing trainer. The
balanced small subsets have equal class weights; the full original reference
has unequal class counts and compensating weights. No augmentation is applied.

## Why validation stays fixed

Every trial is checked after each 20 updates, at the same 28 points. We keep
the checkpoint with the lowest unweighted validation cross entropy. This gives
every label budget the same selection opportunities.

Validation both selects checkpoints and compares conditions, so these results
are development evidence. The existing validation consists of one single-user
session. It does not establish accuracy on strangers or reliable robot operation.
New sessions and separate final-test people are still needed.

This recipe differs from experiment 001's epoch-based shuffled batches. Its
all-original reference is the controlled comparison here; a score difference
from the old study cannot be attributed to one change.

## Why this is simulated label efficiency

We already collected and labeled the recordings. This experiment deliberately
withholds some targets from training. Selecting balanced subsets uses those
known labels; it is not a real active-labeling policy with an unknown pool.

If 30 labels perform well, that suggests a useful low-label operating point to
investigate. It does not mean we actually saved the time spent labeling the pool,
or that 30 recordings suffice for every new person. Later human review,
calibration, and audit labels must also count toward the research costs.

## Files and their jobs

- `experiments/configs/002_label_budget.json`: choices fixed before execution.
- `experiments/label_budget.py`: source filtering, nested subsets, fixed-update
  training, evaluation, and artifact capture; reuses the existing preprocessing
  and temporal network.
- `experiments/test_label_budget.py`: checks that withheld labels/corrections do
  not alter training, nested budgets are reproducible, and archived predictions
  match saved weights. These software checks do not establish gesture accuracy.
- The new run's `selections.json`: exactly which labels each trial used, with
  class/session counts. `development_features.npz` and `manifest.json` freeze the
  available inputs and provenance for reproduction.
- Per-trial `model.pt`, `history.json`, `metrics.json`, `session_metrics.json`,
  and `validation_predictions.json`: checkpoints, selection history, and errors.
- `summary.json`, `report.md`, and `label_budget_curve.png/.pdf`: all outcomes,
  readable comparisons, and a standalone scientific plot. Error bars measure
  subset/model run spread, not confidence intervals for new people.

## Connection to the robotics research

This establishes the supervised reference before adding unlabeled learning or
consequence-aware auditing. Later methods must beat a strong comparison at the
same total human-review budget, and preserve useful robot task completion.

Pay attention to NO_COMMAND becoming movement and STOP mistakes as well as
overall accuracy. Here these are raw clip predictions. The model is not connected
to a robot, and we have not measured live false starts or physical stopping delay.
No candidate checkpoint is silently selected for the camera application.
