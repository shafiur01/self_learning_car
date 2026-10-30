# Lesson 6: testing a useful idea and finding a mixed result

## What we tried to learn

We suspected that examples focused on confusing gestures would help more than
ordinary additional recordings. To test that idea, both groups received the
same 472 existing training clips and 40 extra human-labeled clips. One received
familiar gestures; the other received focused resting/downward and directional
variations. Counts per class were equal. Both groups came from both sittings.

The question was about the choice of examples. We did not change the network,
features, training duration or prediction gate to favor one group. If many
choices change together, we cannot tell which change mattered. Here the prompt
instructions still bundle style/coverage differences, so we cannot isolate
one angle or finger configuration as a cause.

## How the training was controlled

For each of five seeds, the two models began with identical random weights and
received the same sequence of batch row indices. Their 512 rows align: identical
base examples first, then corresponding class/sitting/repeat positions. The new
examples in those positions differ; class weighting and update budget match.
We saved hashes that verify matching initial weights and batch indices.

Each model performed 560 updates of 32 examples: 17,920 training presentations.
Those are repeated presentations from 512 labeled clips, not 17,920 new labels.
An update predicts a training batch, compares it with your labels, computes
gradients and changes weights using Adam. Repeated exposure is learning; saving
a recording by itself does not change weights.

Every 20 updates, the original47 validation clips supplied checkpoint-selection
loss. We saved the lowest-loss weights for each seed. The other41 clips were
used after selection to report development outcomes. Neither set supplied
gradient updates. Because both already informed our planning, they remain
development data, and a final claim needs genuinely new evaluation.

## What actually happened

| Model group | Mean accuracy on the41 monitoring clips | Mean macro F1 |
| --- | ---: | ---: |
| Original weights | 86.34% | 0.8577 |
| Routine additions | 88.29% | 0.8788 |
| Targeted additions | 85.37% | 0.8497 |

Our primary measure was targeted-minus-routine macro F1: **−0.0291**. Targeted
scored lower in four of five matched runs. The expected advantage did not appear
in this pilot. We keep that finding rather than change the benchmark or select
one lucky seed. This does not prove focused collection is always ineffective.

Both additions corrected the specific resting-hand clip that all original
models called DOWN. But another non-command clip still became DOWN, and other
classes remained difficult. Fixing one known mistake is different from learning
a reliable rule across examples. Routine STOP/NO_COMMAND recall increased, but
UP recall fell from84% to72%; FORWARD stayed64%. More data can help some classes
while changing decisions for others.

Macro F1 combines precision/recall and treats every class equally. Accuracy
counts correct clips overall. All targeted models got35/41 correct, but their
macro F1 and mistakes differ. An identical headline accuracy does not mean
identical weights or behavior. Five seeds reuse41 clips, not205 independent
human examples; run spread describes training randomness.

## What was saved, and what comes next

`research/experiments/boundary_train.py` exists to enforce this comparison. Its
fitting function receives only training and selection tensors, so monitoring
cannot choose weights. It creates a new study instead of replacing checkpoints.

The study's `manifest.json` identifies all inputs and row roles. Its feature
archive keeps base/additions/selection/monitoring separate. Each `model.pt`
stores trained numerical weights; histories show their selection, prediction
files show every correct/incorrect guess, and `verification.json` records that
saved weights reproduce those guesses. Full results and paired plots are in
[the finding](findings/004_boundary_data.md).

We acquired80 usable new labels even though each compared model uses40. The
discarded16 and all previous annotation effort still count. The logged34.51
minutes include capture/replay/preparation/pauses; they are not pure annotation
time. This experiment measures a supervised development comparison, not labeling
savings or a working semi-supervised algorithm.

The next proposed step is to freeze a comparison policy and evaluate the saved
candidates on genuinely new sessions before trusting the observed difference.
No camera model was automatically changed, and no physical robot was tested.
