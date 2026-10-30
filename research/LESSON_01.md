# Lesson 1 — what makes this a useful experiment?

The first question is small: does transforming our existing training recordings
help the same model classify a separate recording session?

## Start with a baseline

The baseline is the current training procedure. It gives us a point of comparison.
If we changed the model, optimizer, data amount, and transformations all at once,
an improved score would not tell us which change helped. Here those factors stay
fixed and only the transformation condition changes.

We compare all recordings across five different seeds. Each seed changes the
initial random weights and order of training batches. Within a seed, the four
conditions share those choices. This is a paired comparison: differences within
a pair are more informative than comparing unrelated lucky and unlucky runs.

## What the augmentation actually does

Every model input is 32 rows × 67 values. The original recording is preserved;
the runner transforms a copy of a training batch before passing it into the CNN.

Jitter adds very small, smooth perturbations to local joint coordinates. For
example, a normalized coordinate of 0.40 may become about 0.41. The intention is
to discourage dependence on one exact observed shape. The scale is a starting
hypothesis, not a measured specification of MediaPipe noise.

Time warp changes how quickly the gesture progresses through different parts of
its motion. The sequence still starts and ends at the same points, progresses
forward in time, and remains 32 frames. A gesture might reach its midpoint a bit
earlier or later. No LEFT/RIGHT reflection or temporal reversal is introduced.

Both techniques assume the target command remains the same. That assumption can
fail: exact timing or small shape changes may matter. This is why we measure the
result rather than declaring augmentation helpful in advance.

## Training loss versus validation

The training loss supplies gradients that update weights. A low training loss
alone can mean the model memorized our recordings. Validation uses different
recordings, receives no augmentation, and does not supply gradient updates.

We keep the epoch with the lowest validation loss. That choice already depends
on validation, and then we also compare conditions on validation. Therefore the
reported scores are development scores. Final evidence needs new independent
sessions after our method choices are settled.

## Why save so many artifacts?

An accuracy number alone cannot tell us what happened. The manifest identifies
the recordings; file hashes identify their bytes; the saved array freezes the
actual model inputs. Source copies and package versions identify the procedure.
Loss histories show learning across epochs. Predictions and confusion matrices
show which commands were mistaken for which other commands.

Our five seeds reuse the same 72 validation clips from one session. They are five
training repetitions, not five independent validations across people or days.
Macro F1 gives all six classes equal weight, while accuracy counts all correct
clips together. Neither statistic measures live false starts between gestures.

## Where this fits in the labeling research

This first pilot uses every current human label. It asks whether we can strengthen
the baseline without asking for more labels. It does not yet show that fewer
labels achieve the same performance.

That later question needs a label-budget experiment: hide some training labels,
keep evaluation consistent, and compare methods at equal budgets. For true cost
claims we must also record human labeling/review time. Semi-supervised training
will require a separate unlabeled pool and controls for incorrect pseudo-labels.

When reading the results, identify the mean effect, variability across seeds,
class-level mistakes, and the amount of independent evidence before choosing
the next step. An unfavorable result can rule out a starting setting and is
still a useful research record.
