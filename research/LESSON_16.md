# Lesson 16 — a new recording session tests our conclusion

We evaluated your96 new recordings with the115 models whose weights were frozen
before you recorded them. No retraining occurred. This asks: do the improvements
we saw on familiar development examples carry over to new recordings?

The answer is mixed. That is a useful research result, and we preserve it.

## What changed from the previous result?

The models did not change; the examples changed. Previously, the camera-motion
models averaged about98–100% on48 development clips used to choose checkpoints
and inform our design. On the new96 clips they average about93–96%.

| Model | Earlier development48 | New recordings96 |
| --- | ---: | ---: |
| CNN, camera motion | 99.17% | 93.33% |
| LSTM, camera motion | 97.92% | 94.58% |
| BiLSTM, camera motion | 99.58% | 95.83% |
| Local set, camera motion | 98.75% | 95.00% |
| Feature set, camera motion | 99.17% | 92.92% |

Each number averages five separately trained models. It is not one combined
ensemble. We did not pick a lucky seed or retrain on the new examples.

A lower new-cohort score does not by itself identify the cause. Gesture style,
coverage, recording conditions and the difficulty of individual clips may differ.
We need to inspect evidence before deciding what to change.

## Compare the right pairs

To ask whether explicit motion helps, compare a model with motion inputs against
the **same family** without explicit motion, on these same new recordings.
The full four-representation table is in [finding012](findings/012_prospective_motion_sessions.md).

LSTM improves90.21→94.58% and local set94.17→95%. CNN, BiLSTM and feature set
decline slightly in this matched comparison. Allfive families' mean accuracy
improves with motion in sitting1 but declines with motion in sitting2.

That means “motion always helps” is too strong. Our old finding described our
development set; the new evidence limits how broadly we can generalize it.
It also means we cannot yet claim that LSTM memory is the essential mechanism.
Our input velocities already describe local ordered changes, and the different
networks can learn different shortcuts or useful patterns. Their scores alone
do not tell us which computation caused an improvement.

## Overall accuracy can hide a weak command

UP recall is only65–77% for the core camera-motion families. Recall means:
of the examples that truly have this label, what fraction does the model identify?
Every class has12 unique new clips, but five seeds make60 repeated predictions
per class. Those are still only12 different situations.

All12 STOP clips are recognized by every candidate. That is favorable on this
cohort; it does not prove STOP will always work. NONE still sometimes becomes
a movement command. Such an error could matter more to a robot than a harmless
confusion between two nonmovement states. We therefore keep STOP/NONE metrics
alongside accuracy and macroF1.

The new data are from you, in two sittings on one evening. We have not evaluated
new users, different days or actual robot actions. Isolated full clips also do
not establish early command timing; that needs continuous recordings and onset/
offset/WAIT information.

## What the failure counts mean

The viewer shows46 clips missed by at least one of115 models. That does not mean
one model got46 wrong. It is the union across many architectures, representations
and seeds, intended to help us find examples worth reviewing.

UP and NONE each have11/12 clips missed by some candidate, including weaker
conditions. You can narrow the viewer to one model condition and seed to see
that model's actual failures. It also includes correct examples for comparison.

## Your next step: inspect, describe, then decide

Open the new viewer from the project terminal:

```bash
open research/experiments/prospective_motion_review.html
```

1. Start with **NO_COMMAND**, then **UP**. Watch before revealing guesses.
2. Use slow playback or scrub through the original frames. Compare a nearby
   training example; similarity does not determine the correct label.
3. Describe what you actually performed: finger/thumb shape, palm angle, held
   pose versus movement, or a possible tracking/recording mistake.
4. Save each observation, then **Export observations**. Notes remain separate;
   this does not change labels or retrain the model.

In particular, inspect NONE clip `6851961996eb4d1db1ed2d68e5819aae.npz`. Its
prompt says hands out of view, but hand landmarks exist throughout. The replay
may help distinguish what you performed from a tracking artifact or prompt
mismatch. We retain its original label until there is human evidence to review.

We saved weights, raw recordings, every probability, class/sitting metrics,
failure tables, feature audits and a scientific figure. The new
`prospective_motion_analysis.py` builds these derived reports and replays after
evaluation; it never trains or restarts the one-time evaluation.

## Why we do not immediately train on these96

The new recordings let us check frozen choices. Now that we have inspected their
results, they become development material. If we later add them to training,
we must record that decision and evaluate on different new data. Training on
them and testing on them would not measure generalization.

Your collection added96 human labels and about31.25minutes of camera/replay/
preparation/pause time. That is not pure annotation time, and future review is
additional effort. No extra fits occurred. We have not shown reduced label cost.

Our next algorithm decision should follow the review and test one clear
hypothesis under a declared budget. Your larger goal—an evidence-backed,
label-efficient learning procedure useful for robotics—remains the objective.
Today's honest mixed result helps define the problem; it is not yet a novel
algorithm or publication-ready reliability claim.
