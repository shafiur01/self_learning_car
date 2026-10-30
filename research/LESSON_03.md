# Lesson 3: what a new recording session teaches us

## Our question

Can the already-trained models recognize your gestures in a later sitting?
You recorded 15 clips for each of six labels: **90 new validation examples**.
We evaluated the saved models without training them on these examples.

A **checkpoint** stores learned weights. During this evaluation, we loaded those
weights, transformed each clip using our existing features, and computed six
class scores. The highest score supplied the raw predicted class. We compared
that prediction to your saved human label. There was no loss backpropagation,
optimizer update, or new weight selection.

## Why the two scores differ

With 201 original training recordings, our five models averaged 93.33% accuracy
on the earlier validation session and 70.67% on this new one.

The earlier validation data helped choose which training step's weights to save.
This makes it useful for development, but the saved model was selected to work
well there. The new session did not select those weights. It provides a different
check of whether the learned patterns carry over.

Sessions can vary in hand use, gesture style, pace, position, light, background,
or tracking behavior. Some changes may affect our landmark features and others
may not. We have not established which differences caused this drop. It is
evidence of limited cross-session generalization on these recordings, not proof
that a particular preprocessing change or algorithm will fix it.

For example, if a model learns a beckoning motion from one hand/style, it may
fail on another way of beckoning that a person readily understands. We must
define what belongs to each command and test the variation we intend to support.

## Accuracy alone hides the main problem

For the all-original models, mean recall is:

| True label | Fraction recognized correctly |
| --- | ---: |
| FORWARD | 52.00% |
| BACKWARD | 81.33% |
| LEFT | 90.67% |
| RIGHT | 93.33% |
| STOP | 82.67% |
| NO_COMMAND | 24.00% |

**Recall** asks: out of the clips that really have this label, how many receive
that label? For example, FORWARD recall of 52% means an average 7.8 of the same
15 FORWARD clips are correct per model. The decimal reflects averaging five
models; an individual model always gets a whole number of clips correct.

The new errors matter to our robotics question. Ordinary movement labeled
NO_COMMAND often becomes a movement command. But these are raw clip predictions;
we have not tested whether the live three-window gate would issue those commands,
or what any robot would do. A gesture classifier's accuracy is not a measurement
of physical robot safety or task success.

## Why confidence is not correctness

Some incorrect movement predictions on STOP/NO_COMMAND receive scores above
0.95. A softmax score reflects how strongly this model prefers a class over the
others. It does not by itself establish a verified 95% chance of correctness.

This is directly relevant to **pseudo-labeling**: assigning a model's guess as a
training label for an unlabeled clip. If we admit confident mistakes unchecked,
we may train the next model to repeat them. Our planned review/audit mechanism
must be experimentally compared against strong alternatives; we have not yet
shown it improves recognition or saves human time.

## Why we keep these files

[`experiments/evaluate_sessions.py`](experiments/evaluate_sessions.py) runs this
comparison. It checks that the recordings are new sessions, the old files and
features remain compatible, and the checkpoints reproduce their original
predictions. It rejects duplicate clips and saves every model's results.

Inside the saved evaluation:

| File | Why it exists |
| --- | --- |
| `manifest.json` | Identifies exactly which new recordings were evaluated |
| `evaluation_plan.json` | Records every frozen model and its hash before inference |
| `new_validation_features.npz` | Preserves the input arrays used for prediction |
| `predictions.json` in each trial | Lets us inspect each prediction and its six scores |
| `metrics.json` / `trials.json` | Preserves all class errors and all seeds |
| `summary.json` / `report.md` | Makes the condition comparison readable |
| `source/` / `environment.json` | Preserves the source code and versions needed to reproduce it |

The written [finding](findings/002_new_session_01.md) explains both favorable and
unfavorable results. Keeping disappointing results is part of doing research.

## The next step

You have now described slightly changed hand angles and creative FORWARD/BACKWARD
styles. This makes missing style coverage a hypothesis to test. Original
NO_COMMAND has 16 examples, STOP 29, and FORWARD 61, but our balanced small
subsets also struggled: equal counts do not create more kinds of examples.
The [next pilot](SESSION_DIVERSITY_PLAN.md) compares original-only training with
a mixture of original and new varied-style recordings, each totaling 60 labels.
It needs five new training clips per label, 30 total, in a separate sitting.

A NO_COMMAND example should have no intended instruction; it is
not an unlabeled command. No hand or an ordinary hand movement may both be
examples, but the class must cover the situations our interface will encounter.

Keep this whole session in validation. Training examples must come from separate
training sessions. We can use validation to develop changes, but after studying
its errors it is no longer an untouched final benchmark. Final claims require
genuinely new sessions and, for our unseen-user objective, new people.

More examples from the same sitting and more independent sittings answer
different questions. Your 90 clips are useful; they represent one person's one
new sitting, not 90 independent users or five times as much evidence because
five models were evaluated.
