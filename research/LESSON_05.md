# Lesson 5: evaluating unchanged weights in another sitting

## What happened this time

Your second validation sitting contains 41 clips. We loaded the five saved
models and asked each to predict those clips. Evaluation does not perform
gradient updates: it computes outputs from existing weights and compares them
with your labels. Adding validation recordings therefore did not teach the
model anything or replace a checkpoint.

The new mean accuracy is **86.34%**, compared with **85.11%** on the earlier
47 clips. These are different inputs to the same models. A score difference
does not mean the model improved through learning. It tests how well the saved
models carry over to another sitting. We report the two sets separately because
the first also selected the saved training steps.

## Accuracy, recall and precision answer different questions

Accuracy asks: of all recordings, how many received the correct label?
Recall asks: of the real examples of one class, how many did we recognize?
Precision asks: of the predictions of that class, how many were correct?

Every model recognized all five genuine DOWN clips, giving 100% DOWN recall.
But models also called some NO_COMMAND clips DOWN, giving mean DOWN precision
80.95%. Detecting every actual command does not establish that ordinary movement
will be left alone. FORWARD recall is only 64%, despite fairly high overall
accuracy. This is why a single headline number is insufficient.

Each class has only five recordings here, except RIGHT with six. A mistake on
one of five clips changes a model's class recall by 20 percentage points.
Repeating five training seeds checks training randomness; it does not replace
more people or independently recorded sessions.

## A confident model can still learn the wrong distinction

All five models predicted DOWN for the same NO_COMMAND recording, each with a
score above 99.6%. You clarified that your hand was naturally resting after
lowering, rather than properly pointing downward as a command. The human
NO_COMMAND label remains unchanged.

The score expresses how strongly the network favors DOWN among its eight
outputs. It is not a verified 99.6% chance of understanding your intention.
The models also share the same training data and network design, so all can
share a mistake. We do not yet know which cue caused this one.

A **pseudo-label** is a model-generated target used to train on an otherwise
unlabeled example. Blindly accepting this high-score DOWN prediction would teach
the same error again. A threshold alone cannot reject this specific example
at ordinary confidence cutoffs. That supports testing human audits and gesture
boundary coverage; it does not prove a proposed auditing algorithm succeeds.

## What the files do

`research/experiments/evaluate_v2_sessions.py` is separate from training. It
checks that the old recordings/weights are unchanged, identifies fresh validation
sessions, loads saved weights, and writes every result. It never runs an optimizer.

Each evaluation has its own folder. `evaluation_plan.json` identifies the models;
`manifest.json` identifies the new recordings; `new_validation_features.npz`
preserves the numeric inputs; prediction files preserve correct and incorrect
guesses; `summary.json` compares sessions separately; `verification.json` records
integrity checks. Keeping these files lets us distinguish a genuine method
change from changing recordings or selecting a lucky seed after seeing results.

See [the second-session finding](findings/003_commands_v2_session_02.md).
Next, design a controlled training comparison for the observed confusion,
collect training examples in separate sessions, and keep these 88 validation
clips outside training updates. Final test remains genuinely new until choices
freeze. This is still a supervised perception pilot, with no physical robot
experiment or demonstrated label savings.
