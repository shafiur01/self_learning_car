# Lesson 11 — representing motion is different from learning to use it

Yes: our input contains the movements of 21 hand points over time. It contains
hand shape relative to the wrist, wrist XY displacement, apparent hand-size change
and visibility at 32 chronological time samples. It does not contain raw RGB
video, explicit joint velocity/acceleration or absolute clip duration.

Each time sample has 67 values. A temporal Conv1d filter slides across nearby
samples and combines their values; it can learn to respond to a changing finger
shape or movement pattern. Two layers learn local temporal features. Four ordered
pooled time regions feed the final classifier, so it can combine early, middle
and late evidence. We use this temporal CNN, not LSTM/BiLSTM/3D-CNN.

LSTM maintains a learned memory state while reading a sequence. BiLSTM reads the
whole supplied clip in both directions. Temporal CNN uses filters over time;
graph models can additionally encode which hand joints are connected. All are
ways to process sequences, not guarantees of recognizing intent. 3D coordinates
are data values, while 3D convolution is an operation over a defined volume;
they are different concepts.

For example, the wrist may move up while the fingers stay open, or the fingers
may curl while the wrist barely moves. Our features can represent those different
signals. But the learned weights may prefer whichever cue predicts training
labels most easily. They do not automatically discover the physical reason you
intended a command. If two identical observations mean different commands only
because of private intention, extra observable context/convention is required.

We therefore tested the saved weights without changing them. On the 48 already
inspected validation clips, each seen by 5 models:

- Zero wrist displacement: 0/240 predicted labels changed.
- Shuffle time samples: 0/240 labels changed under the chosen permutation.
- Reverse raw motion and recompute features: 1/240 changed.
- Repeat one center hand shape: 17/240 changed; BACKWARD agreement with original
  labels dropped from 70% to 36.67%.

This suggests changing articulation matters, while strong reliance on chronology
is unproven on this particular sample. Probability scores still changed, so the
motion channels are not demonstrated to be unused. The revised UP task is mostly
held-pose recognition: using pose to identify it is appropriate.

Modified inputs can be unrealistic or change the gesture. A frozen BACKWARD
motion may no longer deserve BACKWARD as its label. Thus the modified-input
scores are original-label agreement, not new valid accuracy. This is a diagnostic,
not proof that time order never matters or that removing motion improves a model.
To test learning benefits, train controlled representations/architectures under
the same data, meanings and budgets, then evaluate fixed choices on new sessions.

We also reviewed close research. Shape/motion streams, hand graphs, attention,
self-supervised pretraining, pseudo-label learning and human-label selection
already exist. Research begins by identifying what those methods still fail to
solve, specifying a mechanism and comparing it honestly, not by renaming known
blocks. Our proposed gap concerns costly confident mistakes in robot commands
and whether budgeted human audits can improve which machine labels train the
next model. Its novelty and benefit remain open questions.

Keep two studies distinct: better representations/architectures, and better
label-efficient learning policies. Otherwise a gain from more parameters or
pretraining could be mistaken for an audit algorithm improvement. Count human
confirmations/corrections/validation as labeling work and report runtime.
The full source map, falsifiers and next controls are in
[TEMPORAL_ALGORITHM_PLAN.md](TEMPORAL_ALGORITHM_PLAN.md).
