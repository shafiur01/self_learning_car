# Lesson 14 — give the network shape, motion and a clock

Your review identified the right distinction: one open-palm frame might belong
to STOP or to a moving FORWARD gesture. A classifier needs evidence about the
surrounding hand changes. STOP also needs the appropriate shape; a stationary
unrelated pose must not automatically become STOP.

We tested explicit motion and a hand reference while keeping one temporal model
fixed. This is a feature-engineering experiment, separate from the main future
label-efficient learning policy.

## Position, displacement and velocity

Position describes where a joint is. Displacement is the change between two
observations. Velocity divides displacement by elapsed time:

`velocity = (position_now − position_before) / seconds_between_samples`

For example, moving one normalized palm length in .5 seconds gives two palm
lengths/second; the same change in one second gives one palm length/second.
Those are normalized coordinate units, not calibrated real-world metres.
We explicitly tested that doubling timestamps halves our computed velocities.

The old input resampled each clip to 32 chronological samples, but did not expose
absolute duration/sample interval or calculate velocity. Now the model sees
shape, wrist direction, size change, real-time derivatives, tracking masks and
elapsed seconds. A LSTM can learn changing patterns from positions alone; explicit
derivatives make some changes easier to use, rather than magically adding new
recorded information.

Our derivative validity mask prevents hand loss/reappearance or large timestamp
gaps from becoming artificial movement spikes. Missing data is different from
a hand that is genuinely still. We do not treat visibility changes as velocity.

## A hand reference without deleting command direction

We center on the wrist and scale by estimated 3D palm length. Anatomical axes
come from the wrist, middle-finger knuckle and index/little-finger knuckles.
The hand-coordinate version expresses every joint along those axes. This can
make finger relationships easier to compare when the palm is tilted.

But freely rotating every gesture into the same view could remove LEFT/RIGHT
information. We therefore retain the palm's orientation relative to the camera
and the global wrist movement separately. Tests reconstruct camera-oriented
shape from the hand coordinates plus that orientation, before clipping.
The model receives the information; it still has to learn how to use it.

MediaPipe depth is estimated and relative, so mathematical coordinate properties
are not proof of real viewpoint robustness. A computed hand basis may also move
with tracking noise or changing palm geometry. Normalization is a design choice
to test, not a guaranteed improvement.

## Four conditions, one LSTM

All use the same 49,276-parameter network, initialization, sampled clips, 560
updates and five seeds. The first two rows differ by whether per-second movement
is explicitly provided; the last two repeat that comparison in hand coordinates.

| Representation | Mean validation accuracy | NO_COMMAND→movement | STOP recall |
| --- | ---: | ---: | ---: |
| Camera / position | 84.17% | 36.67% | 73.33% |
| Camera / motion | 97.92% | 0% | 93.33% |
| Hand / position | 91.67% | 13.33% | 86.67% |
| Hand / motion | 92.08% | 6.67% | 96.67% |

The matched camera-motion model improves both recognition and NO_COMMAND errors
here. Hand alignment helps position inputs but does not outperform camera motion
overall. Hand motion recognizes STOP slightly better while making more other
errors. We preserve that result instead of assuming our preferred transformation
must win. Two camera-motion runs score 100%, but the five-run mean is 97.92%.

This new camera-position baseline differs from our earlier 67-feature LSTM in
scaling, timing/orientation channels and hidden width. Compare the matched rows
above to isolate today's intended differences; do not attribute every difference
from earlier experiments solely to velocity.

These are the same 48 already-inspected validation clips from one user, six per
class. Validation chooses checkpoints and has influenced our design. Zero errors
on six NO_COMMAND clips across five models does not mean a zero real-world error
rate. We have not shown SOTA, a new algorithm or saved human labeling effort.

## Model processing time and decision time are different

The camera-motion network takes about .184 milliseconds for a preprocessed
window locally. That excludes capturing frames and understanding the gesture.
Waiting for enough evidence may dominate robot response time.

We also froze the weights and supplied only observations captured by .5, 1.0
or 1.5 seconds of each recording. Camera-motion agreement with the recorded
full-clip intent was 92.08%, 96.25% and 98.33%. Prefix features were recomputed
independently, without using later frames to normalize or interpolate them.

Those numbers are an exploratory prefix probe. An early hand shape may genuinely
be ambiguous, models trained only on full clips face shorter inputs, and our
recording starts are not annotated gesture onsets. It would be incorrect to call
these measured robot response times or validated early-command accuracies.
We need continuous recordings, onset/end annotations, ordinary motion between
commands, an explicit wait/defer option, false commands/minute and STOP delay.
NO_COMMAND and "not enough evidence yet" are different meanings.

## What we implemented and what comes next

- `experiments/landmark_representations.py`: versioned geometry, clocks and
  masked velocity calculations, leaving the old application features intact.
- `experiments/motion_representation.py`: fixed-budget matched fits, saved
  models, prefix probes and artifact verification.
- `experiments/test_motion_representation.py`: mathematical properties and
  tiny training/checkpoint checks, without interpreting synthetic scores.
- `MOTION_REPRESENTATION_PROTOCOL.md`: choices recorded before real results.
- `findings/010_motion_representation.md`: complete gains, failures and limits.

All 20 models and 3,840 full/prefix prediction vectors were verified; 2,920
feature arrays reproduced from original recordings. Old data/checkpoints remain
preserved, and the camera application has not automatically switched models.

Next we need genuinely new session/person/speed/view evidence after freezing
choices. Speed comparisons should distinguish active movement speed from held
poses, and camera-relative versus robot-relative direction must be defined.
Explicit bone vectors/angles, hand graphs, causal temporal networks and motion-set
baselines are later controlled comparisons. Good motion recognition is a stronger
foundation for the separate low-label/audited learning experiment; it does not
finish that research question.
