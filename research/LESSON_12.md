# Lesson 12 — an experiment can improve accuracy and worsen an important error

Your research target has three connected parts:

1. Learn how ML works well enough to carry out experiments independently.
2. Develop a method that uses fewer human labels/review minutes at reasonable
   computation cost, while recognizing gestures under new sessions and people.
3. Show why the method helps robotics through measured command errors and robot
   outcomes. A paper for ICML, ICLR or a strong journal needs a defensible new
   contribution and convincing comparisons; we cannot promise acceptance.

The current application is a perception prototype. We have no robot/hardware
results yet. Our first task today was to test representation/architecture, before
testing the proposed learning policy. Separating these avoids crediting a new
label-selection algorithm for a gain that really came from a different network.

## The question we tested

Would remembering the sequence of hand observations improve recognition?
The current CNN processes short temporal patterns. LSTM maintains a learned
state while reading successive samples. BiLSTM reads a completed supplied window
both ways. A pose-set baseline encodes each hand shape separately and averages
those encodings. Its average cannot distinguish the order of the same poses.
A feature-set baseline also includes the wrist/size channels before averaging.
Those channels are measured relative to the beginning and can preserve some
trajectory information even though their sequence order is unavailable.

These are established model families, not our invented algorithm. Their weights
are trained from random initialization on our data; MediaPipe's upstream hand
detector remains pretrained. All models use the same 538/48 training/validation
split and have roughly 49,000 trainable parameters. Parameters are the numbers
the optimizer adjusts, not the number of training clips.

## How we make a useful comparison

We held the labeled clips, meanings, preprocessing, batch size, number of
updates, optimizer, class weighting and checkpoint-selection rule constant.
Each trial makes 560 updates of 32 sampled clips: 17,920 clip draws, with repeats
from the 538 unique training clips. More draws do not create more observations.
Equal updates also do not equal equal computation; we measured actual runtime.

We trained each family with five seeds. A seed initializes the random choices;
we matched the sampled clips across models for each seed. Different networks
cannot start with identical weights. Five seeds expose variation rather than
letting us pick a lucky run. They still share one validation dataset, so they
are not five independent studies of different people.

Every 20 updates, we calculate validation loss without updating model weights.
We save the checkpoint with the lowest loss, even if its accuracy is not the
highest seen. Because validation chooses this checkpoint, its reported score
is development evidence. Final evaluation needs new data after choices freeze.

## What happened

| Model | Mean validation accuracy | NO_COMMAND predicted as movement |
| --- | ---: | ---: |
| Current CNN | 85.83% | 13.33% |
| Pose set | 85.83% | 16.67% |
| Feature set | 85.42% | 13.33% |
| LSTM | 89.58% | 33.33% |
| BiLSTM | 88.75% | 10.00% |

LSTM is 3.75 percentage points higher in average accuracy than CNN. That is a
development improvement under this recipe. It recognizes more FORWARD/BACKWARD
examples, but mistakes more ordinary/no-command observations for movement.
Of the six NO_COMMAND clips, each seen by five models, CNN makes 4/30 raw
movement predictions and LSTM makes 10/30. Those are repeated predictions on
six observations, not 30 independently recorded gestures.

This is why accuracy alone is insufficient for a robot. Correcting several
directional gestures can raise total accuracy while additional unintended
movement commands create a worse interface. We did not connect the predictions
to motors or evaluate the application's gate, so these are recognition errors,
not measured collisions or physical motion.

BiLSTM has fewer NO_COMMAND-to-movement predictions here, but still misses the
same two STOP examples in every seed. Its STOP recall is 66.67%, matching CNN.
LSTM's average STOP recall is 76.67%, but its results vary more across seeds.
Neither candidate solves all the important errors. High-scoring wrong commands
also remain: one LSTM false movement prediction scores about .995.

Macro F1 gives each class equal weight in averaging its precision/recall balance.
It is different from accuracy: LSTM's .8909 macro F1 is not its 89.58% accuracy.
Both measurements help, but neither replaces class-specific command errors.

The set baselines approximately matching CNN suggests useful pose cues in this
dataset. It does not prove sequence order never matters. Recurrent improvements
also do not prove their mechanism is specifically useful trajectory reasoning;
the architectures and optimization behave differently. More targeted comparisons
and new-session/person evaluations are needed.

## Where the work is saved and why

- `experiments/temporal_comparison.py`: model definitions, fixed fitting recipe,
  isolated workers, strict checkpoint loading and result verification.
- `experiments/test_temporal_comparison.py`: checks gradients, permutation
  invariance, feature isolation, preserved inputs and tamper detection.
- `TEMPORAL_COMPARISON_PROTOCOL.md`: records choices before real results existed.
- `findings/008_temporal_architectures.md`: reports both gains and regressions.
- `experiments/runs/temporal_comparison/20261014T011524632211Z/`: every model,
  loss history, score vector, input/source fingerprint, costs and scientific plot.

All 25 weights and 1,200 saved score vectors were checked, and the five CNN fits
reproduced the original models. Old recordings/checkpoints remain preserved.
The camera still uses its existing explicitly selected model.

The next reasoning task is to investigate why STOP/NONE are confused with motion
without automatically treating model guesses as correct labels. Then we can
test the main hypothesis: whether budgeted human audits prevent confident errors
from becoming harmful pseudo-labels during semi-supervised learning. An unlabeled
clip has an unknown label; it must not be stored as NO_COMMAND. Fewer requested
training labels, validation effort and human minutes must all be accounted for.
