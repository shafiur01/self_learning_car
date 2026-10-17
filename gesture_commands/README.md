# Gesture commands: learn your own movements

This is a separate, screen-only application. The original `python main.py` hand
tracker still works. This application reuses its model path and drawing function.

**Current status:** recorder, temporal-model training, held-out evaluation, and
live prediction/correction code are implemented. Models have been trained on
your recorded gestures and retrained with corrections. Research comparisons and
their limitations are tracked in [../research/CONTEXT.md](../research/CONTEXT.md).
There is no serial, network, ESP32, or motor-control code here.

## Why a custom model?

MediaPipe's built-in Gesture Recognizer recognizes a limited set of hand poses:
open palm, closed fist, pointing up, thumbs up/down, victory, and I-love-you.
Its standard classifier does not provide your natural movement commands.
Video input alone does not imply that a classifier learns motion sequences.

We use the existing pretrained hand detector, then train a small PyTorch 1D CNN
on short sequences of coordinates. A temporal CNN learns patterns across time.
It can learn several versions of a gesture under one label, provided you supply
examples of those versions. It will not understand every unseen variation.

Sources:
- https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer
- https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

## Gesture meanings for this prototype

| Label | Starting definition | Save key |
| --- | --- | --- |
| FORWARD | Beckon toward yourself with one, several, or all fingers | f |
| BACKWARD | Push your hand outward to signal go away | b |
| LEFT | Point toward the left side of the mirrored preview | l |
| RIGHT | Point toward the right side of the mirrored preview | r |
| STOP | Hold a closed fist still (keep distinct from beckoning) | s |
| NO_COMMAND | Resting hand, unrelated movement, transitions, or no hand | n |

These are labels **you teach**, not hard-coded pose rules. If two visually
identical actions get different labels, the model cannot infer your intention.
Keep camera viewpoint, mirroring convention, and meanings consistent.
Training and prediction both process unmirrored coordinates and mirror only
the preview. Do not horizontally flip training samples without also adjusting
direction labels. The model does not rotate hands upright or merge left/right
hands; demonstrate both hands if both must work.

The preview convention is not yet a mapping to the car's frame of reference.
Resolve and test that mapping before integrating hardware. FORWARD will mean
forward movement, not automatically finding you, steering toward you, or
measuring how far away you are.

## 1. Record your first examples

Run commands from the repository root with the virtual environment active:

```bash
source .venv/bin/activate
python -m gesture_commands.app --split train
```

Dependencies were installed for you. To reproduce the setup later:

```bash
python -m pip install -r gesture_commands/requirements.txt
```

Close the previous camera demo first. Click the new camera window so it receives
keypresses. Start with **five examples per label just to learn the recorder**:

1. Keep one hand visible. Perform one complete gesture over about two seconds;
   for a pointing or STOP pose, hold it for two seconds.
2. Immediately press **Space**. The last two seconds freeze and replay.
3. Watch the replay. Press the label key only if it shows the intended action.
   Press **x** to discard it if you were late or tracking failed.
4. A save clears the buffer. Perform a fresh gesture for the next example.
5. Press **q** to finish the session.

The model records all 21 points, timestamps, visibility, label, and session ID
in `data/*.npz`. Camera images exist only in memory for replay and are not saved
or uploaded. Each file is an independent clip, not an individual video frame.
Severe tracking loss rejects movement labels. No/multiple hands stop prediction;
detected hand changes and capture stalls reset the motion history. This is not
a multi-person identity tracker: use one person and one hand at a time.

For an initial experiment, aim for **30–50 independent clips per class across
several sessions**, then expand weak classes. This is a starting budget, not a
reliability guarantee. Record different speeds, hand positions, distances, and
finger combinations, including the future camera height. Include ordinary
movements and gesture transitions as NO_COMMAND. Keep gestures near two seconds
initially; much shorter/longer gestures require later windowing improvements.

## 2. Record separate validation and test sessions

```bash
python -m gesture_commands.app --split validation
python -m gesture_commands.app --split test
```

Quit each run before starting another. Each run creates a new session ID.
Record all six labels in each split, ideally in separate sittings. Aim for
10–15 clips per class in validation and test for the first experiment.

- Training examples change the model's weights.
- Validation examples help choose which saved training epoch works best.
- Test examples provide an untouched check after model choices are settled.

The training command requires at least five training and two validation clips
per class, but that minimum only permits a pipeline experiment. It does not
establish reliability. Whole sessions stay in one split; neighboring frames are
never randomly split. If generalizing to new people is a goal, reserve entire
people for testing, not just sessions. Do not keep tuning against the final test
set; it then becomes development data and you need a new untouched test set.

## 3. Train

```bash
python -m gesture_commands.train
```

An epoch is one pass through the training examples. The program trains for 80
epochs by default, saves the epoch with the lowest validation loss, and writes
a new timestamped folder under `runs/`. It prints the exact path to `model.pt`
and saves a validation confusion matrix (actual labels are rows, guesses columns).
Existing models are preserved. Training runs on CPU for this small baseline.

No checkpoint is silently installed as the active model. Choose the new file
explicitly when running the application. Scores are not calibrated probabilities
of correctness, and high scores do not prove an unfamiliar gesture is known.

## 4. Try predictions and correct mistakes

Replace `YOUR_RUN` with the folder printed by training:

```bash
python -m gesture_commands.app --model gesture_commands/runs/YOUR_RUN/model.pt
```

The program repeatedly evaluates the recent two-second window. It displays the
gesture prediction and a separate proposed command. Movement needs three
consecutive qualifying predictions (checked about every 0.2 seconds), a score
of at least 0.8, and a top-vs-second score gap of at least 0.2. These are initial
demo settings to validate, not guarantees. NO_COMMAND, uncertain results,
missing/ambiguous hands, incomplete windows, and replay all produce STOP.

When it guesses incorrectly, immediately press **Space**. Review the captured
clip, then press its correct label key. This saves that exact sequence plus the
raw model guess and scores, preventing accidental labeling of a later movement.
You can also save correct examples. Corrections are always training data.

Correction labels do not change the running model immediately. Quit, run the
training command again, and compare the candidate with the previous model using
validation and separate practice sessions. Retraining includes earlier examples
as well as corrections to reduce forgetting. Keep data balanced; correcting one
class hundreds of times can bias the dataset. Train/test selection is a deliberate
step; nothing retrains in the background.

## 5. Final held-out evaluation

```bash
python -m gesture_commands.train --evaluate gesture_commands/runs/YOUR_RUN/model.pt
```

This reports test accuracy and the confusion matrix without training. The test
sessions must not have developed the model. Test continuous video separately:
trimmed-clip accuracy does not measure accidental commands between gestures.
Measure false starts, missed STOP gestures, and reaction delay before hardware.

## Understand the code

| File | Role |
| --- | --- |
| `app.py` | Camera, rolling buffer, frozen replay, labels, live predictions |
| `data.py` | Recording format, common preprocessing, command stability filter |
| `model.py` | Small temporal network and compatible checkpoint loading |
| `train.py` | Training, validation-based selection, and separate test evaluation |
| `test_pipeline.py` | Automated checks using synthetic data, not real gestures |

Each raw clip contains N frames × 21 landmarks × 3 coordinates. `features()`
resamples by actual timestamps into 32 frames, making modest speed differences
easier to learn. Every frame has 67 input values:

- 63 wrist-relative, hand-size-normalized joint coordinates;
- two wrist-displacement values, preserving lateral motion;
- one hand-size-change value, a visual cue for approach/retreat;
- one visibility flag, distinguishing missing data from valid coordinates.

Hand size and landmark z are not accurate measurements of distance to a person.
The network combines nearby time steps using `Conv1d`, then predicts six label
scores. Training compares those scores with your labels and adjusts numerical
weights to reduce mistakes. The same preprocessing is used in both modes.

## Later hardware stage

Before motor integration, agree on the car's left/right convention and add an
ESP32 command timeout, explicit STOP priority, and independent obstacle/distance
handling. This screen demo has roughly a two-second observation window plus
stability delay, so its learned STOP gesture is not an emergency stop. Use an
independent physical stop for hardware tests. Pi compatibility and camera-to-command
latency must be tested on the actual Pi; the current package pins target this Mac.

Run automated checks from the project root:

```bash
python -m unittest gesture_commands.test_pipeline -v
```
