# Next pilot: gesture variation at a matched training-label budget

**Status:** superseded as the immediate task by the user's eight-command upgrade
request. This six-class matched-count comparison was not implemented or trained.
Follow [`COMMANDS_V2_WORKFLOW.md`](COMMANDS_V2_WORKFLOW.md) for current steps.

Recorded 2026-10-07, after inspecting the first new validation session and
receiving the user's description. This is a plan; no new models were trained.

## What we learned from the user

The user reports slightly varied hand angles, more deliberate direction gestures,
and creative FORWARD/BACKWARD styles somewhat different from training. This
supports investigating a change in input styles between sessions. It does not
prove angles caused the accuracy drop, confirm individual label correctness,
or identify every changed condition. Handedness and exact angles remain unknown.

Original training counts in study 002 are FORWARD 61, BACKWARD 38, LEFT 31,
RIGHT 26, STOP 29, NO_COMMAND 16. NO_COMMAND is underrepresented. STOP is less
represented than FORWARD/BACKWARD but is not the smallest class. In contrast,
the 5/10/15-per-class conditions have equal class counts. The trainer already
uses class-weighted loss. Equal counts and weighting do not supply missing styles.

Existing preprocessing normalizes wrist-relative hand shape and hand size. It
preserves hand orientation and directional motion. Arbitrary rotation or mirroring
could change command meaning, especially LEFT/RIGHT; do not apply such transforms
and assume unchanged labels. The completed jitter/time-warp study did not test
hand-angle rotations or all creative ways to express a command.

## Question and comparison

At the same 60-label training budget, does replacing half of the original-style
examples with new-session varied-style examples improve recognition on the
already-inspected 90-clip validation session?

| Condition | Original recordings/class | New training recordings/class | Total training labels |
| --- | ---: | ---: | ---: |
| A: original-only | 10 | 0 | 60 |
| B: mixed-session/styles | 5 | 5 | 60 |

This tests the bundled effect of changing training-session/style coverage at a
fixed count. It does not isolate hand angle from pace, session, or other changes.
The new collection is informed by examined validation errors, so the comparison
is a development experiment, not an untouched test or evidence of unseen-user
performance. Both conditions remain supervised.

## First human-dependent step: 30 new training clips

Start a separate recording sitting from the completed validation sitting:

```bash
source .venv/bin/activate
python -m gesture_commands.app --split train
```

Omit `--model`: collect labels without choosing clips based on live predictions.
Record **five clips for each of the six labels**, 30 total. Keep each clip about
two seconds, SPACE to freeze/review, then f/b/l/r/s/n to save; x to discard a
mistimed or incorrectly recorded clip, q to finish. Restarting the application
creates a new session; do not move the old 90 validation clips into training.

Retain the existing command meanings from `gesture_commands/README.md`. In the
new session use representative variation you want the system to support:
different reasonable hand angles and, for beckoning/pushing, finger styles.
Keep directional intent unambiguous. Five examples are a small diagnostic budget,
not a claim that five cover every possible gesture. Record the hand used and a
brief description of styles/conditions with the session ID before training.

STOP uses the existing held-closed-fist convention. NO_COMMAND means no command:
resting hands, ordinary repositioning, and unrelated movement. Include ordinary
visible-hand movements, not only no-hand clips. Avoid assigning both a command
and NO_COMMAND to indistinguishable completed motions without an observable
cue; our classifier receives hand landmarks, not private intention. If such
ambiguity is desired, additional interaction/context design needs its own study.

Do not narrow the validation gestures to make the model's score look better.
Save representative training examples of legitimate styles while keeping the
same label meanings. Label-definition changes require a new version/protocol.

## Execution design after the recordings exist

Audit and freeze the new training session before fitting any model. Require
five original recordings/class, quality/provenance checks, unchanged historical
files, and no exact duplicates with development recordings. Preserve all data.
Archive this collection as targeted after validation inspection, even though
file metadata says `source=recording`; include all additional human labeling time.

Reuse study 002's original-only 10/class models as condition A, after verifying
source/input compatibility and reproducing their predictions. For B, use the
same seeds 11/22/33/44/55 and their archived nested 5/class original subsets,
plus the five new recordings/class. Train from random initialization with the
same seed, architecture, preprocessing, Adam lr 0.001, 560 updates, batch 32,
uniform replacement sampling, and no augmentation. Do not fine-tune A into B.
Sort the five new files within each label and archive the exact training order.

Use only the archived original 72 validation clips for the 28 checkpoint checks
at steps 20,...,560 and lowest unweighted-loss selection, matching A. Report
performance on the separate 90 new validation clips for every seed. The 90 clips
already informed collection/design; keep that adaptation cost and selection
history explicit. Do not select a winning seed, ensemble, or confidence gate.

Primary outcome: paired macro F1 difference B−A on the 90 clips. Also report
accuracy, all class recalls/confusions, NLL, raw STOP/NO_COMMAND-to-motion,
per-session results, exact training labels, optimizer draws, compute, and all
label/review effort. Preserve negative outcomes. Do not claim significance or
population precision from five models on one person's same evaluation clips.

The model input budget is 60 labels per condition; **actual additional collection
is 30 human-labeled clips**. Those labels are not free. Original data acquisition
and the evaluation labels remain part of the research's cost history.

If B improves, it supports this training mixture on current development data.
It does not establish a best workflow or real-world robot reliability. If it
does not, record that and investigate representation/semantic ambiguity and
other hypotheses rather than assume semi-supervised learning will solve it.
Final claims need new people/sessions after choices freeze. SSL and robotics
audit/controller comparisons still require separate matched-resource protocols.

Implementation of this comparison is pending the new training recordings.
The strict study-002 new-session evaluator rejects added training files; do not
rerun it on a changed training pool and silently treat that as the original
frozen evaluation. Use the archived evaluation and an explicit next-study runner.
