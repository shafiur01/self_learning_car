# Lesson 4: a corrected label and a newly trained model

## What your correction changes

A recording has input **X** (the landmark sequence) and target **y** (your command
label). For the clip you identified, X remains identical, but y changes from
FORWARD to LEFT. We saved its old bytes and the reason for the correction.

This clip is validation data. Its corrected label affects how accurately we
judge each candidate checkpoint; it does not enter gradient updates as a training
target. Wrong validation labels could otherwise favor a model that repeats a
labeling mistake. Human meaning is authoritative; a model's guess is not a
reason by itself to replace a label.

## What we just trained

The eight-command model receives 472 training recordings. The 90 old validation
clips you authorized for reuse are now training in v2, with their previous role
preserved. It also receives UP/DOWN examples. The new 47 validation clips stay
outside gradient updates.

We trained five models from random initialization. Each received 560 updates.
An update predicts a batch, compares predictions with training labels, computes
gradients, then changes weights using Adam. Every 20 updates we checked validation
loss without changing weights and kept the lowest-loss checkpoint for that seed.

This is supervised retraining, not automatically learning as a recording is
saved. It is also not fine-tuning an existing six-class checkpoint: the new task
has eight output scores, and this baseline starts its weights afresh. We retain
all old weights and all five new candidates.

## How to read the outcome

Mean accuracy is **85.11%** across five models on 47 validation clips. Individual
models correctly classify 38, 40, 41, 40, and 41 clips. Averaging gives 40/47;
it does not construct an ensemble or turn 47 recordings into 235 independent
examples. Run variation describes training randomness on these same recordings.

The score is development validation: those same clips chose the saved training
step. A new session that did not select weights provides a different check.
Final test must remain genuinely new until choices freeze.

Overall accuracy hides class-specific errors. LEFT is 7/7 for every model and
DOWN is 5/5, but these small counts do not prove every angle/person will work.
NO_COMMAND has mean recall 53.33%; 46.67% becomes movement classes. Those are
raw guesses, not measured robot actions. This matters because a robot interface
must also distinguish commands from ordinary motion.

UP/DOWN expand the task. Changes in labels, data and validation sessions mean we
cannot compare this overall score with the old six-command score and attribute
the difference to one training improvement.

## Files that preserve the experiment

| File | Why it exists |
| --- | --- |
| Clip's `label_history` and backup | Preserve the human correction and original recording |
| Study `manifest.json` | Identifies all 519 input files and their corrected labels/hashes |
| `config.json` and `contract.json` | Record training choices and the eight meanings |
| `development_features.npz` | Preserves the exact numeric input and target arrays |
| Each `seed_N/model.pt` | Stores the trained weights and compatibility/provenance |
| Each history/prediction/metric file | Shows weight selection and every development error |
| `environment.json` / `source/` | Preserves versions and the actual code used |
| `verification.json` | Records that saved weights reproduce saved predictions |

Read [the finding](findings/003_commands_v2.md) for every seed and the limits.
The separate second validation sitting is now evaluated without fitting or
reselection. Continue with [Lesson5](LESSON_05.md) and its
[finding](findings/003_commands_v2_session_02.md) to understand carryover,
precision versus recall, and a shared confidently wrong prediction.
