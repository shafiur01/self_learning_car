# Experiment 004: which additional examples help?

Prepared: 2026-10-08 (America/Chicago). **Collection and fitting completed.**
Both sittings supplied40 clips (80 total), balanced routine/targeted groups,
all tracking checks passing. Study `20261008T072040903427Z` trained five fresh
models/group and retained all five references. Routine/targeted monitoring
accuracy88.29%/85.37%, macroF1 .8788/.8497: targeted advantage absent in this
pilot. See [finding004](findings/004_boundary_data.md) and [Lesson6](LESSON_06.md).
The prespecified collection/training instructions below remain historical;
current proposed next phase is joint candidate freeze/policy and genuinely new
evaluation. No final freeze/test or camera activation occurred.

## Question and comparison

At the same additional label budget, do examples focused on observed confusions
help more than additional familiar-style examples? The confusions motivating
this pilot are relaxed/resting NO_COMMAND→DOWN and FORWARD↔BACKWARD. They were
identified during development validation, not final testing. Missing coverage
is a hypothesis; predictions do not identify the learned cue or prove a remedy.

| Condition | Existing training | New training examples | Total model training labels |
| --- | ---: | ---: | ---: |
| Existing reference, all five saved seeds | 472 | 0 | 472 |
| Routine additions | Same 472 | 40 familiar-style clips | 512 |
| Targeted additions | Same 472 | 40 clips focused on observed distinctions | 512 |

Both added groups have exactly 10 FORWARD, 10 BACKWARD, 10 NO_COMMAND and
10 DOWN clips. Class counts, network, preprocessing, update budget, loss weighting
and validation policy match between the two added groups. The saved reference
has a smaller training pool, not an equal-label budget. No winning seed or
ensemble is selected retrospectively.

Collect **80 new human-labeled clips total** across **two new physical sittings**.
Each sitting contributes five of each of the four classes to each group:
20 routine + 20 targeted = 40. Both groups occur in both sittings; routine comes
first in sitting 1, targeted first in sitting 2. Keep camera, hand and environment
consistent within a sitting. These controls reduce simple order/session
confounding but do not establish independence or isolate one causal property.
This is a bundled instruction/coverage comparison, not a random-sampling baseline.

The 80 new labels, preparation/replay/review and failed attempts count as effort.
Each model sees 40; that does not reduce project cost to 40. Existing labels,
inherited corrections and retired validation acquisition still count. Camera
attempt logs measure elapsed capture/replay/preparation/pause intervals, not
pure annotation time. Record offline preparation/review separately for later
total human-cost claims. These are provisional single-user diagnostic quantities,
not a publication power calculation or demonstrated labeling savings.

This is supervised collection and an eventual controlled comparison. No
pseudo-labeling, self-supervised pretraining, new architecture or new algorithm
is introduced. It is a diagnostic foundation for later audit/SSL studies.

Implemented runner: `experiments/boundary_train.py`. It reads the frozen plan
and complete collection, aligns paired training rows, reproduces base features,
keeps original47 selection/second41 monitoring separate, saves all candidates
and verifies their archived predictions and unchanged inputs. It does not
silently load a camera checkpoint. Running it again creates a new study, not
new independent evidence on these same recordings.

## Current human step: sitting 1

From the project folder, with the existing virtual environment:

```bash
.venv/bin/python -m research.experiments.boundary_pilot record --sitting 1
```

The plan is already prepared. The recorder opens the existing camera/MediaPipe/
replay interface without a gesture checkpoint. It shows clip number, group,
required label, instruction and saving key.

1. Read the instruction. Show one visible hand and perform it for roughly
   two seconds, long enough to replace the previous buffer.
2. Press SPACE. Review the frozen replay and confirm it matches the prompt.
   Press x if it does not; the task stays the same.
3. Press the displayed key: f FORWARD, b BACKWARD, n NO_COMMAND, d DOWN.
   Only the current task's key saves. Pressing it is your human confirmation.
4. Follow the next prompt. After 40 saves, press q to close the window.

There are five clips/class/group in this sitting. Routine prompts ask for usual
comfortable gestures. Targeted prompts include relaxed naturally lowered/resting
hands, ordinary repositioning, deliberate downward pointing/whole-hand motion,
and clear beckoning versus pushing-away with representative style variation.
The eight existing meanings stay unchanged. Do not make routine examples
unrepresentative to create an easier comparison.

For NO_COMMAND, keep the hand visible and naturally relaxed; do not make all
five hand-absent negatives because the observed error involved a visible hand.
A relaxed hand at a low position is not automatically DOWN. For deliberate
DOWN, visibly express the directional gesture. If two motions look identical
and only private intention changes, discard and report the ambiguity: landmarks
cannot directly observe intention. STOP remains the existing held fist; do not
use it as NO_COMMAND. Review RGB replay before saving; only landmarks are stored.

q pauses without losing saved clips. The same command resumes the next task;
resume sitting 1 only while it is genuinely the same physical sitting. If a
later restart should count as another sitting, tell us so the plan can be
versioned explicitly rather than silently mixing identities.

Optional: append `--hand right` or `--hand left` for your actual hand. Leaving
it off records `unspecified`. Click the camera window to receive key presses.

Inspect progress without opening the camera:

```bash
.venv/bin/python -m research.experiments.boundary_pilot status
```

## Later human step: sitting 2

Finish sitting 1, then return after a break for another genuine sitting. The
recorder reverses group order and collects the other 40 clips:

```bash
.venv/bin/python -m research.experiments.boundary_pilot record --sitting 2
```

A different capture-session identifier alone does not prove a new physical
sitting. All added recordings remain training; do not split their physical/
capture sessions between training and evaluation.

## Fixed fitting/evaluation policy before additional fitting

After collection/provenance audit, prepare two new studies. Use the existing
67-feature preprocessing and eight-output temporal CNN, with fresh weights
for seeds11/22/33/44/55. Each condition gets the same 472 base clips plus its
own40. Existing recipe: weighted CE, Adam .001,560 updates,32 uniform-replacement
samples/update,no augmentation,28 validation checks. Align base rows and added
class/sitting/repeat positions across conditions for paired batch-index streams.

**Only original47 v2 validation clips select checkpoints.** Report the41-clip
second sitting separately after weight selection. Neither session enters
gradient updates. Both already informed development; neither is final untouched
test. A comparison runner follows the frozen new-data audit; this collector
does not fit models. Do not use ordinary v2 `train` to silently combine these
isolated groups or use all88 clips for selection.

Primary comparison: targeted minus routine **mean macro F1** over all five
seeds on the41-clip monitoring session. Report every seed and descriptive paired
differences. Secondary: accuracy,NLL,all class precision/recall,raw NO_COMMAND→
DOWN/all-motion rates,FORWARD↔BACKWARD counts,STOP recall/to-motion errors,
training time and human effort. Keep selection and monitoring outcomes separate;
record mixed and unfavorable results. The single shared error is a diagnostic
case, not the primary benchmark or a duplicated training example.

No confidence/gate/controller change is bundled into this study. Five models
share the validation recordings; run SD is training variation, not population
uncertainty. No robot command rate is measured. Final evaluation is later:
freeze choices, then collect genuinely new sessions/participants appropriate
to the claim. Unseen-user/robot reliability, novelty, optimal acquisition and
publication readiness remain unestablished.

## Files and preservation

| File | Purpose |
| --- | --- |
| `experiments/boundary_pilot.py` | Prepare, show recording prompts and audit progress |
| `experiments/test_boundary_pilot.py` | Synthetic task, hash, session, duplicate, resume and adapter checks |
| `experiments/data/boundary_pilot/plan.json` | Schedules,560 reference inputs,five checkpoint hashes and comparison policy |
| `plan.sha256` | Detect accidental edits to the prepared plan |
| `recordings/*.npz` | New clips with human labels and group/task/sitting/capture-session provenance |
| `recording_attempts/*.json` | Completed,paused/failed camera intervals and save counts |

The new folder is isolated from `commands_v2/`; original recordings/checkpoints
are preserved. A temporary overlay adapter adds prompts and is restored on exit;
no existing source file or camera default is edited. Tracking checks do not
establish meaning. Quality failures are retained/reported for review, with no
automatic exclusion or model-driven relabeling. Use existing `.venv` dependencies.
