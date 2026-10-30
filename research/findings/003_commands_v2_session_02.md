# Eight-command models: second validation sitting

> Git-tracked records: [configuration, results and provenance](../results/003_second_session/README.md). Original run/data links below require the local private archives.

Date: 2026-10-08 (America/Chicago).
Original training study: `20261008T025353704748Z`.
Evaluation: `20261008T052213777935Z`.
**Development validation; no new training, final test, or robot outcome.**

## Question and data

Do the five already trained models recognize gestures recorded in a sitting
that did not select their weights? Evaluate all saved candidates unchanged,
using the same eight-class contract, 32×67 landmark features and raw predictions.
The protocol and candidate/input hashes were recorded before predictions.

The new session is `20261008T050854-92265627`: **41 original validation clips**,
five per class except RIGHT, which has six. All pass tracking checks. No exact
raw duplicates or split-crossing sessions were found. The original study's
519 input files remain unchanged. Training remains 472 clips in five sessions;
validation now has 88 clips in two sessions; test remains empty.

Different session identity does not establish independence across people. This
is still a single-user pilot. Keep the new results separate from the original
47-clip session, which selected the saved checkpoints.

## Every saved candidate

| Seed | Old accuracy / 47 | New correct / 41 | New accuracy | New macro F1 | NO_COMMAND → motion / 5 | STOP → motion / 5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 80.85% | 37 | 90.24% | 0.8957 | 2 | 0 |
| 22 | 85.11% | 37 | 90.24% | 0.8984 | 1 | 1 |
| 33 | 87.23% | 34 | 82.93% | 0.8231 | 1 | 1 |
| 44 | 85.11% | 36 | 87.80% | 0.8727 | 1 | 1 |
| 55 | 87.23% | 33 | 80.49% | 0.7986 | 2 | 1 |

Mean new accuracy: **86.34% ± 4.43 percentage points run SD**.
Primary outcome, mean macro F1: **0.8577 ± 0.0447 run SD**.
Mean NLL: **0.4690 ± 0.0970**, compared with old 0.3380; lower is better.
Accuracy changes by seed are +9.39/+5.14/−4.31/+2.70/−6.75 points.
Mean accuracy is 1.24 points higher than the old 85.11%, with unchanged weights.
This is a difference between recording sets, not an improvement caused by
learning. Different class counts and gesture styles prevent a causal attribution.

Seeds 33 and 55 tied for highest old accuracy, yet score below seeds 11 and 22
on this sitting. Do not retrospectively select a winning seed using either
session or silently create an ensemble. Run SD measures training variability
on shared clips, not an uncertainty interval across people. Five models on
41 clips do not give 205 independent human observations.

## Class-specific results

Recall is the fraction of a true class recognized correctly, averaged across
the five models on the same recordings.

| True class | Old mean recall | New clips per model | New mean recall |
| --- | ---: | ---: | ---: |
| FORWARD | 80.00% | 5 | 64.00% |
| BACKWARD | 73.33% | 5 | 96.00% |
| LEFT | 100.00% | 5 | 96.00% |
| RIGHT | 93.33% | 6 | 96.67% |
| STOP | 96.67% | 5 | 80.00% |
| NO_COMMAND | 53.33% | 5 | 72.00% |
| UP | 84.00% | 5 | 84.00% |
| DOWN | 100.00% | 5 | 100.00% |

FORWARD and STOP recall are lower on this sitting. Across repeated model–clip
predictions, nine FORWARD errors go to BACKWARD; one BACKWARD goes to FORWARD.
NO_COMMAND goes to DOWN six times and RIGHT once. STOP goes to UP three times,
FORWARD once, and NO_COMMAND once. UP goes to STOP three times and FORWARD once.
There is also one LEFT→NO_COMMAND and one RIGHT→NO_COMMAND error. These are
repeated predictions on shared recordings, not counts of independent events.

Raw NO_COMMAND→motion mean is **28% ± 10.95 points run SD**, compared with
old 46.67%; STOP mean recall is **80%**, compared with old 96.67%. The different
small clip pools limit interpretation. DOWN's 100% recall does not imply every
DOWN prediction is correct: its mean precision is **80.95%**, because some
non-command clips are predicted as DOWN.

## Shared confident error and human clarification

Exploratory inspection after the primary evaluation found that all five models
classify NO_COMMAND clip `7601353c641440f9bca0b5c65cfed650.npz` as DOWN:

| Seed | Score assigned to DOWN |
| ---: | ---: |
| 11 | 0.9961367 |
| 22 | 0.9999322 |
| 33 | 0.9968060 |
| 44 | 0.9993725 |
| 55 | 0.9999932 |

The user explains that the hand was lowering into a natural resting position;
DOWN normally involves properly pointing downward, while a normal resting
hand is NO_COMMAND. Preserve the NO_COMMAND label. Earlier authorized deliberate
whole-hand DOWN gestures remain supported; this clarification does not redefine
all DOWN examples as pointing-only or justify altering the frozen contract.

The models share architecture, preprocessing and training data. Their agreement
is not an independent semantic check. Tracking success and high scores do not
override the user's intended label. Missing examples of the resting/command
boundary are a hypothesis; we have not established the learned cue or a cause
from these predictions. Stored clips contain landmarks, not replayable RGB.

Across all raw STOP/NO_COMMAND→motion errors, six scores are ≥0.8 and five are
≥0.95; maximum 0.9999932. The ≥0.95 errors are the five predictions of this one
clip. No margin/three-window gate or continuous stream was simulated. These
results do not measure issued motor commands or false starts per minute.

This is a concrete counterexample to using high score or same-family model
consensus as sufficient proof for a pseudo-label. It motivates an audit
comparison; it does not establish that our proposed semi-supervised algorithm
will fix the problem or save annotation effort.

## Next development decision and limits

Before blindly pseudo-labeling or retraining, design a controlled training-data
comparison focused on relaxed resting/repositioning versus deliberate DOWN,
with FORWARD/BACKWARD confusion also recorded. Obtain any new training examples
in separate sessions; retain both validation sessions outside gradient updates.
Count all human labels, review effort and compute. Fix the comparison and
evaluation policy before fitting; do not promise a particular accuracy gain.
No such comparison or new training occurred in this evaluation.

Both validation sessions are now inspected development data. Final evaluation
requires genuinely new sessions after choices freeze; no final freeze exists
yet. Unseen-user accuracy, robotics task success, label savings, stage-order
advantage, novelty and publication readiness remain unestablished.

## Files and verification

Artifacts:
[`20261008T052213777935Z`](../experiments/runs/commands_v2/20261008T025353704748Z/session_evaluations/20261008T052213777935Z/).

- `evaluation_plan.json`: saved checkpoint hashes and no-fitting policy.
- `manifest.json` / `new_validation_features.npz`: the 41 new input identities,
  labels and numeric features, separate from the old validation archive.
- `seed_N/predictions.json` / `metrics.json`: every new guess, score and error.
- `summary.json`, `trials.json`, `report.md`: all seeds and separate old/new
  descriptive comparisons.
- `environment.json` / `source/`: package versions, hashes and execution source.
- `verification.json`: preserved inputs/weights and reproducible predictions.

The separate helper `experiments/evaluate_v2_sessions.py` preserves the original
training source and results. Six meaningful software checks passed, covering
fresh-session eligibility, unchanged old inputs, minimum class counts, source/
checkpoint integrity, keeping tracking failures, separate aggregation and
saved-checkpoint inference without an optimizer. Synthetic checks are software
evidence, not gesture accuracy. Real verification passed: archived hashes,
41 new input files, 205 saved predictions, and reproduction of all 235 original
score vectors. No weights, labels, application model or dependencies changed.
