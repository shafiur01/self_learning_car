# First eight-command supervised training result

> Git-tracked records: [configuration, results and provenance](../results/003_commands_v2/README.md). Original run/data links below require the local private archives.

Date: 2026-10-07 (America/Chicago). Study `20261008T025353704748Z`.
**Development validation only: final test and robot performance remain unevaluated.**

Update 2026-10-08: the second sitting is complete and was evaluated without
changing these weights. Its separate [finding](003_commands_v2_session_02.md)
reports 86.34% mean accuracy on 41 clips. The original results below are preserved.

## Dataset and label correction

The user reviewed the fresh validation clips and explicitly corrected
`e38cd85291d840689e82676107d709f6.npz`: **FORWARD → LEFT**. We changed only
the human label and its revision history. Coordinates, timestamps, visibility,
session, and validation split are unchanged. Exact pre-edit bytes and both file
hashes are preserved under
`experiments/data/commands_v2/label_revisions/20261008T025236017041Z/`.
This was human semantic review before fitting, not a relabeling based on guesses.

Training: **472 clips in five sessions**. Composition: 201 original legacy
recordings, 133 historical corrections, 90 retired/inspected validation clips
explicitly recycled on user instruction, and 48 new UP/DOWN recordings.
All prior acquisition/review costs still count; recycling does not erase them.

Fresh validation: **47 clips in one new session**,
`20261008T024032-d194a121`. All eight labels meet the minimum five/class.
All 47 pass tracking quality, imported copies remain unchanged, and no exact
raw development duplicates or split-crossing sessions were found. Test: zero.

| Label | Training | Fresh validation |
| --- | ---: | ---: |
| FORWARD | 114 | 6 |
| BACKWARD | 75 | 6 |
| LEFT | 68 | 7 |
| RIGHT | 59 | 6 |
| STOP | 57 | 6 |
| NO_COMMAND | 51 | 6 |
| UP | 24 | 5 |
| DOWN | 24 | 5 |

The earlier two-validation-sitting/80-clip suggestion remains incomplete. Before
execution, the protocol records using this one complete session for the first
pilot, with another independent sitting pending. Human label review and numerical
tracking checks do not by themselves establish broadly valid gesture semantics.

## Fixed procedure

Same 67 landmark features and temporal network, with an eight-score output head
(48,520 parameters). Train each seed from random initialization; no fine-tuning,
pretraining, semi-supervised stage, augmentation, ensemble or threshold changes.
Five seeds: 11/22/33/44/55. Each has 560 Adam updates at lr 0.001, full batches
of 32 sampled uniformly with replacement, class-weighted cross entropy, and
28 unweighted-validation-loss checks at steps 20,...,560. Keep lowest-loss
weights separately for each seed. No camera model was activated.

Validation both chooses checkpoints and supplies the following outcomes.
Its scores are development evidence with selection bias, not untouched test.

## Outcomes: all seeds

| Seed | Correct / 47 | Accuracy | Macro F1 | Saved step | NO_COMMAND → motion / 6 | STOP → motion / 6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 38 | 80.85% | 0.8051 | 260 | 4 | 1 |
| 22 | 40 | 85.11% | 0.8576 | 560 | 2 | 0 |
| 33 | 41 | 87.23% | 0.8675 | 280 | 3 | 0 |
| 44 | 40 | 85.11% | 0.8547 | 560 | 3 | 0 |
| 55 | 41 | 87.23% | 0.8811 | 460 | 2 | 0 |

Mean accuracy **85.11% ± 2.61 percentage points run SD**.
Mean macro F1 **0.8532 ± 0.0288 run SD**; mean NLL 0.3380.
Mean training time 3.8458 seconds per seed on this environment. These timings
exclude collection/review, artifact verification and previous training effort.
Model initialization, sampling and dropout vary across seeds; examples stay fixed.
Run spread is not an interval across people or independently recorded sessions.
Five models on 47 clips do not create 235 independent human observations.

## Class-level behavior

Recall measures how many clips of a class receive that class. These are means
over five models on the same small class-specific pools.

| True label | Clips per model | Mean recall |
| --- | ---: | ---: |
| FORWARD | 6 | 80.00% |
| BACKWARD | 6 | 73.33% |
| LEFT | 7 | 100.00% |
| RIGHT | 6 | 93.33% |
| STOP | 6 | 96.67% |
| NO_COMMAND | 6 | 53.33% |
| UP | 5 | 84.00% |
| DOWN | 5 | 100.00% |

LEFT and DOWN being correct on these clips does not establish universal
reliability. UP has four UP→STOP errors across the five models on the same five
clips. BACKWARD→NO_COMMAND occurs eight times and FORWARD→BACKWARD five times
across the same repeated pools. Full confusion matrices/predictions are preserved.

The main current weakness is **NO_COMMAND-to-motion mean 46.67%**. Per-seed
counts are 4/2/3/3/2 out of six; 12 of the 14 repeated model–clip errors go to
BACKWARD, one to UP and one to FORWARD. These are raw classifications, not
issued motor commands or continuous false starts. STOP has one STOP→UP error
across models, in seed 11.

**Exploratory score inspection after training:** one STOP/NO_COMMAND-to-motion
prediction scores ≥0.8 and ≥0.95; the maximum is 0.9829. We did not simulate
margin/three-window gating, tune a threshold, or claim all raw errors would
execute. Confidence alone still does not establish semantic correctness.

## Meaning and limits

An eight-class model is now trained and retained. The broader practical command
interface still has errors, particularly ordinary non-command movements. These
results are not evidence for label savings, a best staged-learning procedure,
novelty, unseen-user accuracy, physical robot reliability, or publication readiness.

The task, training pool and validation recordings differ from the six-class
experiments. Do not subtract the old 70.67% or 93.33% from this score and call
the difference a controlled gain. No real test accuracy exists yet. UP/DOWN are
perception labels; drone descent or humanoid sitting require a separate controller.

Next: a second genuinely new development validation sitting, five/all eight
classes (40), to evaluate these frozen candidate weights without retraining or
reselecting on that new session. Keep its results separate from this 47-clip
checkpoint-selection session. Only after development choices freeze should
genuinely new final test sessions be collected. No final candidate freeze was
declared in this run.

## Verification and artifacts

All five saved checkpoints regenerated their archived 47-clip score vectors;
input/source hashes verified, all 519 development file hashes stayed unchanged,
eight-output/schema/session provenance matched, and the corrected file is LEFT
in the training study's validation manifest. Archived arrays have train shape
(472,32,67) and validation shape (47,32,67). No test payload enters training.
These checks verify preservation and software, not broad recognition reliability.

Artifacts under
[`20261008T025353704748Z`](../experiments/runs/commands_v2/20261008T025353704748Z/):

- `summary.json`, `trials.json`, `status.json`: all five completed results.
- `manifest.json`, `config.json`, `contract.json`, `development_features.npz`:
  precise data, labels, meanings and training configuration.
- `environment.json`, `source/`: exact code, versions and input/source hashes.
- `seed_N/model.pt`, histories, predictions, metrics and session metrics:
  every candidate, including unfavorable outcomes.
- `verification.json`: checkpoint hashes and completed artifact checks.

No old checkpoints were replaced, no dependency was added, no physical camera
was opened in the real training run, and no model was silently activated.
