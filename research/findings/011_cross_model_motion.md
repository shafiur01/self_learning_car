# Experiment 011 — explicit motion across models and controlled comparisons

> Git-tracked records: [configuration, results and provenance](../results/011_cross_model_motion/README.md). Original run/data links below require the local private archives.

Completed 2026-10-15. Explicit motion inputs improve all five model families on
the existing development cohort. Averaging models also perform strongly, so this
experiment does **not** establish a benefit from recurrent sequence memory.

Study: [`cross_model_motion/20261015T164901242592Z`](../experiments/runs/cross_model_motion/20261015T164901242592Z/REPORT.md).
Predeclared design: [cross-model protocol](../CROSS_MODEL_MOTION_PROTOCOL.md).
Teaching: [Lesson 15](../LESSON_15.md).

## Matched comparison

Use the unchanged 538 training / 48 v3 validation clips, version2 32×146 features,
five seeds (11/22/33/44/55), 560 Adam updates, replacement batch32, class-weighted
training cross-entropy, learning rate .001. Every20 updates select the earliest
strict minimum unweighted validation loss. Models have approximately49k parameters.
No architecture-specific search. Sampling streams match across families;
initialization matches across conditions within a family, not across architectures.

Eighty new core fits extend experiment010 to CNN, BiLSTM, local set and feature
set; its20 exact LSTM states are retained as references. Fifteen further control
fits give115 saved models total. All use the same numerical training recipe.
All original recordings/checkpoints remain intact; none is activated in the app.

Mean development accuracy (%) across all five seeds:

| Model | Camera position | Camera + motion | Hand position | Hand + motion |
| --- | ---: | ---: | ---: | ---: |
| CNN | 89.58 | 99.17 | 85.83 | 87.08 |
| LSTM | 84.17 | 97.92 | 91.67 | 92.08 |
| BiLSTM | 89.17 | 99.58 | 91.67 | 92.50 |
| Local set | 84.58 | 98.75 | 81.67 | 87.08 |
| Feature set | 85.00 | 99.17 | 85.42 | 89.58 |

Camera modes retain camera-oriented palm-relative geometry; hand modes express
local geometry in an anatomical basis but retain camera orientation/global
direction. Both share clocks and validity inputs; position conditions mask
derivatives. Hand alignment is not consistently better. Local set masks global
wrist/size position and velocity; it still receives local motion and is not the
old pose-only baseline.

Primary paired mean camera-motion minus camera-position macroF1:
CNN +.095059; LSTM +.144212; BiLSTM +.104811; local set +.140303;
feature set +.141876. Allfive differences positive in each family. These are
paired development effects, not population significance or fresh replication.

## Errors relevant to a future robot

Camera-motion conditions, allfive seeds:

| Model | Accuracy mean ± seed SD (%) | MacroF1 | STOP recall (%) | NONE → motion (%) |
| --- | ---: | ---: | ---: | ---: |
| CNN | 99.17 ± 1.14 | .9916 | 100 | 0 |
| LSTM | 97.92 ± 2.08 | .9793 | 93.33 | 0 |
| BiLSTM | 99.58 ± .93 | .9958 | 100 | 3.33 |
| Local set | 98.75 ± 1.14 | .9874 | 100 | 0 |
| Feature set | 99.17 ± 1.14 | .9916 | 100 | 0 |

BiLSTM33 misclassifies repositioning NONE
`ee21e3af784b4c648f9a29f2c1528904.npz` as LEFT: raw score.52578 versus NONE.47164.
It has the highest mean accuracy but makes an unwanted-motion error. Do not
declare a universal robot-reliability winner. Scores are not calibrated safety
probabilities. Six unique NONE clips repeated across five models are not30
independent situations. Zero observed errors is not a zero future risk estimate.

Unique clips missed by at least one of five camera-motion seeds (six/class):

| Model | FORWARD | BACKWARD | LEFT | RIGHT | STOP | NONE | UP | DOWN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| LSTM | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 1 |
| BiLSTM | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Local set | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| Feature set | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

These union counts describe review coverage, not one model's error rate.
[`full_clip_errors.csv`](../experiments/runs/cross_model_motion/20261015T164901242592Z/full_clip_errors.csv)
lists all464 misclassification instances across115 models/conditions, with
original clip path, actual/predicted labels, raw score and prediction archive.
The earlier HTML viewer still shows experiment008's25 models; it has not been
silently replaced with these new models.

## Controls and interpretation

Camera-motion means:

| Control | Accuracy (%) | MacroF1 | STOP recall (%) | NONE → motion (%) |
| --- | ---: | ---: | ---: | ---: |
| Causal CNN | 98.33 | .9832 | 96.67 | 0 |
| Feature set, elapsed-time channel removed | 98.75 | .9874 | 100 | 0 |
| LSTM, elapsed-time channel removed | 97.92 | .9790 | 93.33 | 3.33 |

The standard feature set scores99.17%, close to BiLSTM99.58% and above LSTM97.92%.
Removing elapsed time leaves a strong set result98.75%. Thus this cohort does not
show an accuracy advantage from recurrent sequence memory. Preserve this result
even though the initial expectation favored temporal recurrence.

The set network is invariant to permutations of **already computed feature
rows**. Velocities encode ordered neighboring samples, global displacement uses
an initial reference, and sample intervals remain. This is not proof that raw
gesture order is irrelevant, nor a license to reverse/shuffle gestures with the
same label. No explicit phase/acceleration/bone graph mechanism was learned or
identified by this experiment.

Causal convolution layers use only earlier/current samples; the pooled classifier
still combines the whole observed window. Prefix agreement is exploratory,
recomputed independently by .5/1/1.5s, not validated early-action accuracy or
onset latency. No streaming cache, WAIT/defer policy or robot experiment exists.

## Audit every existing cohort without recycling old tests

The user delegated the policy choice to support proper validation without false
claims. We audited historical raw data and kept original performance reports.
No historical test was re-scored. Current48-clip validation supplies development
scores above; it selected checkpoints and has influenced design repeatedly.

Canonical audit:1,854 files,942 unique raw recordings, zero malformed recordings,
zero conflicting labels among canonical copies. Reused versioned files do not
increase independent sample size. One obsolete label-revision archive is excluded.

| Cohort | Files | Common-definition / split-compatible rows | Limitation |
| --- | ---: | ---: | --- |
| Legacy validation | 162 | 72 | 90 recordings are already in current training |
| Legacy six-class test | 52 | 52 | Previously inspected; lacks current UP/DOWN |
| v2 validation | 88 | 78 | 10 old-UP gestures differ from today's meaning |
| Boundary test | 80 | 70 | 10 old-UP; already inspected |
| Current v3 validation | 48 | 48 | Used for checkpoint/design selection |

This compatibility audit is not fresh accuracy validation or independent visual
confirmation of every label. Historical STOP semantics depend on the qualified
human report of mostly open palms. No-hand NONE and tracking-quality flags remain
explicit. An earlier expected count of20 v2 UP was a documentation arithmetic
error; actual metadata has10 and code always classified the actual labels.

All1,854 current file hashes rechecked. All80 boundary-test files/content hashes
match its original evaluation manifest; freeze/access ledgers unchanged. No
complete original per-file hash baseline was located for the52 legacy test
recordings; current numerical validity/snapshot hashes are established, historical
byte-for-byte preservation from first evaluation is not asserted.

## Fresh, frozen next step

All115 candidates and executable sources frozen at
`experiments/data/prospective_motion_011`. Two new physical sittings,48 clips each,
six/class/sitting, current0/96. Human prompted labels, no model guesses during
collection; historical raw/session identities refused. Both sittings must finish
before one joint evaluation of all115 models. No winner seed or refit.

Plan: [prospective session check](../PROSPECTIVE_MOTION_PLAN.md).
Start from project terminal:

```bash
.venv/bin/python -m research.experiments.prospective_motion record --sitting 1
```

SPACE freezes the preceding two seconds; inspect replay, save with the prompted
label key, or x to repeat. After48, q and take a break before sitting2. An
interruption preserves progress; resume that numbered sitting only if it remains
the same physical sitting. Models see no new clips until both sittings finish.

This is a new-session pilot for the current user, not an unseen-person or
factorial speed/view experiment. After evaluation, the new cohort is inspected
development evidence. Broader paper/final evaluation needs separate people and
sessions after later choices are frozen. Continuous robotics requires onset/
offset/non-command periods, early abstention and action/STOP-delay metrics.

## Files, verification and cost

New `cross_model_motion.py` supplies adapters/fitter/strict research checkpoint
loader and verifier. `dataset_validation.py` supplies numerical/lineage auditing
without old-test inference. `prospective_motion.py` freezes candidates, guides new
collection and enforces one joint evaluation. Tests cover parameter/gradient
checks, pooling invariance, causal future exclusion, input immutability, tiny
paired fits, old-role/semantic exclusions, resumable collection, no model-assisted
labels, incomplete-cohort refusal and retirement/tamper detection.

Five comparison checks and six prospective checks passed. Reproduced115 saved
checkpoints,22,080 vectors/metrics/labels, selection minima and pairing;586 fitting
input hashes unchanged. Sources/configs/arrays and exact old reference files
archived. Scientific PNG/PDF inspected. No dependency installation.

95 additional fits total197.36s numerical fit loops;20 references reused without
extra training. That excludes prior fits, development, audits/verification and
human effort. All538 training and48 development labels plus prior reviews remain
real costs; no new labels for this comparison does not demonstrate savings.
The next96 labels and all attempts count separately. Forward times exclude
tracking/acquisition/evidence wait/control; process peak RSS is cumulative, not
isolated per-model memory. Equal updates/near-equal parameters do not establish
architecture-optimal training or equal compute.

No SOTA, novelty, publication readiness, general/robot reliability, semi/self-
supervised gain or label savings established. The main audited label-efficient
learning procedure remains future research. All unfavorable outcomes retained.
