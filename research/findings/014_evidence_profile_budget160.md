# Pilot 014: more labeled support enables selection, without a student advantage

> Git-tracked records: [configuration, results and provenance](../results/014_budget160/README.md). Original run/data links below require the local private archives.

Executed 2026-10-16, America/Chicago. Five fixed seeds, 65 fresh fits. No new
recordings or validation-to-training transfers. [Frozen protocol](../EVIDENCE_PROFILE_BUDGET_PROTOCOL.md),
[complete run report](../experiments/runs/evidence_profile_budget160/20261016T133936682213Z/REPORT.md),
[figure](../experiments/runs/evidence_profile_budget160/20261016T133936682213Z/analysis/figure.png).

## What changed and why

Pilot 013 admitted only one profile pseudo-label across five seeds. This run
tests the unchanged selection rule with more labeled support: 160 revealed
training labels (20 per command), with the remaining 245 labels hidden from the
learner. All 405 original inputs are identical; the same 133 model-selected
corrections remain excluded. The original 80 labels are nested in every seed.
Both whole-session teachers and opposite-group anchor banks now have 80 labels,
10 per command. All three UP styles occur in both teachers for every seed,
without style-based reselection. This changes student, teacher and anchor support
together; it cannot isolate teacher quality from anchor coverage or style support.

The model, masks, selectors, confidence gate 0.95 and optimization settings are
unchanged. Same 48 development recordings choose checkpoints. Same historically
inspected 96 recordings provide a retrospective comparison after all fits freeze.
160 training plus 48 development labels means 208 supervised labels for training
and selection; endpoint labels and shared pretrained MediaPipe count separately.
This is a label-budget simulation, not actual annotation effort saved.

## Results

Means across all five seeds on the same 96 retrospective clips:

| Method | Accuracy (%) | Macro F1 | NO_COMMAND to movement (%) |
| --- | ---: | ---: | ---: |
| Supervised 160 | 86.25 | .8580 | 28.33 |
| Supervised 160, 1680 updates | 86.25 | .8579 | 26.67 |
| Full-teacher confidence | 86.25 | .8579 | 28.33 |
| Native-subset-teacher confidence | 87.71 | .8710 | 25.00 |
| Proposed response profile | 87.29 | .8699 | 25.00 |
| Embedding support | 88.54 | .8814 | 26.67 |
| Count/weight-matched confidence | 87.08 | .8672 | 26.67 |
| Count/weight-matched random | 87.92 | .8736 | 26.67 |
| Full 405-label supervised reference | 93.33 | .9326 | 8.33 |

Primary paired profile minus native-confidence macro F1: **-.001118**. Per-seed
differences: -.020895, 0, +.030056, -.014751, 0. One positive, two negative, two
tied. By sitting: -.006189 / +.003693. No supported advantage over the primary
baseline, embedding support or matched random. The profile numerically exceeds
matched confidence by .002715 macro F1, but this does not establish reliable
selection superiority. Full paired outcomes are archived; no winning seed chosen.

Profile accuracy rises from 74.38% at 80 labels to 87.29% at 160. Supervised-only
also rises from 74.17% to 86.25%; confidence rises from 74.79% to 87.71%. This
cross-budget increase is not evidence for the new algorithm. The unchanged full
405 reference reproduces all five parent weight-state hashes exactly.

## Admission and label correctness

| Seed | Teacher agreement /245 | Confidence accepts | Profile accepts | Profile weight mass |
| --- | ---: | ---: | ---: | ---: |
| 11 | 135 | 3 | 1 | .765 |
| 22 | 167 | 0 | 0 | 0 |
| 33 | 182 | 69 | 32 | 19.444 |
| 44 | 185 | 17 | 4 | 2.786 |
| 55 | 177 | 0 | 0 | 0 |

After all fits/targets froze, recorded-label diagnostics show:

- Profile: 0 errors / 37 admitted seed–clip pairs, involving **35 unique clips**.
- Native confidence: 5 / 89 errors, involving 79 unique clips. Four NO_COMMAND
  clips are predicted DOWN; one FORWARD is predicted BACKWARD.
- Embedding: 0 / 39 errors, involving 35 unique clips.
- Matched confidence: 1 / 37 errors; matched random: 2 / 37 errors.

These pooled counts differ from the report's mean weighted error across nonempty
seeds. Repeated seed–clip pairs are not independent recordings. No observed error
in this selected small pool is not a precision guarantee or a response-specific
advantage: embedding also makes no observed pseudo-label error. Empty precision
is undefined. The 37 profile pairs cover six predicted commands: DOWN7,
BACKWARD12, FORWARD9, STOP4, LEFT3, RIGHT2. **No UP or NO_COMMAND pseudo-label is
accepted.** All eight classes are in supervised training, but pseudo-label
coverage is incomplete and heavily concentrated in seed33.

Interpretation: this budget makes the selection experiment more active, but
cleaner retained labels do not translate into a better student than the main
baseline. Confidence, coverage, class distribution and training influence matter
together. Favoring easy/redundant examples is a hypothesis, not an established
cause. The loss still averages weights over all 245 hidden inputs, including
rejected zero weights. Active/inactive SSL dropout streams differ; use the active
matched controls when interpreting small effects versus supervised-only runs.

## Robotics, costs and integrity

Profile and native confidence misclassify 25% of the 12 unique NO_COMMAND clips
as movement on average across seeds. All eight low-label student variants recognize all 12 STOP
clips in this endpoint; this is not a guarantee about robot stopping. UP recall
for the profile is 66.67%. No continuous early-command or robot action trial.

12 software checks passed before fitting. Verification reproduces all 65 model
checkpoints, 7,440 saved score vectors, five target/selector bundles and 549 raw
feature arrays. Exact input/selection identity, nested teacher labels, source
function deltas, all UP styles and hidden-label refusal are checked. Parent
artifact hashes remain intact; no old evaluator/ledger restarted. Weights are
research-only, not activated in camera. No dependencies or new lesson files.

65 numerical fit loops total 174.89s. Mean standalone charged fit/selector
envelopes: supervised160 1.94s, profile 7.06s, native confidence 7.04s, full405
1.96s. Charge both teachers despite physical reuse; setup/I/O and earlier work
are excluded. The long supervised control matches optimizer updates, not FLOPs
or runtime. No measured memory, inference or total efficiency saving.

## Decision

Preserve both budgets and report the absent main advantage. Do not increase the
budget again or loosen the gate as an unreported rescue. The next candidate
study should directly investigate whether response profiles add information
beyond confidence and embedding at controlled coverage, including why UP and
NO_COMMAND receive no targets. That study is **not declared or executed** here;
any changes require a new fixed protocol before fitting. These retrospective
single-user findings do not establish novelty, broad generalization, actual label
savings, SOTA or publication readiness.
