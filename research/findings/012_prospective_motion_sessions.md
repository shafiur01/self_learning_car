# Experiment 012 — frozen models on two new recording sittings

> Git-tracked records: [configuration, results and provenance](../results/012_later_sessions/README.md). Original run/data links below require the local private archives.

Completed 2026-10-15 (America/Chicago). All96 new clips were evaluated jointly
with all115 predeclared checkpoints, without retraining or selecting a seed.
The large development-set motion advantage did **not** replicate consistently.
UP is the main weakness; STOP is correct on the12 examples in this cohort.

[Declared plan](../PROSPECTIVE_MOTION_PLAN.md) · [Lesson16](../LESSON_16.md) ·
[Scientific figure](../experiments/data/prospective_motion_011/analysis/figure.png) ·
[All model/sitting metrics](../experiments/data/prospective_motion_011/analysis/model_results.csv).

## Data, access and scope

Two48-clip sittings, six/class/sitting, same participant, one evening. Capture
sessions `20261015T230148-8e32d742` / `20261015T232301-711ada77`, separated by
about3.07minutes between completed camera intervals. These are not different
days, unseen participants or independent population replicates. Style prompts
match the existing six v3 styles/class, class order reversed in sitting2.
Natural variation is not a controlled speed/view factorial.

The115 models, exact source chain/settings, input projections and schedule were
frozen before collection. No predictions shown while recording, no adaptation,
seed/threshold/ensemble selection or partial-sitting performance inspection.
All96 complete before the access ledger and first raw evaluation payload access.
Joint evaluation completed at23:36:44UTC. All96 remain outside gradients.

All96 raw shapes/times/visibility/content identities passed the frozen loader;
no identity/session overlap with historical data. Re-extracted all384 version2
feature arrays identically.95 tracking-quality flags pass; the one all-no-hand
NONE clip is valid under the existing policy and retained. No feature clipping
in these96. Numerical validity is not independent semantic label verification.

The cohort is **now inspected development material**. Future decisions may use
its failures, but it cannot subsequently become an untouched final benchmark.
Original evaluation, access ledger, freeze, recordings and checkpoints preserved.

## All four representations

Mean full-clip accuracy (%), allfive seeds per cell,96 clips per seed. These are
averages of separate models, not ensemble predictions. Original538 labeled
training clips and all fitting choices unchanged.

| Model | Camera position | Camera + motion | Hand position | Hand + motion |
| --- | ---: | ---: | ---: | ---: |
| CNN | 94.17 | 93.33 | 93.54 | 89.17 |
| LSTM | 90.21 | 94.58 | 95.62 | 91.88 |
| BiLSTM | 96.25 | 95.83 | 93.33 | 91.46 |
| Local set | 94.17 | 95.00 | 94.58 | 91.67 |
| Feature set | 94.17 | 92.92 | 95.83 | 90.00 |

Predeclared primary paired camera-motion minus camera-position macroF1:

| Family | Mean difference | Positive seed pairs /5 |
| --- | ---: | ---: |
| CNN | −.011623 | 2 |
| LSTM | +.046892 | 3 |
| BiLSTM | −.007715 | 2 |
| Local set | +.006571 | 3 |
| Feature set | −.016830 | 1 |

Unlike the earlier48-clip development result, motion is not uniformly beneficial.
Hand-motion accuracy is lower than hand-position accuracy in allfive families.
No population p-values or universal architecture/representation winner declared.

## New-session performance and robot-relevant errors

Camera-motion conditions only; SD is across seeds on the same96 clips, not
uncertainty across users or future environments.

| Model | Accuracy mean ± seed SD (%) | Sitting1 (%) | Sitting2 (%) | UP recall (%) | NONE → motion (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 93.33 ± .93 | 96.25 | 90.42 | 65.00 | 3.33 |
| LSTM | 94.58 ± .87 | 97.08 | 92.08 | 70.00 | 6.67 |
| BiLSTM | 95.83 ± 2.66 | 96.67 | 95.00 | 73.33 | 5.00 |
| Local set | 95.00 ± 1.71 | 96.67 | 93.33 | 76.67 | 5.00 |
| Feature set | 92.92 ± 2.13 | 95.42 | 90.42 | 65.00 | 5.00 |

Motion improves mean accuracy over its matched position condition in sitting1
and reduces it in sitting2 for every family. This session interaction matters,
but does not identify a cause. Class-order/gesture execution/tracking/coverage
may differ; these were not independently randomized factors. The new cohorts
also differ from the old validation, so the old-to-new gap alone is not a causal
measurement of overfitting or a particular viewpoint effect.

All115 candidates predict all12 STOP clips correctly (1,380 repeated guesses
on12 unique clips). This is favorable evidence for these examples, not proof
that STOP is solved or that robot stopping is reliable. NONE→motion remains
nonzero in every camera-motion family. Twelve unique NONE clips repeated five
times are not60 independent situations. Raw scores are not calibrated confidence.

## Predeclared controls

| Camera-motion control | Accuracy (%) | MacroF1 | NONE → motion (%) |
| --- | ---: | ---: | ---: |
| Causal CNN | 91.46 | .9080 | 1.67 |
| Feature set, elapsed removed | 93.54 | .9312 | 3.33 |
| LSTM, elapsed removed | 95.21 | .9505 | 5.00 |

The elapsed-masked LSTM is above the matching set baseline here, but this small
two-sitting result does not isolate useful long-range memory. Architectures
have different inductive biases/optimization; both receive precomputed velocities,
initial-reference displacement and intervals. Full-window causal convolution
does not establish early recognition, streaming latency or a robot-action benefit.
No new-cohort prefix accuracy or early exit was evaluated in this declared check.

## Failures available for human review

Across any of115 candidates,46/96 unique clips have at least one error;730
repeated misclassification instances. This **union is not one model's error rate**.

| Original label | Unique clips missed by any candidate /12 |
| --- | ---: |
| FORWARD | 8 |
| BACKWARD | 6 |
| LEFT | 5 |
| RIGHT | 4 |
| STOP | 0 |
| NO_COMMAND | 11 |
| UP | 11 |
| DOWN | 1 |

Open the separate, new viewer from the project terminal:

```bash
open research/experiments/prospective_motion_review.html
```

It includes all96 clips, all23 model conditions/five seeds,89 nearby training
references chosen by normalized shape profile, original timestamps/mirroring,
slow playback/scrub/centered view, successful controls and separate observations.
No RGB video exists. Shape-neighbor proximity is descriptive, not label truth,
a motion similarity score or a causal explanation. Guesses start hidden. Inspect
NONE and UP; save observations and export `gesture_review_notes.json`. Do not
change labels solely because many models agree. The earlier25-model viewer is
unchanged. Human review of this new bundle is pending.

Priority examples:

- `6851961996eb4d1db1ed2d68e5819aae.npz`, sitting1 NONE: prompt says hands out
  of view, yet the tracker reports a hand throughout.13/25 core camera-motion
  models err (12UP,1LEFT), maximum wrong score.9473. Actual gesture, prompt/replay
  consistency and possible tracking artifacts need human inspection. Label retained.
- `37729d949e6f4148a24161488242667d.npz`, sitting2 index-UP:20/25 core
  camera-motion models err, including RIGHT/STOP/FORWARD/NONE.
- `38b0a34c67de466580f1ea021411c895.npz`, sitting2 thumbs-UP:15/25 core
  camera-motion models err; maximum wrong score.9973. High score does not
  establish correctness or label error.
- `771318df0da64ac9b33742db6a10c501.npz`, sitting2 all-no-hand NONE: retained
  without fabricated shape neighbors; some non-camera-motion conditions fail.

Post-hoc UP-style breakdown is saved separately in `up_style_posthoc.csv`.
Each style has only four unique clips across the two sittings, not20 independent
examples. Index-UP recall ranges30–70% across core camera-motion families;
two-finger70–95%; thumb45–80%. This describes a coverage hypothesis, not evidence
to remove a style, relabel failures or declare a mechanism.

## Verification, costs and next controlled decision

Original joint evaluation reloaded115 frozen checkpoints and reproduced11,040
saved score vectors/labels/pooled+sitting metrics. Independent raw feature audit
reproduced384 arrays and quality/content identities. New derived analysis does
not modify frozen executable sources or rerun the one-time evaluator.

Nineteen real Chrome UI checks pass; screenshot/PNG/PDF inspected. Synthetic
browser observations are QA, not human feedback. Browser used a temporary isolated
profile; nonfatal macOS display/allocator diagnostics preserved. A derived-report
attempt failed at a regex replacement of Unicode names; partial artifacts/source
preserved under `analysis_attempt_20261015T234014630154Z`. Fixed derived helper,
regenerated summaries/viewer only, no model evaluation restart or score changes.

New collection cost:96 human labels and1,874.8065s (31.25min) total completed
camera/replay/preparation/pause intervals. Not pure annotation; break/offline
effort, earlier labels/review/training/pretraining and software work remain extra.
New fits0. This pilot does not measure label savings. Review effort will count
separately when real observations arrive.

Next: human reviews existing failures before selecting **one** declared coverage/
representation intervention. Keep this cohort inspected, freeze the intervention
and matched training budget before fitting, and retain new people/sessions for
later evaluation. Do not simply add these96 to training and then report their
resubstitution accuracy as progress. No additional collection or model activation
is needed to inspect current failures. The main audited label-efficient learning
algorithm still needs a separate label-budget/unlabeled-pool design and baselines.

We have not established SOTA, novelty, publication readiness, unseen-user or
cross-day robustness, calibrated robot reliability, early-action performance,
semi/self-supervised gains or reduced labeling/training cost.
