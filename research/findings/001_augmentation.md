# Finding 001 — supervised augmentation pilot

> Git-tracked records: [configuration, results and provenance](../results/001_augmentation/README.md). Original run/data links below require the local private archives.

Recorded 2026-10-07. Study: `20261007T050900637997Z`.
Status: all 20 trials completed. This is development evidence, not final test
evidence, and not a demonstrated novel method or reduction of human labeling.

Full local artifacts: [study report](../experiments/runs/20261007T050900637997Z/report.md),
[summary](../experiments/runs/20261007T050900637997Z/summary.json),
[per-trial results](../experiments/runs/20261007T050900637997Z/trials.json).
Those local artifacts are Git-ignored; this written finding persists separately.

## Fixed design

334 training clips, three sessions (201 original, 133 corrections). Validation:
72 clips, one session. Six classes, 48,390-parameter temporal CNN, CPU/two threads,
Adam lr 0.001, class-weighted cross entropy, batches of 32, 80 epochs/trial.
Each trial retains the lowest validation-loss epoch. Five paired seeds:
11, 22, 33, 44, 55. No augmentation on validation. Test predictions not evaluated.
No recordings failed the recorded movement-tracking quality check.

## Results

| Condition | Accuracy mean ± seed SD | Macro F1 mean ± seed SD | Raw NO_COMMAND → motion mean | Training seconds mean |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 96.67% ± 1.58 percentage points | 0.9647 ± 0.0187 | 9.09% | 6.19 |
| Coordinate jitter | 96.39% ± 0.76 points | 0.9623 ± 0.0075 | 7.27% | 6.38 |
| Time warp | 96.11% ± 1.16 points | 0.9586 ± 0.0129 | 14.55% | 6.36 |
| Combined | 96.67% ± 1.58 points | 0.9631 ± 0.0177 | 12.73% | 6.51 |

Raw NO_COMMAND rates reuse the same 11 clips across seeds; these are not live
command events, and their denominator is not 55 independent recordings. STOP
recall was 100% for all conditions on the same ten STOP clips; this does not
establish reliable STOP detection under unseen conditions.

| Condition | Seed 11 accuracy | Seed 22 | Seed 33 | Seed 44 | Seed 55 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 97.22% | 97.22% | 95.83% | 94.44% | 98.61% |
| Jitter | 97.22% | 95.83% | 95.83% | 95.83% | 97.22% |
| Time warp | 95.83% | 97.22% | 95.83% | 94.44% | 97.22% |
| Combined | 97.22% | 97.22% | 98.61% | 94.44% | 95.83% |

## Interpretation and decision

The augmentation hypothesis did not produce a mean accuracy or macro F1 gain
under these initial settings. Combined augmentation improves one seed and hurts
another by the same two-clip amount. Selecting only its 98.61% run would conceal
that variability. Keep the unaugmented baseline as the primary reference.

Jitter has smaller seed spread and a lower mean raw NO_COMMAND-to-movement rate
here, but differences involve very few clips. Treat this as a question for an
independent comparison, not a validated reliability advantage. Time warp alone
has no accuracy wins and a higher mean raw false-motion rate here; do not make
this setting the default on this evidence.

The current baseline's score cannot be interpreted as an improvement over the
historical 93.1% validation result: both the seeds and training dataset differ
(334 clips now versus 331 in the correction retrain). No new test comparison was
run. Checkpoint/condition selection on the same validation session can inflate
apparent performance.

Next: collect at least one new development/validation session in a separate
sitting, with five independent clips per class as a small first addition, and
evaluate the frozen candidates by session. More sessions/people are needed for
broader claims. Preserve separate new final-test sessions for later. Then define
a session-aware label-budget comparison using the stronger supervised reference
before adding pseudo-labeling. No extra labels were acquired by experiment 001,
but every current training label was used; it has not measured label savings.

## Reproduction identity

- Configuration SHA-256: `8786bb2979f8eaf891f6b4b6c0e1075dae9b5626c94dbf0c1d97b7e2e971e95c`
- Development manifest SHA-256: `6983dd295504092ac898fa168adaeca4f7fe5a3e33349ebb09903b98df3959eb`
- Frozen feature archive SHA-256: `52e6ff4d5ab4273465c8dd9e9c742d1a793ab534b994cd4f157123c4c9ef8efe`
- Runner SHA-256: `6a5ba8db327ad4c48657faaeff0e9307d561b96a32b56fbbccb5626c54b42b9a`
- Protocol SHA-256: `5a0f9c88ec2d33f61e889c80ee6bed8db07d6b6fc3e66ade0963d98aef7afac2`
- Python 3.14.7, Torch 2.14.0, NumPy 2.5.3, macOS 26.6.2 arm64.

Git HEAD at execution: `1aa67d04df9506d10c796568f4d450456d62ad54`.
Workspace included uncommitted source; use archived `source/` and the environment
hashes, not Git HEAD alone, for the actual experiment implementation. Software
verification: 11 new checks passed, including baseline equivalence and complete
artifact/checkpoint loading; `git diff --check` passed.
