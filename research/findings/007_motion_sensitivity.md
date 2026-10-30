# Experiment 007 — what motion information affects the fixed v3 models?

> Git-tracked records: [configuration, results and provenance](../results/007_motion_probe/README.md). Original run/data links below require the local private archives.

Executed 2026-10-12 after a focused primary-source/code review. This is an
exploratory diagnostic on the already-inspected 48 development validation clips,
not a new test or architecture improvement. All 5 v3 checkpoints are unchanged.

Study `../experiments/runs/motion_probes/20261012T221029684780Z`, source helper
`../experiments/motion_probe.py`. No new labels, fitting, activation or test access.
Three synthetic checks passed before execution: intervention channel isolation/
input preservation, full-row permutation integrity, raw reversal/timing reanchor.

## Question and prespecified perturbations

The model receives changing landmarks and wrist movement, but architecture alone
does not establish reliance on trajectory order. Fix the saved weights, compare
their predictions on original inputs and specific modified versions, preserve
all results. Center means sample 16 of 32; the fixed shuffle seed is 20261012.

| Input | Changed predictions / 240 | Mean JS divergence from original scores | Original-label agreement |
| --- | ---: | ---: | ---: |
| Original baseline | 0 | 0 | 85.83% |
| Wrist XY displacement set to 0 | 0 | .000112 | 85.83% |
| Log size-change set to 0 | 1 | .000093 | 86.25% |
| Center local shape repeated; trajectory retained | 17 | .028393 | 82.08% |
| Center local shape repeated; wrist/size zero; visibility retained | 17 | .027626 | 82.92% |
| Fixed frame shuffle, complete 67-feature rows together | 0 | .000149 | 85.83% |
| Raw coordinates/visibility reversed, times remapped and features re-extracted | 1 | .000450 | 86.25% |

The 240 predictions are 48 clips seen by 5 models, not 240 independent gestures.
For modified inputs, the original human label may no longer describe the input's
meaning. Those agreement percentages are not newly valid classification accuracy
and cannot support an improvement claim. In particular, reversal is not a
label-preserving augmentation or guaranteed FORWARD↔BACKWARD mapping. Shuffling
destroys smoothness and moves the initial-reference row; it is out of distribution.

## Interpretation

Top-label decisions here were insensitive to removing global wrist XY or this
one frame permutation, but scores were not identical. Maximum single probability
changes: .09079 wrist removal, .08011 size removal, .05751 shuffle, .15418
raw reversal. This demonstrates some numerical reliance/response; it does not
establish useful semantic trajectory reasoning or imply those channels are unused.

Repeating a single local hand shape changes 17 predictions; BACKWARD's agreement
with original labels falls 70%→36.67%. That is consistent with changing articulation
being relevant to this class. It does not isolate temporal ordering from the
set/variation of poses, and the modified static clip may no longer be BACKWARD.

STOP original-label recall stays 66.67% in every probe. The shared STOP errors
are therefore not simply repaired by suppressing the present wrist/size cues or
making the local shape static. This does not establish their cause, wrong labels,
or that more powerful temporal architectures would necessarily help.

The cautious inference is that final decisions on this small, pose-heavy sample
show stronger sensitivity to changing local shapes than to the probed chronology
or global trajectory channels. The new UP command is deliberately held-pose
recognition; weak time-order sensitivity for it is unsurprising and not a failure
to support the task. Dynamic F/B still need dedicated ordering/style evidence.

The diagnostic cannot distinguish redundant cues, robustness, pose shortcuts,
insufficient motion variation or loss of informative content at resampling. A
single shuffle/center frame is a finite prespecified probe, not an invariance
proof over all paths. Do not select a different shuffle until a desired effect
appears or promote artificial-input agreement gains as improved models.

## Integrity and next experiment

Reproduced all 240 original score vectors before perturbation, retained every
variant/seed output and checked weights before/after. Independent archive
verification then reproduced all 1,680 saved diagnostic vectors, metrics, flip
rates, JS values and summary; checked original output/source fingerprints.
Saved `independent_verification.json` and full per-clip guesses. No original
records, weights or meanings changed. The old inspected test was not reopened.

Next compare a pose-only/order-invariant reference, the current temporal CNN,
capacity-matched LSTM/BiLSTM and a justified motion/graph baseline under the same
v3 definitions, data and finite budgets. Retrained feature ablations are needed
to ask whether explicit motion improves learning; deleting channels only after
training cannot answer that causal question. Detailed review and staged research
plan are in [TEMPORAL_ALGORITHM_PLAN](../TEMPORAL_ALGORITHM_PLAN.md).
