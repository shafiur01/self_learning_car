# Experiment 008 — recurrent models improve average development scores, with trade-offs

> Git-tracked records: [configuration, results and provenance](../results/008_temporal_architectures/README.md). Original run/data links below require the local private archives.

Completed 2026-10-13 (America/Chicago). UTC study:
`experiments/runs/temporal_comparison/20261014T011524632211Z`.
Protocol: [TEMPORAL_COMPARISON_PROTOCOL](../TEMPORAL_COMPARISON_PROTOCOL.md).
Implementation: `experiments/temporal_comparison.py`; teaching: [Lesson 12](../LESSON_12.md).

## What this tests

Train five architectures on the unchanged v3 538 training / 48 inspected
validation clips, five seeds each. Same 560 Adam updates, .001 learning rate,
replacement batches of 32, class-weighted training loss and minimum unweighted
validation-loss checkpoint selection over 28 checks. No augmentation, extra
labels, family-specific tuning, new test access or camera activation.

CNN retains the existing temporal network. pose_set averages learned local
pose/visibility features without frame order or wrist/size channels; feature_set
does the same with all 67 inputs. Both observe a collection of changing poses,
not one static photograph. LSTM and BiLSTM have comparable parameter counts and
four ordered pooled regions. This is a fixed-recipe inductive-bias comparison,
not a pure order ablation or each family's optimized potential.

## Primary and secondary results

All 25 fits completed, all seeds retained. Values below are five-seed means;
accuracy SD is variation across initializations on the same 48 clips.

| Model | Parameters | Accuracy (seed SD, percentage points) | Macro F1 | STOP recall | NO_COMMAND→motion |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 48,520 | 85.83% (1.74) | .8611 | 66.67% | 13.33% |
| Pose set | 48,792 | 85.83% (1.74) | .8600 | 66.67% | 16.67% |
| Feature set | 49,176 | 85.42% (1.47) | .8568 | 66.67% | 13.33% |
| LSTM | 48,944 | 89.58% (4.89) | .8909 | 76.67% | 33.33% |
| BiLSTM | 49,256 | 88.75% (4.80) | .8869 | 66.67% | 10.00% |

Primary mean macro F1 differences versus CNN: LSTM +.029814 (three of five
paired seeds positive), BiLSTM +.025838 (four of five positive). Individual
differences LSTM +.0780/+.0590/+.0342/−.0156/−.0065; BiLSTM
+.0598/+.0153/+.0609/−.0227/+.0159. Pose set −.001142 and feature set −.004332.
Neither recurrent candidate uniformly dominates the CNN across seeds.

LSTM correct/48 by seed 11/22/33/44/55: 45/45/44/40/41; BiLSTM 44/43/45/39/42.
Do not report the best 93.75% seed as the method's general performance.

| Model | F | B | L | R | STOP | NONE | UP | DOWN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 73.33% | 70.00% | 100% | 100% | 66.67% | 86.67% | 90% | 100% |
| Pose set | 80.00% | 66.67% | 100% | 100% | 66.67% | 83.33% | 90% | 100% |
| Feature set | 73.33% | 70.00% | 100% | 100% | 66.67% | 86.67% | 86.67% | 100% |
| LSTM | 93.33% | 80.00% | 100% | 100% | 76.67% | 66.67% | 100% | 100% |
| BiLSTM | 90.00% | 76.67% | 100% | 96.67% | 66.67% | 90% | 90% | 100% |

Recurrent models improve FORWARD/BACKWARD recall on this sample. That supports
further temporal investigation, but does not identify chronology as the causal
source: architecture, pooling and optimization also differ. The order-invariant
models approximately matching CNN is consistent with useful pose cues; it does
not establish that all gestures can be recognized without time.

## Robotics-relevant unfavorable results

LSTM increases NO_COMMAND→motion from 4/30 to 10/30 repeated predictions; the
six unique NONE clips are each seen by five models. NONE→DOWN rises from 0/30
to 6/30. LSTM improves mean STOP recall but still emits false movement for
7/30 STOP predictions, versus CNN 9/30. BiLSTM reduces NONE→motion to 3/30 but
STOP→motion is 10/30, and the same two STOP clips remain wrong in every seed.
No motors, execution gate, continuous stream or robot task was evaluated.

Post-primary error audit (`post_primary_errors.json`, reproducible helper in the
study) counts raw movement predictions for STOP/NONE: CNN 13, pose_set 15,
feature_set 14, LSTM 17, BiLSTM 13 of 60 repeated STOP/NONE predictions each.
Counts with scores ≥.8: 8/6/4/10/7; ≥.95: 4/2/1/3/2. Highest scores among these
mistakes .99396/.97735/.96658/.99506/.97600. These cutoffs are descriptive counts,
not tuned/evaluated execution gates or probabilities verified as calibrated.

Shared STOP filenames `31f155fc13db4984b9b5108026540f79.npz` and
`906b8197e62b4f2d80809490b0cd8028.npz` still fail all BiLSTM seeds. LSTM has no
clip missed by all five seeds, but its errors vary more across seeds; this is
not an ensemble evaluation or guarantee that voting would repair mistakes.
Human labels and gesture contracts remain unchanged.

## Resources, integrity and scope

Mean fit-loop seconds per model CNN/pose_set/feature_set/LSTM/BiLSTM:
3.85/.44/.41/1.95/2.92. Total fit loops 47.80 seconds; fresh-worker wall time
75.54 seconds, excluding parent preparation/verification/plots. Local median
forward-pass times .061/.038/.033/.211/.333 ms for one preprocessed 32-frame
window. These are CPU measurements with two threads, 40 warmups and 200 timed
passes; acquisition, hand tracking and two-second window delay excluded.
Mean process peak RSS approximately 332.5/329.8/329.8/333.9/336.0 MB includes
Python/Torch/data/optimizer. Background load and fixed condition order limit
universal timing comparisons; it is not isolated model memory.

Nine software checks passed (five new and four imported prior fitter checks).
CNN refits reproduced original initial states, batch streams, chosen updates,
saved state tensors and predictions. All five architectures share the exact
per-seed sampled-index stream. Distinct architectures cannot share identical
initial weights/dropout masks. No claim that all randomness is paired.

Reloaded all 25 models and reproduced all 1,200 saved score vectors, human row
labels, aggregate/per-class metrics and minimum-loss checkpoint selection.
Verified 586 original inputs, archived sources/config/arrays and all new/old
checkpoint hashes. Repeated CLI verification passed. Preserved every model,
history, negative result and derived plot/error-audit source. No real fit failed.

![Experiment 008 results](../experiments/runs/temporal_comparison/20261014T011524632211Z/figure.png)

This is checkpoint-selected validation from one user and six examples per class,
already inspected and used to motivate this comparison. Seed SD is not population
uncertainty; 25×48 predictions are not 1,200 independent human gestures. No claim
of statistical significance, SOTA, novel algorithm, annotation saving or robot
reliability follows. Zero new labels were collected, but the 538/48 existing labels
and their historical costs still count.

Next use the existing development errors to investigate NONE/STOP boundaries
and representation sensitivity, without automatically changing labels. Retain
CNN and both recurrent families as candidates. Additional development sessions
must stay grouped and be counted; genuinely untouched people/sessions are needed
after final choices freeze. Audited label-efficient learning remains a separate
future experiment, with its own matched budgets and strong baselines.
