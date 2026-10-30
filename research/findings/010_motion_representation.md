# Experiment 010 — explicit motion helps this development set; hand alignment is a trade-off

> Git-tracked records: [configuration, results and provenance](../results/010_motion_representation/README.md). Original run/data links below require the local private archives.

Completed 2026-10-15, America/Chicago. Study:
`experiments/runs/motion_representation/20261015T051753816210Z`.
Predeclared [protocol](../MOTION_REPRESENTATION_PROTOCOL.md);
beginner explanation [Lesson 14](../LESSON_14.md).

## Question and controls

User reviewed failing clips and requested movement/relative-shape/speed learning,
a hand reference preserving meaningful orientation, and earlier recognition.
Narrative guidance was available in chat. During fitting, the named Documents/
comments folder contained no visible files. Afterwards, a scoped filename search
found nine cumulative exports directly in Documents. They were validated and
imported separately: eight unique clips all confirm original labels, with one
note revision. No labels changed or refit followed. See
[review reconciliation](../notes/2026-10-15_review_reconciliation.md).

Train four camera/hand coordinate × position/explicit-motion conditions with
the same new 146-feature LSTM, 47 hidden units, four pooled bins, 49,276 weights.
538 original training and 48 inspected development-validation clips; seeds
11/22/33/44/55; 560 Adam updates at .001; replacement batch32; weighted training
CE/unweighted validation CE every20 updates; earliest strict minimum checkpoint.
All initial states and sampled-index streams match by seed across conditions.
No extra augmentation, prefix selection, architecture tuning or winning seed.

All conditions use wrist-relative shape normalized by estimated 3D palm length,
camera global XY/size trajectory, visibility, palm-basis orientation/validity and
elapsed/sample seconds. Hand conditions express shape in that anatomical basis;
camera conditions retain camera-coordinate shape. Motion conditions additionally
provide 66 per-second differences with a validity mask; position conditions zero
those slots. Total parameter count is identical, but zero inputs change effective
capacity and motion information; this is not a claim of identical informative
parameters. The older 67-feature/62-hidden LSTM differs and is context only.

Basis orientation allows camera-relative pose reconstruction before clipping;
hand alignment does not discard camera direction. No-hand geometry is zero/masked,
time remains. Derivatives cannot cross raw invisible frames/gaps/invalid bases.
Estimated z and palm units are not calibrated physical depth/metres per second.

## Full-clip results: all five seeds retained

| Representation | Mean accuracy (seed SD, pp) | Macro F1 | STOP recall | NONE→motion | STOP→motion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Camera / position | 84.17% (5.23) | .8351 | 73.33% | 36.67% | 26.67% |
| Camera / motion | 97.92% (2.08) | .9793 | 93.33% | 0% | 3.33% |
| Hand / position | 91.67% (3.90) | .9153 | 86.67% | 13.33% | 10% |
| Hand / motion | 92.08% (4.97) | .9214 | 96.67% | 6.67% | 0% |

Prespecified primary camera-motion minus camera-position mean paired macro F1
+.144212, positive in all five seeds: +.222009/+.150579/+.200298/+.083442/+.064735.
Accuracy increases 13.75 percentage points in this matched comparison. This
supports explicit derivatives as useful input under this recipe on these clips.
It does not isolate true chronology reasoning, useful real-speed generalization,
or the relative importance of finger, wrist and size derivatives.

Secondary hand-motion minus hand-position F1 +.006099 (two/five positive);
hand-position minus camera-position +.080235 (five/five positive);
hand-motion minus camera-motion −.057879 (all five negative).
Thus hand alignment helps the position condition here, but is not an across-the-
board improvement once motion is provided. Hypotheses include basis noise,
coordinate factorization/optimization and data coverage; cause not established.

Camera-motion seed correct/48: 48/46/48/46/47. These two 100% seeds are not the
method's general performance and are not selected for activation. Per-class mean
recalls F/B/L/R/S/N/U/D: 100/100/100/100/93.33/100/93.33/96.67%.
Hand-motion recalls: 83.33/93.33/100/83.33/96.67/93.33/90/96.67%.

The previous relaxed-repositioning NONE→LEFT error is correct in all five new
camera-motion models; all 30 repeated NONE predictions are correct. These are
six unique NONE recordings, not evidence of a zero population false-command rate.

## Remaining errors and robotics trade-offs

Camera-motion has five repeated errors on four unique recordings:

| File | Seed | Recorded label → prediction | Top score |
| --- | ---: | --- | ---: |
| `906b8197e62b4f2d80809490b0cd8028.npz` | 22 | STOP→BACKWARD | .513 |
| Same STOP clip | 44 | STOP→NO_COMMAND | .552 |
| `bda917a4c0f148ab876d11514343287d.npz` | 22 | UP→LEFT | .637 |
| `e52aff04f3054ba7baec65477caf14ef.npz` | 44 | DOWN→NO_COMMAND | .566 |
| `6beb1ebaeec64430a217834cf81d255f.npz` | 55 | UP→NO_COMMAND | .599 |

The other shared STOP clip31f155fc13db4984b9b5108026540f79 is correct in all five.
Hand-motion emits no movement for STOP in these runs, but has two no-hand NONE
false movements, several wrong directions and lower total recognition. These
trade-offs prohibit declaring one universally reliable controller interface.
Raw scores are not calibrated correctness probabilities, and low-score errors
do not establish that an execution gate solves them without coverage loss.
No live gate, continuous event detector, robot task or hardware was evaluated.

## Prefix experiment: use only available observations

Weights remain fixed after full-clip training. Re-extract each prefix using only
raw samples captured by .5/1/1.5 seconds from recorded start, then predict.
Mean actual observations .466/.967/1.466 seconds, with 8/15–16/23 raw frames.
No full-clip interpolation, basis, normalization or suffix leaks into features.

| Representation | By .5s | By 1.0s | By 1.5s | Complete clip |
| --- | ---: | ---: | ---: | ---: |
| Camera / position | 83.33% | 84.58% | 84.58% | 84.17% |
| Camera / motion | 92.08% | 96.25% | 98.33% | 97.92% |
| Hand / position | 91.25% | 92.92% | 91.67% | 91.67% |
| Hand / motion | 88.75% | 92.50% | 92.08% | 92.08% |

Prefix numbers are agreement with the original full-clip intent, not validated
early-command accuracy. Some early observations can be inherently ambiguous;
models were trained on full clips and earlier inputs are distribution-shifted.
Recording starts are not annotated gesture onsets, so this is not STOP response
latency from onset. One repeated prediction accounts for the camera-motion
1.5s/full difference. Do not choose an early-exit deadline/threshold from this
table and then claim an untouched evaluation. Continuous onset/offset data,
non-gesture intervals and explicit defer/coverage metrics are future work.

## Integrity, cost and interpretation

Five mathematical/tiny-fit checks passed .631s: translation/scale, reconstructible
orientation, real-time stretch/velocity, gap masking, no-hand/degenerate basis,
prefix isolation, paired fits/gradients and strict version loading.
All 20 real fits completed; no failed runs. Reloaded and reproduced 3,840 full/
prefix vectors, all metrics/summary and lowest-loss checkpoint provenance.
Re-extracted 2,920 feature arrays from raw recordings and matched every archived
feature/audit. All 586 input hashes and prior 25 checkpoints unchanged.

No feature clipping occurred; the only invalid geometry is the one legitimate
no-hand validation recording, 32 samples per representation. Training clip
durations span 1.725–2.000 seconds; that is not a designed slow/fast gesture study.
Motion timings depend on the temporal pattern within each recording as well.

Total fit loops 34.93 seconds. Mean fit seconds 1.745/1.748/1.746/1.747; local
single-window forward times .187/.184/.186/.188 ms, two CPU threads, 40 warmups/
200 passes. These exclude landmark extraction, feature work, acquisition, waiting
for gesture evidence and controller response. Fast neural inference does not
mean a command can be recognized .184 ms after the human starts moving.
Zero new labels acquired; past 538/48 labels, reviewed validation and upstream
pretraining remain costs. No annotation-saving conclusion.
The subsequently imported feedback adds eight unique validation confirmations
and one revised note; nine exports are cumulative and must not be counted as43
independent labels. Their maximum reported cumulative viewer counter is44.915
minutes, including thought/pauses, not pure annotation time. Separate
`review_cost_addendum.json` records this without rewriting the fitting archive.

![Full and prefix development results](../experiments/runs/motion_representation/20261015T051753816210Z/figure.png)

These results are promising single-user development evidence, on validation
already used for checkpoint selection, error review and experiment design.
They cannot establish statistical significance across people, broad viewpoint/
speed robustness, novelty, SOTA, label efficiency or real robot reliability.
The model may still exploit shortcuts; explicit differences themselves already
encode local temporal direction. A future order-invariant motion comparator and
semantically valid timing/order challenges can test stronger temporal claims.

Next freeze candidate definitions before genuinely new session/user/speed/view
evaluation; keep final participants/sessions untouched. Do not automatically
activate research weights. Stronger representation baselines support, but do not
replace, the separate audited label-efficient learning question. Explicit geometry/
motion and viewpoint regulation have prior art: [DD-Net](https://arxiv.org/pdf/1907.09658)
and [view-adaptive RNNs](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.html).
