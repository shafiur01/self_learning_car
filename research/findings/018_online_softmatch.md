# 018 — Online weighting helps selected checkpoints, but the ranking reverses

> Git-tracked records: [configuration, results and provenance](../results/018_online_softmatch/README.md). Original run/data links below require the local private archives.

2026-10-20. Completed the fifteen-fit comparison declared in
[the frozen protocol](../ONLINE_SOFTMATCH_PROTOCOL.md), following finding017.
[Run and machine-readable results](../experiments/runs/online_softmatch/20261020T205543410735Z).

**Decision:** the declared development continuation gate passes, but this is a
checkpoint-sensitive, mixed result. Do not promote an algorithm or claim robust
superiority. Preserve both confidence and soft weighting as comparison baselines.

## What was tested

Each seed uses160 revealed training labels (20 for every command) and245 hidden-
label training inputs from the same405 original recordings. Same five splits,
raw149-input LSTM/49,840 parameters, initialization, labeled batches/dropout,
Adam .001,560 updates and two CPU threads. No405-label warm-start or133 model-
selected corrections. Existing48 development clips select checkpoints; the96
retrospective endpoint was not loaded or scored. All recordings were already
labeled historically, so this simulates label withholding, not actual savings.

Three methods: supervised160, online confidence .95, and a paper-based SoftMatch
adaptation. All have140 supervised-only warm-up updates. Both SSL methods then
take420 joint updates with the same140-step unsupervised ramp and hidden draws.
The pseudo-label pass is deterministic and gradient-free; the learning pass
uses existing classifier dropout .2 on the identical sequence. No geometric or
temporal transformation changes command meaning, but this also means we are
not testing viewpoint invariance or paper image augmentation.

Soft weights use online raw-confidence mean/unbiased variance, uniform alignment
for weights only, and unchanged hard targets, following
[SoftMatch Algorithm1](https://arxiv.org/pdf/2301.10921). Explicit adaptations:
first-batch bootstrap, statistics half-life one expected hidden-pool pass
(m=2^(-32/245)), no parameter EMA, shared warm-up/ramp and classifier-noise views.
The current USB aligned-statistics hook differs from the paper ordering; we
recorded that choice before training. This is not a full published-benchmark
reproduction or a new method.

For a beginner: the student practices on160 known answers and245 clips for
which its training code has no answers. The confidence method accepts only very
confident guesses. Soft weighting gives some less-confident guesses partial
influence. Both can learn from correct guesses and reinforce wrong ones.

## Results: do not hide the checkpoint choice

Values are means over five paired seeds on the same48 inspected development
clips. “Selected” means the first minimum development cross-entropy among28
checks, **not** maximum F1. “Final” is step560 for every run. Macro F1 gives each
command equal weight; accuracy counts all correct clips. NONE→movement is the
fraction of actual NO_COMMAND clips predicted as any movement command.

| Checkpoint | Method | Accuracy | Macro F1 | NONE→movement | STOP recall |
|---|---|---:|---:|---:|---:|
| Selected | Supervised160 | 85.00% | .8456 | 30.00% | 86.67% |
| Selected | Confidence | 86.25% | .8575 | 20.00% | 86.67% |
| Selected | Soft weighting | 87.50% | .8721 | 13.33% | 90.00% |
| Final560 | Supervised160 | 84.17% | .8370 | 20.00% | 86.67% |
| Final560 | Confidence | 89.58% | .8950 | 16.67% | 90.00% |
| Final560 | Soft weighting | 88.33% | .8818 | 13.33% | 83.33% |

Primary selected Soft-minus-confidence F1 +.014560 (1.46 percentage points).
Paired differences for seeds11/22/33/44/55:
-.045688,+.073443,0,+.021212,+.023830. Three positive, one negative, one tie;
NONE→movement improves6.67points, STOP recall improves3.33points. All four
predeclared feasibility conditions pass. Soft selected F1 exceeds supervised
by .026436, but only two seeds improve over supervised, one declines and two tie.

The fixed-final comparison reverses mean F1: Soft-minus-confidence -.013291,
with worse STOP recall by6.67points. Final Soft-minus-confidence seed differences
are approximately +.0042,+.0646,-.1138,-.0435,+.0221: three positive seeds alone
does not imply a positive mean. Do not replace the declared primary result with
the more favorable endpoint or ignore this unfavorable check.

Checkpoint selection is a material part of this training procedure. Soft selected
steps are320/560/100/160/120, confidence560/160/100/160/360. Soft seeds33 and55
select a checkpoint **before pseudo-label training**. Seed55's positive selected
comparison therefore preserves the supervised checkpoint rather than showing a
better classifier learned from pseudo-labels. Cross-entropy penalizes confident
mistakes, so it can prefer an earlier checkpoint with lower classification F1;
the observed reversal establishes selection sensitivity, not its full cause.
No new selection rule was fitted or chosen after these results.

Each development class has only six unique clips. For example, selected
background-to-movement counts are6 versus4 across30 repeated seed–clip cases,
not30 independent people or sessions. Selected STOP counts are26 versus27
correct across the same kind of repeated cases. The modest changes and seed
variation do not establish statistical significance or robot reliability.

## What the saved training trace adds

All ten SSL runs save420x32 weak predictions, target classes, weights and input
row IDs. Hidden labels were opened only after all15 fits/30 checkpoints/traces
froze. Their errors are retrospective diagnostics, never feedback to training.
Both methods now generate pseudo-labels for all eight predicted command classes.

| Online method | Total weight mass | Incorrect-target mass | Incorrect fraction | Correct-UP mass | Actual NONE→movement target mass |
|---|---:|---:|---:|---:|---:|
| Confidence | 57,475.00 | 7,074.00 | 12.31% | 5,524.00 | 1,606.00 |
| Soft weighting | 62,255.97 | 9,283.35 | 14.91% | 5,906.43 | 2,168.46 |

These are sums over repeated training exposures before the ramp coefficient;
coefficient-adjusted versions and ramp/full phases are in the CSV. A weight of
.5 contributes half a unit, not half a recording or an independent observation.
Soft weighting uses more correct-UP mass **and** more wrong background targets.
Its pseudo-label error fraction is higher even where selected development F1
improves. Thus target purity alone cannot rank trained students. These models
also generate different predictions as they learn; this is not a counterfactual
test of one selector on a fixed identical candidate pool or a causal utility proof.

## Cost, verification and artifacts

- Fifteen fresh fits, no retries or sweeps. Fitting loops including development
  checks:48.74 seconds total; fresh-worker subprocess wall time65.39 seconds.
  Review, setup, software checks, plotting and verification are additional costs.
- Five supervised fits10.06s; five confidence19.37s; five soft19.31s. The two SSL
  variants have similar measured cost here; this is not a general speed claim.
  Same560 optimizer updates do not mean equal compute: training forward samples
  total89,600 supervised versus224,000 for each SSL group (2.5x); weak passes
  have no backward pass. Peak worker RSS340.25MiB includes Python/Torch/data.
- Nine preflight software checks pass: unchanged inputs/private RNG, raw targets,
  detached gradients, zero-mask gradients, full-batch loss denominator, statistics
  decay/shift, invalid input rejection and payload allowlist. No test did model
  optimization. One scalar expectation was corrected for float32 underflow before
  fitting; no study settings changed after freezing.
- All five supervised selected states exactly reproduce014; all methods share
  initial/warm-up/labeled draws/dropout streams. All30 checkpoint predictions
  (1,440 vectors),134,400 online targets/weights and hidden draws reproduce.
  An independent scalar audit reproduces67,200 soft weights chronologically;
  independent confusion calculations reproduce all30 checkpoint metrics.
- Training payload logs contain only the matching learner bundle and48 selection
  arrays. All parent014 files remain pinned/unchanged. Sources/config/plan,
  preflight, fit freezes and oracle timestamps are preserved. Existing sources
  and checkpoints are not replaced; no camera model is activated.

Useful artifacts in the run:
[per-seed results](../experiments/runs/online_softmatch/20261020T205543410735Z/development_per_seed.csv),
[per-class results](../experiments/runs/online_softmatch/20261020T205543410735Z/development_class_metrics.csv),
[every failed development case](../experiments/runs/online_softmatch/20261020T205543410735Z/development_failures.csv),
[pseudo-label diagnostics](../experiments/runs/online_softmatch/20261020T205543410735Z/pseudo_label_diagnostics.csv),
[paired-seed figure](../experiments/runs/online_softmatch/20261020T205543410735Z/development_comparison.png)
and its standalone PDF. Figure visually checked. Failures repeat clips across
models/seeds/checkpoint choices; distinct clip counts are in independent_verification.json.

## Research consequence

This establishes a useful online baseline and exposes checkpoint-selection
sensitivity. It does not establish a novel algorithm, state of the art, actual
annotation savings, unseen-user performance, physical robot safety, or that
SoftMatch generally wins/loses. Our modified consistency/statistics recipe needs
separate controls before attributing effects to individual ingredients.

No additional local fit or96 rescoring follows this result. Next justified work
is a public multi-person temporal-gesture benchmark compatibility/split/compute
audit and a frozen checkpoint-evaluation rule before further algorithm selection.
Keep confidence and soft baselines, all negative seeds and final results. Any
new risk/utility mechanism must improve against these controls and relevant prior
work on independent units, not merely improve this repeatedly inspected48.
