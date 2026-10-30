# 018 — Online pseudo-label weighting: controlled small-data adaptation

Declared 2026-10-20 before any new fit, following finding017. User authorizes
training as needed; retain the fifteen-fit cap. This is a published-method
adaptation/control experiment, not a proposed invention or paper reproduction.

## Question and resources

Does online soft weighting improve the use of hidden-label training inputs
compared with a .95 online confidence gate, without worsening background-to-
movement or STOP recall on our development clips?

Use the exact five160/245 splits from pilot014, seeds11/22/33/44/55. All eight
commands have20 revealed labels each. Original405 inputs only; exclude the133
model-selected corrections. Pretrained MediaPipe resources shared. No405-label
warm-start, no new labels, and no selection/endpoint inputs in gradient updates.
48 inspected development clips select checkpoints; the96 inspected endpoint
is **not loaded or scored** in018. This is retrospective development research.
Any label savings are simulated withholding;160+48selection labels are208,
not a complete historical annotation/pretraining cost account.

Exactly three methods x five seeds = fifteen fresh fits. Keep the raw149-input,
32-frame,49,840-parameter LSTM, Adam .001,560updates,batch32,two CPU threads.
No learning-rate, threshold, label-budget, view or EMA sweep. Same initial
weights(seed+300), labeled draws and labeled dropout stream in all methods.

## Common view and optimization contract

Raw146 features and observed availability bits are copied identically in all
passes, preserving timestamps, spatial directions, finger relationships,
velocities and tracking validity. No geometric or temporal augmentation, channel
deletion, viewpoint simulation or synthetic new clip. Weak pseudo-label pass:
current online model, eval mode, no gradient. Strong consistency pass: same
inputs, model train mode, existing classifier-head dropout .2. There is no LSTM
dropout, batch normalization or changing sensor input. This is classifier-noise
consistency and is weaker than paper weak/strong image augmentation; do not
claim image-domain FixMatch/SoftMatch reproduction or viewpoint robustness.

All methods optimize supervised CE on160 labels throughout560 updates. First140
updates supervised only (one quarter of the existing budget/ramp scale), then
420 joint updates for both SSL variants; ramp unsupervised coefficient from1/140
at step141 to1 at280. Supervised warm-up follows the general Dash motivation
and reduces arbitrary early pseudo-label feedback; its length is fixed, untested
as optimal. New warm-up/ramp choices are identical for both SSL methods.

Sample32 labeled and32 hidden inputs with replacement per joint update. Isolate
labeled and consistency dropout RNG streams so extra SSL passes never shift
labeled dropout or labeled draw streams. Weak deterministic logits detach.
Each update uses CE_l + ramp * mean_over_all32(weight * CE_strong(hard_target)).
Do not normalize by accepted count/weight mass. Execute the strong pass even for
a zero mask, count that compute, and check finite losses/gradients.

1. supervised160: supervised CE only.
2. online_confidence: target raw weak argmax, weight1 if raw confidence>=.95 else0.
3. online_softmatch: same target; uniform alignment changes only the confidence
   used for weights. Use truncated Gaussian weights with variance/4 (n_sigma2).
   Online mean/unbiased variance are estimated from **raw** weak maximum
   probabilities, following paper Algorithm1; marginal class probabilities
   update from raw predictions. Update statistics before computing weights.

Reference: [SoftMatch, ICLR2023](https://arxiv.org/pdf/2301.10921), equations5–9,
Algorithm1 and Section4.1. The current
[USB implementation](https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/softmatch/softmatch.py)
passes aligned probabilities to its weighting-hook update, whereas the paper
Algorithm1 updates mean/variance from raw p before UA. We choose the paper
ordering explicitly; this differs from that current implementation. Hard targets
are unaligned in both. No claim of bitwise parity with the external package.

## Statistics timescale and preflight

Do not copy m=.999 into a560-update schedule unexamined. Fix an interpretable
half-life of **one expected hidden-pool pass**: m=2^(-32/245), about .9134 per
joint batch. Mean/variance/marginal bootstrap from the first post-warm-up batch,
then use this common m. This is an explicit small-data adaptation, not a new
algorithm, a tuned optimum, or a claim to match the paper's longer-memory model.
Variance floor1e-12 prevents division by zero; alignment denominator floor1e-12;
neither is tuned. Clamp exponent with min(confidence-mean,0), weights<=1.

Before fitting, test closed-form exponential decay, synthetic confidence-shift
traces, raw-target invariance under alignment, detached gradients, weighted-loss
normalization, allowed file inputs, identical view bytes and deterministic weak
versus stochastic strong passes. Tests do no optimization. Record preflight
traces and hashes before training. Parameter EMA is omitted for **all** methods;
evaluate current model states, avoiding another smoothing/selection choice.

## Freezes, selection and evaluation

Copy/hash source, plan, configuration and parent files before fits. Training
NumPy payload allowlist: five learner bundles and selection.npz only. No oracle,
96endpoint, reference405, rawrecording or saved endpoint probabilities. Each fit
in a fresh process records peakRSS, fit/worker elapsed time, sample-forward counts,
initial/warm-up/label-draw/dropout states and supervised/unsupervised losses.
Save every joint weak prediction, target, weight and rowID for later audits.

Check development CE every20updates (28checks), first minimum wins, common to all.
Save both CE-selected and fixed-final560 checkpoints and development predictions.
Selected states may predate SSL; report that explicitly. Freeze all thirty
checkpoints/traces before separate hidden-training-oracle error diagnostics.
Compare allfive new supervised selected states with saved014 counterparts;
shared first140 trajectories must match across three methods. Any mismatch is
an audit failure to investigate, not an additional hyperparameter search.

Primary endpoint: mean48-development-selected macroF1 softmatch minus confidence,
allfive paired differences. Continuation gate inherited from017: positive mean,
at least3/5positive differences, no increase in mean NO_COMMAND-to-movement rate
and no decrease in mean STOP recall. Feasibility only, not significance. Also
report comparison against supervised160; beating a worse SSL method alone is
not improvement over supervised learning. Fixed-final results are mandatory
secondary checks, not a route to select a more favorable claim.

Report every seed, per-class confusion/recall, selected step, label coverage,
training pseudo-target error by actual/predicted class and phase, all costs and
failed gates. Logged per-update exposures repeat clips; they are not independent
samples or annotation savings. Oracle analysis is retrospective and cannot
change weights, checkpoints, settings or this gate. If the gate fails, stop this
comparison and record it; no automatic rescue runs. Do not score96 to find a win.
No deployment/checkpoint activation or more personal recordings.
