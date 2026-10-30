# 017 — Literature-grounded decision and frozen-output diagnostic

> Git-tracked records: [configuration, results and provenance](../results/017_selector_replay/README.md). Original run/data links below require the local private archives.

2026-10-17. Main question: improve learning from a fixed small label budget for
temporal robot-command gestures, while measuring erroneous movement commands.
This is a focused primary-source review, not an exhaustive novelty search.
Published image/planning results motivate tests; they do not predict our gains.

## What our evidence supports

- 014: response-profile selection did not beat confidence on mean macro F1;
  fewer incorrect pseudo-labels did not produce a demonstrated student benefit.
- 015: 61 correct UP teacher agreements all failed the .95 gate. NO_COMMAND
  was different: the ensemble was correct on only 25/105 hidden-training
  seed–clip pairs. All 37 admitted profile targets were already recognized
  correctly by the supervised student. These are retrospective diagnostics.
- 016: normalization moved accuracy from 86.25% to 87.08%, but NO_COMMAND
  falsely predicted as movement rose from 28.33% to 43.33%. Do not promote it.
  Earlier feature-coordinate comparisons did not isolate camera-viewpoint gains.

Inference: distinguish **admission starvation**, **wrong targets**, and
**redundant targets**. A selector can address the first without solving the other
two. Further architecture/normalization sweeps do not directly test our main idea.

## Relevant published evidence and its limits

“Method inspected” below means the accessible method/assumption sections were
read, not that every proof, appendix or experiment was independently reproduced.

| Primary source | Evidence relevant to our decision | Access and decision |
|---|---|---|
| [Learning to Reweight Examples for Robust Deep Learning, ICML 2018](https://proceedings.mlr.press/v80/ren18a/ren18a.pdf) | Section 3 measures example value through its effect on a small clean guide set, using gradient-based reweighting. Section 3.3 estimates roughly 3× ordinary training time for its implementation. | Method inspected. “Learning usefulness” is established territory. Any guide labels must come from our counted training budget; our 48/96 must not become gradient supervision. Defer this more expensive route until simple baselines are credible. |
| [Dash, ICML 2021](https://proceedings.mlr.press/v139/xu21e/xu21e.pdf) | Algorithm 1 uses supervised warm-up and an evolving loss threshold. Table 1 reports CIFAR-100/400-label error 44.76±0.96% versus FixMatch-RA 48.85±1.75%, under that paper's setup. Its convergence result assumes smoothness, bounded gradients, a PL condition and distribution/noise conditions. | Method/assumptions inspected. Dynamic admission is evidence-backed, not new; convergence is not a command-correctness guarantee. Warm-up and schedule introduce additional choices, so not our first diagnostic. |
| [SoftMatch, ICLR 2023](https://arxiv.org/pdf/2301.10921) | Sections 3.1–3.2 use truncated-Gaussian weights and uniform alignment for weights, retaining original predictions for targets. Table 2 reports CIFAR-10/40-label error 4.91±0.12% versus FixMatch 7.47±0.28%; some settings do not improve. Section 4.1 uses confidence variance divided by four. | Method inspected. First soft-weight comparison target. Original training uses online prediction statistics, weak/strong views and model EMA; static replay is not a reproduction. Image flips/RandAugment are not automatically valid for directional landmarks. |
| [FreeMatch, ICLR 2023](https://arxiv.org/pdf/2205.07246) | Equations 5–12 combine global/class-specific moving confidence thresholds and fairness. Table 2 reports ImageNet/100-labels-per-class error 40.57% versus FixMatch 43.66%, both 0.4 seconds/iteration in its setup. Section 5.1 uses dataset-specific SVHN restrictions and reports best checkpoint errors. | Method inspected. First adaptive-threshold comparison target, with its fairness term separately identified. Never call threshold-only replay full FreeMatch; do not inherit test-checkpoint selection. No measured speedup for our code. |
| [Rethinking Confidence Scores and Thresholds in Pseudolabeling-based SSL, ICML 2025](https://proceedings.mlr.press/v267/vishwakarma25a.html) | PabLO learns confidence scores and thresholds with an explicit error parameter, reporting improved accuracy/training efficiency. Directly overlaps any claim to invent error-aware pseudo-label admission. | Official abstract and indexed algorithm inputs inspected; full PDF inaccessible through available web reader (raw PDF type unsupported; OpenReview browser check). Linked code returned 404. Do not implement from the abstract or claim a method-level comparison yet. |
| [Non-Stationary Predictions May Be More Informative, ICML 2025](https://proceedings.mlr.press/v267/pei25a.html) | Two-phase training predictions can identify useful pseudo-labels; the abstract reports mean gains of 1.73 percentage points on image tasks and 1.92 on graph tasks across eight datasets. | Abstract only. Training-time dynamics are different from movement within a gesture. Our saved final predictions cannot reconstruct the proposed metric. Defer until full method and logging requirements are checked. |
| [Influence-Aware Semi-Supervised Fine-Tuning for Hyperspectral Change Detection, ICML 2026](https://proceedings.mlr.press/v306/zhao26w.html) | IA-SFT evaluates low-confidence pseudo-labels by their impact on labeled data, with reported improvements on three hyperspectral benchmarks. | Official abstract only; code listed as forthcoming. Strong additional novelty boundary for “useful rather than confident.” Different modality/foundation-model setup; no gesture transfer demonstrated. |
| [Meta Pseudo Labels, CVPR 2021](https://arxiv.org/pdf/2003.10580) | Section 2 adapts a teacher from student performance on labeled examples through a one-step approximation. | Method inspected. Teacher feedback from learning benefit is already published. Extra teacher/student updates and fair label accounting matter; not the first implementation here. |
| [R3M, CoRL 2022 (PMLR 2023)](https://proceedings.mlr.press/v205/nair23a.html) | Human-video pretraining plus a frozen perception module supports data-efficient downstream manipulation; the abstract reports real-arm learning with 20 demonstrations. | Official abstract/indexed introduction inspected; full PDF exceeded web-reader size limit. Supports reuse of representations, not training a large video model on our 245 inputs. It consumes large upstream data and uses RGB/manipulation, whereas we already reuse MediaPipe and save landmarks. No direct R3M landmark baseline is implied. |
| [Robots That Ask For Help, CoRL 2023](https://proceedings.mlr.press/v229/ren23a/ren23a.pdf) | KnowNo uses calibrated prediction sets to execute or seek human help. Section 3 ties task-completion guarantees to its sampling/calibration assumptions and accurate help. | Method inspected. Supports evaluating action coverage together with errors/help. It is LLM planning, not gesture training, and its guarantee does not transfer to our reused clips or physical robot. |
| [Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf) | Theorem 1 controls expected bounded monotone loss under exchangeability; Section 2.3 explains the non-monotone failure and monotone-envelope remedy. | Method/assumptions inspected. Conditional error among accepted commands is not automatically this theorem's risk. Repeatedly tuned 48/96 cannot provide a fresh calibration guarantee. Keep decision calibration separate from training. |
| [Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?, ICML 2026](https://proceedings.mlr.press/v306/li26et.html) | The abstract identifies using tests for validation as a source of inflated results and advocates final-checkpoint, multiple-run reporting. | Abstract only; modality differs. Reinforces reporting fixed-final as well as development-selected checkpoints, every seed, and inspected endpoints honestly. |

## Diagnostic declared before new weights or oracle summaries

Question: on the **existing** 014/015 hidden-training predictions, do basic
published admission/weighting components put additional weight on correct UP
targets, or also on wrong NO_COMMAND-to-movement targets?

Budget: **zero model fits, zero network forward passes, zero 48/96 payload reads**.
Five seeds, 245 hidden-training clips per seed. These repeat 385 unique clips;
1,225 pairs are not 1,225 independent recordings. Previously inspected labels
make this retrospective even though new weights freeze before oracle access.

Use final saved ensemble probabilities q. Same unaltered argmax target for all
rules, including alignment. Fixed set of rules, no threshold/variance sweep:

1. Original dual-teacher agreement and minimum-confidence .95 mask (reference).
2. Ensemble-confidence .95 mask (separates teacher-agreement changes).
3. Global threshold = full hidden-pool mean maximum probability.
4. Class threshold = global threshold × mean q[class]/max(mean q), a static
   FreeMatch SAT component, without its fairness term or online EMA history.
5. Gaussian soft weight on raw max q, mean and unbiased variance from the pool,
   variance divided by four as specified in SoftMatch Section 4.1.
6. Same Gaussian using max(normalize(q/mean q)) for the confidence argument,
   retaining raw-confidence mean/variance and raw argmax targets (UA component).

Names must say **snapshot/component**. Full-pool statistics are a declared static
substitution for online moving averages, not reconstructed training histories.
Only predictions enter these functions. Copy/hash code, this declaration and
parent inputs before scoring; freeze all weights before separate oracle analysis.
Do not load recordings/models, gradient code, validation or endpoint predictions.

Report every seed/rule and pooled sums, predicted-class and actual-class weight
mass, weighted incorrect-target mass/fraction, correct-target mass where the
saved supervised student errs (only a usefulness proxy), and actual NO_COMMAND
weight assigned movement targets. Report denominators and zero-mass fractions as
undefined. Different mass/coverage means these are **not matched-quality rankings**.
No trained-student accuracy, causal usefulness, safety or superiority conclusion.
Check probability inputs, unchanged targets, finite [0,1] weights, simple analytic
cases, parent hashes and exact reproduction of the saved original mask.
For numerically zero confidence variance (at most (8×float64 epsilon)²), use
the zero-variance limit with an 8-epsilon roundoff tolerance. An analytic equal-
confidence check caught this corner case before any study weights were computed.

## Next training gate (not launched by this review)

First implement a common online confidence-consistency baseline and SoftMatch
adaptation using the existing raw149 LSTM. Audit label-preserving views before
fitting: a direction-changing mirror/rotation or time reversal is not an invariant;
any raw landmark change needs consistent validity/timestamps/derivative features.
Model dropout alone is a declared adaptation, not paper image augmentation.
Include supervised160 and the common fixed-confidence baseline. Reuse identical
160/245 splits, initialization, optimizer, step count and input pipeline; do not
use supervised405 weights or oracle labels. Count 48 selection labels separately.
Preflight the statistics timescale too: SoftMatch's m=.999 retains .999^560 =
57.10% of its initial statistic after our entire 560-update budget. Its image
recipe uses 2^20 updates. Blindly copying that schedule could leave most of our
run in initialization. Fix and disclose the online-statistics initialization/
timescale and EMA evaluation behavior before training, supported by arithmetic
and prediction-only traces; do not tune them against oracle/96 outcomes or claim
an exact reproduction. If the view/statistics contracts are unresolved, no fit.

Cap the first training comparison at **three methods × five fixed seeds = 15
student fits** (11/22/33/44/55), 560 updates each, CPU two threads, no sweep. Reuse a prior
supervised fit only if streams/selection exactly match. Report fixed-final and
48-development-selected checkpoints, every seed, full elapsed time and memory,
pseudo-label class coverage/error diagnostics. No validation/test gradients.
FreeMatch fairness, meta-learning, new architecture and self-supervised pretraining
remain outside that cap and need evidence plus a separate declared comparison.

Do not choose a new gate based on 96 again. On development data a candidate
continues only if mean macro F1 improves over the common confidence baseline,
at least three of five paired differences are positive, and mean NO_COMMAND-to-
movement does not increase or STOP recall decrease. This is a feasibility rule,
not a significance/reliability threshold. Record failure; no automatic rescue
sweeps. Later 96 scoring is optional retrospective description, never a pass/fail
criterion. A new algorithm must additionally beat the strongest applicable
published baseline and matched-count/class controls, not just our old .95 gate.

For a paper, audit a public multi-person temporal-gesture benchmark before any
large download or training, retain native labels/modality, split by person/session
and freeze a genuinely uninspected endpoint before method selection. No new
personal recordings requested. A robot simulation with an explicit command-to-
action mapping can test consequences later; clip accuracy alone cannot establish
continuous false-activation rate, latency or suitability for cars/surgery/drones.

Working hypothesis only: can training use hard, informative pseudo-labels while
avoiding amplification of background-to-movement errors at a fixed label and
compute budget? Risk weighting, utility weighting and staging are not themselves
novel. We need a distinct mechanism, strong comparisons and external replication
before proposing a publication contribution. No evidence of that contribution yet.

## Outcome

Completed run:
[20261017T072612284949Z](../experiments/runs/published_selector_replay/20261017T072612284949Z).
The archived copy of this file contains the pre-scoring declaration. R3M and
the future EMA-timescale preflight were added after scoring; neither changes
any diagnostic rule, weight, result or selection.

All 1,225 seed–clip pairs analyzed; six rules, five seeds, no new fits, forward
passes, raw data loads or 48/96 prediction reads. The replay itself took 0.039
seconds, excluding review, implementation, tests and independent verification.
Seven software checks pass. Scalar formula recomputation matches all weights;
five original masks reproduce exactly; all six parent input files and 32 older
frozen Python sources remain unchanged. Code/plan/inputs and weight freeze hashes,
access ledger, per-seed and actual/predicted-class CSVs are saved.

“Mass” below is the sum of example weights: two half-weight clips contribute one
unit. Hard masks have integer mass. Wrong fraction is wrong-target mass / total
mass, **not a trained classifier's error or a safety guarantee**.

| Static rule | Total mass | Incorrect-target mass | Incorrect fraction | Correct-target mass on student mistakes |
|---|---:|---:|---:|---:|
| Original dual teacher .95 | 89.00 | 5.00 | 5.62% | 1.00 |
| Ensemble .95 | 135.00 | 8.00 | 5.93% | 1.00 |
| Global mean threshold | 656.00 | 56.00 | 8.54% | 15.00 |
| Class-adaptive component | 984.00 | 184.00 | 18.70% | 45.00 |
| Gaussian component | 878.73 | 112.15 | 12.76% | 32.28 |
| Gaussian + UA component | 796.64 | 107.22 | 13.46% | 27.32 |

Correct UP-target mass rises from zero with the original mask to 31 global,
99 class-adaptive, 66.86 Gaussian and 72.23 Gaussian+UA. This supports the
admission-starvation diagnosis. It also brings noise: class-adaptive accepts
125 predicted-UP pairs, 26 incorrect, and all 53 predicted-NO_COMMAND pairs,
28 incorrect. Of 105 actual NO_COMMAND pairs it admits 90, including 58 with
movement targets and 7 with wrong STOP targets. Gaussian+UA puts 51.87 total
mass on actual NO_COMMAND, including 34.54 on movement targets. It reduces
that conditional fraction versus the raw Gaussian while increasing absolute
false-movement target mass. Both quantities must be visible.

The extra correct targets on student mistakes are only a proxy for potential
learning opportunity; no student was trained. Different total and class mass,
teacher aggregation and absent online dynamics preclude a method ranking.
This is **not** evidence that SoftMatch/FreeMatch fail or succeed on our task.

Decision: do not promote a lower threshold, normalization or a new “risk/utility”
algorithm now. Establish the bounded online supervised/confidence/SoftMatch
comparison above after the landmark-view/implementation audit. Investigate
background error propagation explicitly rather than assuming confidence or
class balancing repairs it. Only add a distinct mechanism if it improves on
those baselines under the declared development criterion. No extra fits are
launched here; all failures and alternatives remain recorded.

This file exists to connect evidence, source access limits, reproducible cheap
diagnostics and the compute decision; it is not another lesson chapter.
