# Temporal gesture research: mechanism, comparisons and a falsifiable contribution

Latest2026-10-15: experiments011/012 completed; motion's earlier uniform advantage
didnotreplicate acrossmodels/newrecordings. User now prohibits requests fornew
personalvalidation/testclips. [Candidate013](ideas/013_evidence_profile/README.md)
specifies cross-fitted classconditional evidence-response admission for a sparse-
labelteacher/student pilot, withexact failurechecks andexisting-data roles.
This supersedes newcollection as the next action, not the limitations ofolddata.
No new realtraining/SOTA/noveltyclaim; [Lesson17](LESSON_17.md) explains the idea.

Started 2026-10-12 in response to the user's request to understand coordinate
motion and investigate a new algorithm. This is a research plan, not a novelty,
state-of-the-art or publication-readiness finding. The first review is bounded;
not every citation/supplement/code/proof was read or reproduced.

**2026-10-15 update:** [experiment010](findings/010_motion_representation.md) tests
explicit physical-time motion and orientation-preserving hand coordinates with
one matched LSTM. Camera position/motion accuracy84.17/97.92%, hand91.67/92.08%,
on the inspected48. [Lesson14](LESSON_14.md) explains the gains, regressions and
why prefix agreement is not real early-command latency. No unlabeled learning,
newalgorithm or robust unseen-user result follows. Newpost-freeze evidence and
the separate audited policy remain outstanding.

**2026-10-13 update:** The first finite comparison is complete: CNN, pose set,
feature set, LSTM and BiLSTM. Mean inspected validation accuracy 85.83/85.83/
85.42/89.58/88.75%. LSTM improves recognition but increases NONE→motion to
33.33%; BiLSTM has 10% on that error but STOP recall remains 66.67%.
Read [finding 008](findings/008_temporal_architectures.md),
[Lesson 12](LESSON_12.md) and [the declared protocol](TEMPORAL_COMPARISON_PROTOCOL.md).
No representation/architecture selected as an overall reliability winner.
DD-Net/graphs/derivative inputs remain a separate next representation comparison;
the audited learning policy is not implemented yet.

## What we actually use

`gesture_commands/data.py:40` supplies 32 chronological samples of 67 features:
63 wrist-relative/palm-normalized xyz values, wrist XY displacement from the
beginning, log apparent hand-size ratio, and visibility. Raw timestamps determine
interpolation but absolute duration/dt, explicit velocity/acceleration and absolute
wrist depth are absent from the input. Size change is not measured physical depth.

`gesture_commands/model.py:13` is two temporal Conv1d(k5) layers, pooling into 4
ordered time bins, then a dense head. The 8-class research head gives 48,520 weights.
Local convolutional receptive field is 9 samples; the final head accesses all 4
bins/whole clip. Fine order is compressed, not categorically discarded. Current
model is not LSTM, BiLSTM, Conv3d, graph attention or the causal/dilated/residual
TCN often described in papers. It can learn coordinate changes through temporal
filters, but that capacity does not establish what its saved weights rely on.

3D landmark coordinates describe data dimensions; they do not determine the
convolution operator. Conventional video Conv3d processes time/height/width, as
in [C3D](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf).
Skeleton-only Conv3d can use explicit volume encodings, as in
[Liu2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Decoupled_Representation_Learning_for_Skeleton-Based_Gesture_Recognition_CVPR_2020_paper.pdf).
We persist landmarks, not RGB. Neither RGB nor skeleton-volume Conv3d is currently
implemented here; adopting one changes input/encoding/resources and requires a
disclosed adaptation. Liu's indexed abstract/introduction were accessible;
its full representation equations were not verified in this review.

LSTM keeps a learned recurrent state over successive samples. BiLSTM also reads
the complete supplied window backward. For a finished trailing window that
backward pass uses already captured frames, not future camera frames. Window
delay/recomputation cost still matters; it is not a no-latency streaming state.
A causal CNN is another streaming option. These are alternatives to compare,
not universal upgrades. [Bai2018](https://arxiv.org/html/1803.01271v2) evaluated
TCNs against recurrent models across multiple sequence tasks; its results do
not identify a winner for our eight-command dataset.

## Review already completed and novelty objections

Separate primary-source notes retain access and reading depth:

- [Temporal/hand architecture review](notes/2026-10-12_temporal_prior_art.md):
  DD-Net, DG-STA, hand graph/recurrent/attention methods, continual detection,
  skeleton-volume CNN, recent skeleton-to-vision encodings.
- [Label-efficiency and risk review](notes/2026-10-12_label_efficiency_review.md):
  hand SSL, skeleton pretraining/active labels, pseudo-label teachers, cost/risk
  control and recent robotics reliability mechanisms.
- [Exact local mechanism and experimental critique](notes/2026-10-12_mechanism_protocol.md).

Close examples: [DD-Net](https://arxiv.org/pdf/1907.09658) already separates joint
distance shape and multiscale coordinate differences in a compact CNN.
[DG-STA](https://arxiv.org/pdf/1907.08871) models spatial and temporal hand-graph
attention. [MMANet2025](https://link.springer.com/article/10.1007/s40747-025-01807-x)
already joins hand joint/bone/motion representations with unlabeled contrastive
learning and supervised anchors. Thus pose+motion+attention+few labels is not
a new contribution by itself.

For the learning policy, [SESAR](https://arxiv.org/html/2012.01740) already combines
skeleton reconstruction, sparse supervision and active selection;
[AS3L](https://arxiv.org/html/2203.04560) combines self-supervision, active labels
and semi-supervision. [CSST](https://proceedings.neurips.cc/paper_files/paper/2022/file/acb94e709f02895fd98b5867f0b184f3-Paper-Conference.pdf)
uses cost-sensitive consistency/self-training.
[Prediction-powered risk control](https://arxiv.org/html/2412.11174) combines
labeled corrections and many unlabeled predictions under sampling assumptions.
[SemiVLA2026](https://arxiv.org/html/2606.21493v1), method overview inspected,
already gates embodied pseudo-actions with robotics-specific reliability checks.
Their tasks/assumptions differ, but they rule out broad claims based on merely
combining stage names, cost weighting, calibration or robotics.

Unavailable full methods/code are unknowns, not evidence of novelty. Closest
equations and implementations still need comparison before a contribution claim.
Do not call a skeleton-only adaptation a reproduction of a multimodal method
whose original RGB/pretraining resources we lack.

## First empirical result: existing-data motion sensitivity

Executed [experiment007](findings/007_motion_sensitivity.md) on only the inspected
48 v3 validation clips, all 5 saved models. No weight updates or labels changed.
All 240 baseline vectors were reproduced; all 1,680 diagnostic vectors were independently verified.
Removing wrist XY changed 0/240 top labels; shuffling frames changed 0/240; raw
reversal with proper re-extraction changed 1/240. Repeating center local shape
changed 17/240 and reduced BACKWARD original-label agreement 70%→36.67%.
Probabilities still changed, by up to .0908 for wrist removal, .0575 for shuffling and .1542 for reversal.

This suggests local shape changes matter while strong chronological-order
reliance is unproven on this sample. It does not prove unused motion channels,
causal shortcut learning, lack of temporal capacity or a better modified model.
Perturbations are potentially OOD/meaning-changing; their original-label agreement
is not valid new-task accuracy. All STOP shared errors persisted. Current v3 UP
is a held-pose task, so pose recognition is useful rather than inherently wrong.

## Next controlled study: separate representation from algorithm

No new architecture fits were run during the 2026-10-12 review/probe step.
The 2026-10-13 completed fitting comparison is linked in the update above.
Before fitting a finite comparison, archive exact settings and keep v3 meanings,
eligible training files and whole-session roles constant. Initial comparison:

1. Current 48,520-parameter CNN, with original weights as a fixed reference plus
   explicitly disclosed fresh paired runs if row order/implementation changes.
2. Order-invariant/pose-only comparator, to reveal which commands need time order.
   Distinguish averaging the67-feature multiset from using only local pose/visibility.
3. LSTM on the same 67 features. A one-layer LSTM with 62 hidden units and the same 4-bin
   pooling/64-unit head has approximately 48,944 parameters; measure exact count.
4. BiLSTM with 36 hidden units per direction and matching pooling/head, approximately 49,256
   parameters; measure exact count and complete-window latency.
5. One modest prior-art motion/skeleton comparator: DD-Net adaptation or tiny
   hand-GCN+temporal stack. Explicit differences/velocity, angles, graph structure
   and multiscale contexts are established components, not inventions.

Parameter targets are design calculations, not implemented/validated candidates.
Representation additions require separately versioned features/checkpoints;
never edit hash-bound core/collectors or silently activate a new model.

Hold labels, samples, seeds, sample-order streams, per-class weighting, selector
and tuning opportunities matched where compatible. Equal 560 updates does not
equal equal compute: report actual training time, peak memory, parameters and
per-window/continuous latency on intended hardware. The 48 inspected single-user
validation clips cannot support a large hyperparameter/model lottery. Use a small
prespecified comparison and session-grouped development checks; keep further
evaluation labels/costs explicit and final participants/sessions untouched.

First separate component tests: original representation versus explicit masked
motion differences/dt-aware velocities; static versus ordered input; trajectories
versus articulation; temporal-span/dilation versus size-matched recurrence.
Derivatives can amplify tracker noise. Do not normalize away direction or treat
all coordinate rotations/time reversal as command-preserving augmentations.

## Candidate learning mechanism to formalize, not yet a novel algorithm

Research hypothesis: **a learner that audits confident predictions and uses
those audits to control pseudo-label admission can lower consequential semantic
errors at matched human/compute budgets, beyond ordinary SSL and active review.**

The representation could distinguish pose and trajectory, but the candidate
contribution would have to reside in the exact review/admission/update rule and
measured benefits. Current shared STOP→FORWARD scores above .99 show why same-family
agreement/confidence alone cannot label truth. They do not prove our proposal works.

Proposed components to specify before implementation:

- Supervised anchors, optionally after self-supervised motion representation
  training on eligible training-only inputs. Compare stage order; no universal
  supervised→semi-supervised→unsupervised optimum is claimed.
- A distinct unlabeled format, with no NO_COMMAND placeholder, no hidden human
  labels/origin-label metadata leaking into the learner, and no validation/test
  sessions or final-test users entering pretraining.
- Teacher targets frozen per round; motion/pose disagreements, tracking quality,
  predicted command and label-cost neighborhoods define review strata.
- Targeted uncertainty/diversity queries plus a separately accounted randomized
  audit of high-confidence candidates, preserving positive review probabilities.
- An explicitly estimated target-error/contamination quantity for the actual
  proposed admitted pool, with assumptions, sample design and confidence treatment
  stated. A biased actively selected audit average or a bound on a different pool
  cannot certify admission. Repeated rounds require valid fresh/sequential handling.
- Audits may admit, defer, reweight or request supervised coverage; human-reviewed
  labels become anchors. One correction never becomes ground truth for all nearby
  clips. Student/teacher feedback and label leakage must be independently ablated.

These components are not a finished rule or theorem. Allocation, thresholds,
estimators, residual-pool definition, stopping/update schedule and tuning budget
remain to formalize. Existing risk/control/SSL methods must be matched, and
standard concentration inequalities are not themselves a new proof contribution.

Robotics costs can prioritize review and evaluate errors, but a physically
feasible FORWARD rollout does not prove FORWARD was intended. Semantic command
labels stay separate from controller feasibility/state. Define a bounded cost
matrix and robot context before comparing methods; state whether it is a simulated
surrogate or measured robot consequence. Separate improving recognition/learning
from simply refusing more commands at execution.

## Required evidence and falsifiers

Freeze a representation before the learning-policy comparison. Compare sparse
supervised, full-label reference, self-supervised+supervised, adapted standard
FixMatch/MeanTeacher/FreeMatch, hand-skeleton SSL, random/uncertainty/diversity
review, cost-sensitive/risk-guidance methods, then the candidate audited policy.
Use the same pool/starting labels and report extra teacher/pretraining/tuning/
review compute. Ablate random confident audits, consequence weighting and admission
feedback separately, rather than crediting their bundle from one end score.

Budgets count initial labels, confirmed/corrected predictions, repeated reviews,
calibration, validation and final evaluation. A hidden-label simulation on the
already labeled 538 clips can test mechanism comparisons but cannot retroactively show
saved annotation minutes. Audit contamination/coverage and final semantic error
must improve beyond strong matched-budget alternatives; always rejecting or always outputting STOP
cannot win merely by avoiding movement.

Measure macro F1/classes, harmful STOP/NONE→motion, pseudo-target errors, admitted
pool coverage, human minutes and runtime. Stress new sessions/users, unknown motions,
imbalance, missing hands and transitions. Unknown/invalid/ambiguous observations
are not automatically true NO_COMMAND. Later continuous trials add false moves/
uncommanded minute, STOP miss/delay, useful command coverage and task completion.
Participant-separated public and local benchmarks precede broad unseen-user claims;
simulation/hardware evaluations are reported separately.

Falsifiers: no gain over random auditing, benefit explained by extra capacity/
labels/compute, representation gains disappearing under shift, admission reducing
coverage too much, fewer bad pseudo-labels without better recognition, or robot
gains caused solely by gate refusal. Preserve negative results. A specific new
algorithm and publication claim follow evidence/novelty review, not promises.
