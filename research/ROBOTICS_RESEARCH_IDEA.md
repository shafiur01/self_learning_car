# Working proposal — consequence-aware audited learning for robot commands

Latest2026-10-15: no more personal validation/test recordings per user. Immediate
mechanism candidate is [evidence-response pseudo-label admission](ideas/013_evidence_profile/README.md),
using onlyexistingdata and no extra humanreview prerequisite. The earlier
human-audit/robot-control research direction below is notimplemented. Recognition
error costs remain evaluation outcomes; feasibility of a robotmotion still does
not identifyhumanintent. Currentproposal notnovelty/safety/performance finding.

2026-10-12 update: [TEMPORAL_ALGORITHM_PLAN](TEMPORAL_ALGORITHM_PLAN.md) adds
closest hand-SSL/active/risk-control prior art and a fixed-weight motion diagnostic.
The candidate audit/admission question remains a hypothesis, not a novel result.
Representation/temporal changes must be tested separately from learning policy;
semantic intent stays separate from action feasibility or execution refusal.
No robot experiment, unlabeled pool or new algorithm learner implemented yet.

Written 2026-10-07, following the user's request to make robotics a central part
of the research. This is a proposal for discussion, not a completed algorithm,
preregistered experiment, novelty finding, or publication claim. No training,
robot trial, installation, or camera checkpoint change accompanied this review.

## Recommended main question

Can a temporal command learner use fewer human annotation minutes by auditing
and limiting consequential pseudo-label errors, while preserving recognition
of unseen people and useful robot task completion?

Working paper title: **Consequence-Aware Audited Semi-Supervised Learning for
Human–Robot Command Interfaces**. This describes the intended investigation;
it must change if the experiments support a different contribution.

The target is a human-command interface for an indoor mobile assistant. It
learns what a person communicated; a separate controller turns that command into
motion. It does not learn an entire autonomous-driving or humanoid policy from
our existing recordings. Primary unseen-user evaluation remains in force.

## Why this question follows from our project

Our historical development evidence contains confidently wrong NO_COMMAND to
movement predictions, including a FORWARD score around 0.9965. Their existence
motivates auditing confident predictions; it does not establish how frequently
the problem occurs in other people or environments. The old test has already
informed development and cannot certify the proposed method.

Experiment 001 also found no mean accuracy or macro F1 improvement from the
initial jitter/time-warp settings. Thus, more stages or augmentations cannot be
assumed to help. The research should investigate a particular failure mechanism:
wrong machine-generated labels can teach the next model the same mistake, and
different mistakes have different consequences for a robot.

For example, misreading NO_COMMAND as FORWARD can start unwanted motion;
misreading STOP can delay a required stop. Ordinary clip accuracy counts each
wrong label once. Robotics also asks what movement occurred, whether the task
was completed, how long recovery took, and how much help was needed.

## The central distinction: meaning and consequences

Let x be a hand-motion event, y its intended command, and s the robot state.
The semantic recognizer estimates q_theta(y | x). A controller uses the predicted
command and s to choose an action. A LEFT gesture keeps its semantic label even
when an obstacle makes moving left inappropriate.

If a person signals LEFT and the robot goes RIGHT in an empty room, a
collision-free rollout does not establish that RIGHT was the correct label.
Both commands can be physically feasible. Robot outcomes tell us what happened;
they do not generally reveal what the person meant. Explicit intent labels,
corrections, or independently justified task supervision remain necessary.

Define C(s, k, y) as a bounded cost of interpreting the command as k when the
intended command was y. Include semantic/task regret as well as physical effects.
Freeze the controller, horizon, units, normalization, and cost components before
comparing learners; report the individual outcomes alongside any composite cost.

Training decisions must consider prespecified development contexts or stress
strata, not just whether the current room is forgiving. A wrong pseudo-label
accepted in open space can contaminate a reusable model later deployed near an
obstacle. Consequences may tighten admission requirements or prioritize review;
they must not make semantically unreliable labels acceptable in easy states.

## Candidate mechanism — the component we would investigate

The proposed contribution is a joint rule for **human review and pseudo-label
admission during training**, informed by observed annotation errors and robot
consequences. A weighted confidence score alone is insufficient novelty.

1. Optionally learn temporal representations from development-training motions
   without command labels. Compare reconstruction/consistency to no pretraining;
   preserve direction and timing relevant to commands.
2. Use a small, accounted-for human-labeled set to anchor command meanings.
   Give comparison methods the same starting labels and pretrained detector.
3. Produce candidate pseudo-labels, with semantic uncertainty estimates. Keep
   tracking failures, unfamiliar non-command motions, and event boundaries
   visible rather than forcing every window into a known command.
4. Split review resources into targeted acquisition and a separate uniform
   random audit of the frozen candidate pool, including confident predictions.
   Targeted acquisition may prioritize consequential errors; the uniform audit
   supports the bound below. Unequal-probability/stratified audits need their own
   estimator and bound. Review whole events rather than overlapping windows.
5. A concrete candidate rule admits a frozen pseudo-label batch only if both
   semantic- and consequence-contamination audit bounds satisfy prespecified
   ceilings; otherwise defer supervision or acquire corrections. Compare this
   rule to ordinary calibrated weighting and simpler review rules. Corrections become
   human-labeled examples; do not copy one correction to an entire cluster as
   ground truth. Keep supervised anchors throughout retraining.
6. Repeat under a fixed annotation/compute budget, preserving checkpoints and
   provenance. Compare interleaved auditing with fixed stage orders. An audit
   may recommend pausing pseudo-label updates while representation learning or
   supervised corrections continue; that recommendation must be tested.

This is more specific than supervised → semi-supervised → unsupervised. We
propose optional self-supervised structure learning → semantic supervision →
audited semi-supervised rounds. The order is an experimental variable, not an
established best sequence. Unlabeled structure cannot manufacture class meaning.

An acquisition score could estimate the reduction in consequential training
error per review second, with diversity and an exploration/audit allocation.
That estimate is a surrogate to validate, not an oracle. Confident systematic
errors can fool uncertainty, ensembles, and temporal agreement simultaneously.

## What the focused prior-art review already rules out

This is a targeted review, not an exhaustive citation-tree or priority search.
The following distinctions come from primary methods/sections inspected unless
the row explicitly says otherwise.

| Prior work | Established territory and implication |
| --- | --- |
| [TCL, CVPR 2021](https://rpand002.github.io/data/CVPR_2021_TCL.pdf) | Staged self-supervised, temporal semi-supervised, and pseudo-label learning already exists, including gesture videos. Stage order alone is insufficient. |
| [SESAR, 2020 preprint](https://arxiv.org/pdf/2012.01740) | Unlabeled skeleton reconstruction plus sparse labels and active selection is established. A skeleton representation plus active review alone is insufficient. |
| [ThriftyDAgger, CoRL 2021](https://proceedings.mlr.press/v164/hoque22a/hoque22a.pdf) | Novelty and learned task-failure risk already allocate human interventions under a budget. Its target is a robot policy, rather than gesture-label contamination. |
| [Risk-Aware Active IRL, CoRL 2018](https://proceedings.mlr.press/v87/brown18a/brown18a.pdf) | Queries based on posterior policy-loss bounds already connect label selection to downstream consequences. |
| [Learning Semantics of Gestural Instructions, 2018](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2018.00007/full) | Gesture–robot-action associations, incremental feedback, and contextual interpretation already exist. Gesture-to-robot integration alone is insufficient. |
| [Learning Latent Actions to Control Assistive Robots](https://arxiv.org/pdf/2107.02907) | Section 6 combines limited human alignment labels with proportionality, reversibility, and consistency priors. Physics/control-inspired semi-supervision is established. |
| [CSST, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/acb94e709f02895fd98b5867f0b184f3-Paper-Conference.pdf) | Cost-sensitive consistency and pseudo-label selection already optimize objectives beyond ordinary accuracy. Static command weights alone are insufficient. Its theory makes distribution, augmentation, and teacher-quality assumptions. |
| [Towards Cost-Effective Learning, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_Towards_Cost-Effective_Learning_A_Synergy_of_Semi-Supervised_and_Active_Learning_CVPR_2025_paper.pdf) | Active and semi-supervised learning already interact through representation and pseudo-label effects. Search-indexed primary method text inspected; direct PDF retrieval failed. |
| [AdaConG, arXiv v4, 2025](https://arxiv.org/html/2502.16736v4) | Conformal uncertainty already weights pseudo-label guidance inside training; also studies imitation-guided RL. Venue unverified here. Calibration inside training alone is insufficient. |
| [Risk-Calibrated Human–Robot Interaction / RCIP, RSS 2024](https://arxiv.org/html/2403.15959v1) | Intention predictions inform action sets and calibrated help decisions, including sequence-level/multiple risks. Our candidate training mechanism must be distinguished from execution-time planning. |
| [Semi-Supervised Active Learning Using Informative Path Planning, 2024](https://arxiv.org/pdf/2312.04402) | Robot data acquisition, sparse human labels, and uncertainty-filtered pseudo-labels already coexist in semantic segmentation. Robotics plus active/SSL stages is insufficient. |
| [Conformal Policy Control, ICML 2026](https://proceedings.mlr.press/v306/prinster26a.html) | A trusted policy regulates a new policy under a calibrated risk objective. Source and theoretical assumptions screened; no direct claim about gesture learning or robot hardware follows. |

Additional inspected related methods: [SafeDAgger](https://arxiv.org/pdf/1605.06450),
[HG-DAgger](https://arxiv.org/pdf/1810.02890),
[KnowNo](https://arxiv.org/pdf/2307.01928), and
[Unified Learning from Demonstrations, Corrections, and Preferences](https://arxiv.org/pdf/2207.03395).
[Uncertainty-aware emergency gesture HRI, 2025](https://www.sciencedirect.com/science/article/pii/S0924424724011725)
was inspected only through its publisher preview; its unavailable full method
cannot be assumed to lack our proposed component.

The remaining candidate question is whether a budgeted, sampling-aware audit
of consequential confident errors changes temporal semantic learning more
effectively than these strong alternatives. No inspected source establishes
that our exact proposed mechanism is original. The next novelty review must
compare exact admission/audit rules and code, not just paper titles. If our rule
reduces to existing cost-sensitive self-training plus ordinary random audits,
drop the claim of a new algorithm; a rigorous empirical insight may still be
worth investigating if the results support it.

## A possible mathematical component, with a precise limit

Freeze an accepted candidate pool A and its pseudo-labels before auditing. Draw
n events uniformly without replacement. Record ordinary semantic error
e_i = 1[pseudo_label_i != human_label_i]. Define h_i = e_i * C_bar_i in [0, 1],
where C_bar_i is normalized misinterpretation cost averaged over a prespecified
development-state distribution, or within a frozen stress stratum. This measures
consequential pseudo-label contamination, not all robot hazards.

Reviewed labels must follow the frozen annotation protocol. If human annotations
are ambiguous or wrong, the bound measures disagreement with them rather than
verified intent. Record ambiguities and adjudications; do not force a label.

For either loss, a conservative simultaneous upper bound is its audit mean
plus sqrt(log(2/delta)/(2n)), capped at one. Applying the bounded-loss sampling
inequality twice gives confidence at least 1-delta for both frozen-pool means.
This follows established finite-population concentration, not a new theorem:
[Bardenet and Maillard, 2015, Lemma 1.1 / Proposition 1.2](https://arxiv.org/pdf/1309.4029).

These bounds concern that fixed pool, not final-model accuracy or future robot
safety. Small audits can leave wide bounds. Adaptive rounds, threshold searches,
or changed models require valid repeated-use corrections/fresh audits; selective
queries are not a uniform audit. Correlated windows do not break this finite-pool
randomization argument, but can distort its population. Count audit labels and
minutes. A useful research method must earn their cost.

The displayed bounds apply to the original A. If reviewed items are removed,
or an audit is used to choose a different subpool, use a justified residual-pool
bound or fresh/corrected auditing; the original denominator cannot be reused
silently. Additional stress-stratum bounds also require confidence allocation.

Conformal prediction also requires its own calibration assumptions: marginal
label coverage is not accuracy conditional on singleton acceptance, and neither
is collision-free hardware operation. Any stronger theorem would need an
explicit statement and proof linking the specified learning rule to its target.

## Robotics experiments that could support or refute the idea

Start with an indoor differential-drive assistant and a transparent development
simulator. An established physics simulator and small real robot would provide
later evidence; neither has been chosen or installed. TurtleBot3 is one possible
platform with [official simulation documentation](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/),
not a claim that its stack already runs on this Mac.

Freeze command semantics: bounded translation/turn requests in an explicit
robot coordinate frame; STOP interrupts; NO_COMMAND starts no new motion. The
current mirrored-screen LEFT/RIGHT conventions need an explicit mapping. A
beckoning gesture labeled FORWARD does not locate a person or implement
person-following. Version any incompatible semantic change; preserve old data.

Use waypoint assistance, obstacle detours, direction corrections, and stopping
at marked boundaries. Pair the same gesture events with low- and high-consequence
states during development to isolate the contribution of consequences. Use
development worlds only to construct costs. Final evaluation should include
new people, sessions, and held-out layouts/task conditions.

Two distinct comparisons are essential:

- Different trained models with **the same frozen execution rule/controller**:
  does the new admission/audit rule actually improve learned weights?
- **The same frozen model** with different execution rules: how much improvement
  comes from more conservative decisions rather than better learning?

Initially compare recorded streams with identical simulator conditions. Then
use live closed-loop interaction to test response and corrections. Recorded
stationary-camera replay does not reproduce moving-camera observations or a
human reacting to the robot. Report simulation and hardware results separately.

Measure accuracy/macro F1, accepted pseudo-label error, continuous false movement
events/minute, STOP detection delay and actual stop distance, task completion,
contacts/collisions, unintended displacement, recovery time, help requests,
coverage, review minutes, training/retraining cost, and inference latency.
Group events by original streams; seeds and overlapping windows are not
independent people. Compare risk at matched useful execution/task completion.

## Baselines and tests of the proposed mechanism

Use a well-tuned fully labeled reference; low-label supervision; optional
self-supervised pretraining; standard and adaptive-threshold SSL; ordinary
uncertainty/diversity active selection; calibrated guidance weighting;
cost-sensitive SSL; random auditing; and consequence-stratified random review.
Reproduce compatible methods or explicitly label adaptations; DAgger policy
scores cannot be directly ranked against gesture-classification scores.

Match eligible training pools, starting labels, pretrained resources, tuning
allowances, review budget, model capacity, and controller. Include calibration,
confirmation, audit, correction, and temporal-boundary annotation in costs;
also report these categories separately. Fully labeled data is a strong
reference, not a guaranteed upper bound. Simulated hiding of existing labels
estimates label efficiency, not observed savings in human work.

Ablate consequences, confident audits, and the admission rule independently.
Compare fixed versus interleaved stage ordering with matched resources. Test
ordinary motions resembling commands, person/session shifts, tracking loss,
unlabeled-pool mismatch, and cost-model errors. Include always-stop as a diagnostic:
zero collisions with zero task completion cannot establish a useful learner.

Before final testing, prespecify acceptable performance differences and cost/risk
targets using development variability. A favorable conclusion needs improvement
over strong matched-budget alternatives and maintained meaningful performance.
If audit cost erases savings, if simple random review matches us, or if all gains
come from refusal/postprocessing, report that outcome and revise the claim.

## Benchmarks and scope of broader robotics relevance

[SHREC 2021](https://univr-vips.github.io/Shrec21/) is a candidate continuous
gesture benchmark. Revised annotations remove POINTING, leaving 17 classes;
the original 18-class scores are not directly comparable. The
[paper](https://arxiv.org/pdf/2106.10980) describes train/test people separated.
LeapMotion joint data require a checked adapter rather than assumed MediaPipe
compatibility. This benchmark tests perception, not robot task success.

[FR-GESTURE, 2026](https://arxiv.org/html/2602.17573v1) provides static RGBD
commands for UGV interaction, with unseen-signer evaluation. It is useful
robotics prior art or a separate static-image study, not a temporal replacement
for our one-hand landmark clips. Access/license and exact benchmark protocols
must be checked before adopting either dataset.

| Application | Connection and evidence needed |
| --- | --- |
| Indoor mobile/helping robot | Primary proposed command interface; simulation first, hardware evidence later. |
| Assistive manipulator or humanoid assistant | Possible later high-level intent interface. A distinct task/controller and applicable users are required; mobile-base results do not establish manipulation or locomotion. |
| Drones | Possible human-command supervision layer. Flight dynamics, sensing, delay, and recovery require separate evaluation. |
| Autonomous cars | The broader learning-allocation principle could be investigated for perception/intention prediction. Gesture control of a toy/mobile base does not establish road autonomy. |
| Surgical robots | Only a distant research possibility; current data/tasks establish no surgical relevance or clinical reliability. |

A second mobile dynamics/controller experiment would test transfer within mobile
robotics. Claims of transfer across embodiments require a genuinely different
embodiment and corresponding data/task evidence. They are not requirements to
start the pilot and should not inflate the initial paper's claim.

## Current readiness and next planning discussion

We have a recorder, landmark preprocessing, supervised temporal model, explicit
corrections, and one completed augmentation study. We do not yet have the
unlabeled-event format, participant metadata, robot state/action logs, simulator,
calibration/audit implementation, or unseen-user final recordings required here.

The existing 180-new-validation-clip pilot and supervised label-budget comparison
remain useful for current perception development. They cannot by themselves
answer the robotics question. Next discuss the exact robot task, command
coordinate frame, closest baseline implementations, available participants,
annotation budget, and milestones before a larger collection or algorithm run.

This file exists to retain the main idea, its rationale, prior-art objections,
and the evidence it would require. CONTEXT/PROTOCOL/LOG retain project state,
experimental commitments, and the historical sequence of decisions respectively.
