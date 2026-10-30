# Precise candidate: self-training from evidence-response profiles

Version0.1, proposed2026-10-15. Design only; empirical success and originality
unestablished. This document settles one central mechanism rather than combining
many optional techniques and attributing success to a pipeline name.

## Research hypothesis

At fixed revealed-label and compute budgets, admitting pseudo-labels using
class-conditional response compatibility will reduce pseudo-target error and
improve a small student's macroF1 relative to confidence-only and embedding-
distance selection, without increasing STOP/NONE→movement mistakes.

A **pseudo-label** is a model prediction used as a training target. It is not
newly discovered ground truth. The proposal concerns training-data admission,
not a robot execution gate; rejecting a pseudo-label does not output STOP or
NO_COMMAND for the original clip.

## 1. Keep three evidence groups and shared observation context

Reuse version2 32×146 camera-motion features, without changing frozen extractor
code or command meanings. Proposed column partition, Python half-open notation:

| Group | Columns | Meaning |
| --- | --- | --- |
| P, pose | `0:63`, `134:143` | Camera-oriented palm-relative coordinates and estimated palm orientation |
| A, articulation | `67:130` | Derivatives of palm-relative local coordinates in camera axes |
| G, transport | `63:66`, `130:133` | Relative global wristXY/logsize and their derivatives |
| C, shared context | `66,133,143,144,145` | Tracking/derivative/geometry validity, elapsed seconds and interval |

72+63+6+5=146; every numerical column belongs to exactly one group. A is **not**
pure joint articulation disentangled from wrist rotation: camera-axis derivatives
can mix articulation/orientation changes. Coordinate groups remain deterministically
related, and P includes temporal pose samples that can already reveal motion.
The response is conditional on this representation, not identifiable causal
importance. Estimated depth and normalized velocities are not calibrated metres.

Mask an unavailable group to zero and append three explicit availability bits,
giving149 numerical inputs/sample. Numerical zero with bit0 means unobserved,
not an assertion that the hand physically stopped or became a zero pose.
Shared context remains observable and might itself support predictions; this is
another possible shortcut to audit, not evidence that the three groups capture
every source of information.

Only four availability patterns initially: all, withoutP, withoutA, withoutG.
This is missing-observation training. It does not reverse gestures, rotate
directions, swap left/right, invent command labels for corrupted clips, or
assume that removing information preserves recognizability.

## 2. Teach the model what incomplete observations mean

For teacher T trained on its labeled subset L_T, use

    L_T = E_(x,y in L_T) E_(m~pi) [-log p_T(y | mask(x,m), C, m)]

with proposed pi(all)=1/2 and pi(each of the three omissions)=1/6. The teacher
sees these missingness patterns during training; probing an ordinary full-input
model with arbitrary zero blocks would instead risk out-of-distribution artifacts.

This proper conditional cross-entropy estimates P(Y | available observations).
It does not penalize full/subset prediction disagreement or force their
distributions to match. If the same visible pose occurs equally in STOP and
FORWARD, a well-estimated pose-only posterior can be50/50. Finite sparse data
can still overfit this posterior; the objective does not guarantee uncertainty.

Masking is an established enabling technique, closely related to gesture-specific
[ModDrop](https://arxiv.org/abs/1501.00102). It is not our novelty claim. All main
admission comparisons must use the identical subset-trained teachers so any
benefit is not silently credited to this regularizer.

## 3. Compare changes in support, preserving their sign

Let N={P,A,G}. For proposed class c define stable one-versus-rest log odds

    ell_T,c(x,S) = z_T,c(x,S) - logsumexp_{d != c} z_T,d(x,S)

and a three-dimensional response

    r_T,c(x)_j = ell_T,c(x,N) - ell_T,c(x,N without j).

Four forward views suffice. Positive means that removing a group lowered c's
support; negative means that removing it increased support. Do not take absolute
values, clip away negative responses or require every coordinate to be positive.
Redundant useful information can yield a small change even when it matters.

An UP sample may have strong shape support and little incremental motion support.
A beckoning sample may have different support changes. These patterns are learned
from labels; there is no hardcoded rule that every UP is static or every FORWARD
requires a particular finger count. We do not require a partial view to predict c.

The profile is a small model diagnostic, not a semantic explanation certified
by physics. Feature-removal quantities depend on the probabilistic treatment of
missing information; see [Janzing et al.,2020](https://proceedings.mlr.press/v108/janzing20a.html).

## 4. Get reference profiles from another session group

Use two fixed groups of whole training sessions, each containing all eight
classes. Reveal five labels/class/group: L_A40+L_B40,80 unique labels total.
Train T_A on L_A only and T_B on L_B only; freshly initialized weights, no loading
of any538-label checkpoint. The first study has no teacher self-supervised stage.

T_A's anchors are L_B; T_B's anchors are L_A. Each anchor response must be
computed by that **same frozen teacher** that will evaluate a candidate. Do not
train a third full-L teacher and compare its profiles with the two earlier
teachers' profiles; their functions differ. All labels used for anchor selection,
scaling or thresholds count in the80. The two teachers together do not create
160 independent labels.

For teacher k and true class c, retain correctly predicted anchors with full-input
p_k(c)>=.5. This filter is provisional and must be reported: it can remove the
rare styles we need to learn. Fewer than three retained anchors makes that
teacher/class **unsupported**; admit no new targets through it. Unsupported is
not evidence that a gesture is wrong or belongs to NONE.

Two-session UP support permits this two-part construction, but not ordinary
eight-class nested session cross-validation. Session and class are confounded;
do not train an adversary to delete all session information and call it invariant.

## 5. A concrete initial compatibility rule

Compute a teacher-level scale on true-class responses from all its labeled
anchor rows:

    s_k,j = max(.05, 1.4826 * MAD_i(r_k,y_i(a_i)_j)).

The scale floor is in log-odds response units. For eligible same-class anchor
set A_k,c, use standardized squared distance

    D_k,c(u) = min_(a in A_k,c) sum_j ((r_k,c(u)_j-r_k,c(a)_j)/s_k,j)^2.

Use a union of neighborhoods rather than one class mean. This can represent
several observed UP styles but cannot invent support for a style absent from
the labeled anchors. A single mislabeled or unrepresentative anchor can mislead
the rule; nearest-neighbor matching is not a correctness oracle.

Provisional radius:

    R_k,c = max(.05^2,
                median_(a in A_k,c) min_(b in A_k,c, b!=a)
                   sum_j ((r_k,c(a)_j-r_k,c(b)_j)/s_k,j)^2).

R is a squared standardized-distance threshold, with a distinct dimensionless
floor. Require D<=R and define compatibility exp(-D/R); otherwise weight0.
These floors, median and minimum-anchor count are design defaults, not an optimal
or statistically certified choice. The resulting radius is **not** a95%
conformal, pseudo-label precision or robot-safety bound.

For unlabeled u, the initial conservative version requires both full-input
teachers to predict the same c with confidence>=.95, each supported anchor bank,
and both compatibility tests to pass. Use

    q(u) = [p_A(y|u,N)+p_B(y|u,N)]/2
    w(u) = min(exp(-D_A,c/R_A,c), exp(-D_B,c/R_B,c))

on accepted examples, zero otherwise. Full-teacher agreement is shared with
confidence-only controls; it is not the proposed contribution. Partial-view
agreement is never required. An empty admitted pool is a zero-coverage failure,
not perfect pseudo-label precision.

## 6. One teacher round, one ordinary student

Freeze teachers, anchors, scales, radii and targets for one round. Train a fresh
full-input student on the80 human labels and accepted soft targets:

    L_student = mean_(x,y in L) CE(y,p_student(x))
              + lambda(t) mean_(u in U) w(u) CE(q(u),p_student(u)).

U has325 primary-pool clips whose existing labels are hidden. Normalize the
unsupervised loss over the fixed325, with rejected weights0, so an almost-empty
admission pool does not silently receive full unsupervised loss strength.
Proposed lambda ramps0→1 over the first140 of560 student updates. Report the
effective sum of weights; controls must match label counts and include matching
pseudo-label count/weight-mass comparisons.

At inference only the student runs on a full window. No four-view teacher loop,
anchors or additional model are required in the camera application. This keeps
deployment size comparable; total offline training is likely higher and must
be measured. It is not yet an efficiency result. No camera model activation.

Self-supervised pretraining, learned routing, Shapley values, teacher refreshes,
graph encoders and consequence-weighted losses are deliberately excluded from
the first mechanism test. They require separate ablations if later justified.

## Pseudocode

```text
registry = eligible original training clips only
L_A, L_B, U = session-supported fixed budget split; U labels inaccessible
T_A = train_from_scratch(L_A, native_subset_CE)
T_B = train_from_scratch(L_B, native_subset_CE)
bank_A = responses(T_A, known_labeled_L_B)
bank_B = responses(T_B, known_labeled_L_A)
freeze(T_A, T_B, bank_A, bank_B, scales, radii)
for u in U:
    if full teachers disagree or lack confidence/support: continue
    if same-class response compatibility fails: continue
    save soft target and compatibility weight
student = train_from_scratch(L_A + L_B, weighted_saved_targets)
report every seed on predeclared existing evaluation cohorts
```

No hidden label, up_shape, task_hint, source-label metadata or current538-trained
embedding may leak into the learner. Balanced seed sampling is an oracle-assisted
benchmark construction, not an active-label-discovery achievement.

## A precise condition for cleaner pseudo-targets

This is an elementary identity, not a new theorem or a performance proof. On the
fixed confidence-eligible pool B let e_i indicate wrong pseudo-labels and w_i
be proposed weights in[0,1], with positive mean weight. Then

    R_base = mean_B(e)
    R_weighted = mean_B(w*e) / mean_B(w)
    R_weighted - R_base = Cov_B(w,e) / mean_B(w).

Thus our weighting improves pseudo-label error exactly when it is negatively
associated with errors on that pool. No independence assumption is needed for
this finite-pool algebra. Whether the required association actually holds is
the empirical question. Lower pseudo-label error also need not improve student
accuracy if only redundant easy examples remain or rare modes are excluded.

## Why this is not already proved or established novel

Confidence/consistency self-training is established by
[FixMatch](https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html).
[Few-shot Rationalization](https://aclanthology.org/2021.emnlp-main.836/) already
combines self-training with rationale supervision and sufficient/complete evidence
objectives. [ModDrop](https://arxiv.org/abs/1501.00102) already trains gesture
systems with missing modalities. [Efficient modality selection](https://jmlr.org/papers/v25/23-0439.html)
already studies modality utility. More close hand/SSL/robot papers and their
access limitations are in the dated prior-art note.

The narrower candidate difference to test is **same-teacher, cross-session,
class-conditional response-neighborhood admission**. It does not predict token
rationales or force removed-evidence entropy, and is not just a new name for
fusion or confidence thresholding. But a bounded search cannot establish that
this exact construction is unpublished; an adaptation may turn out incremental.

Fatal counterexamples and falsifiers:

- A confidently wrong sample can have exactly the same profile as correct anchors.
- STOP/UP may both depend on shape, yielding similar profiles despite different
  semantic shapes; three diagnostic numbers cannot identify class truth.
- Partial-input training may explain all gains; compare identical teachers.
- Radius matching may only rescale confidence or embedding novelty; use matched-
  confidence and nearest-embedding selectors.
- Rare gesture styles may be rejected; report perclass/style coverage.
- Teachers can share systematic errors; agreement does not certify them.
- Redundant position/motion channels can defeat removal-based diagnostics.
- If gains disappear at equal total compute, do not claim training efficiency.
- If no benefit remains at matched admission coverage, reject the stronger rule.

Eight constructed mathematical checks passed in
`experiments/runs/evidence_profile_design/20261016T010003871383Z`. They verify
signs, missingness ambiguity, channel partition, admission logic and the error
identity, and deliberately demonstrate the shared-error failure. No real gesture
accuracy, label savings, novelty, robustness or deployment result was measured.
