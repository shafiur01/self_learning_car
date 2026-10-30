# Research direction — label-efficient temporal gesture learning

Updated 2026-10-20. Current result:
[018: online weighting and checkpoint sensitivity](findings/018_online_softmatch.md).
Prior source-access record: [017 literature review](findings/017_literature_grounded_decision.md).
Full experimental history stays in LOG.md and findings/. The previous version of
this direction document is archived with run017; its old collection requests and
“fitting unstarted” statements are historical, not current instructions.

## Main question

Can a training mechanism use fewer revealed command labels to improve temporal
hand-gesture recognition, while limiting erroneous movement commands at a fixed
compute budget? A credible publication contribution requires a distinct mechanism,
strong published baselines and independent replication. It has not been found yet.

We use pretrained MediaPipe landmarks, not an end-to-end image model. Report
upstream pretraining, all labels used for training/selection/calibration/review,
all failed runs and full compute costs. Withholding existing recorded labels is
simulated label efficiency, not demonstrated annotation-cost saving.

## What we have learned

- Motion features improved an earlier cohort, but later session results were
  mixed. Coordinate changes did not isolate a physical camera-viewpoint effect.
- 013/014 response-profile admission did not demonstrate an advantage over the
  confidence baseline; error-free pseudo-label selection alone was insufficient.
- 015 separates correct UP proposals rejected by confidence from weak NO_COMMAND
  targets; accepted profile targets were already correctly classified by the student.
- 016 standard normalization increased mean accuracy slightly while worsening
  NO_COMMAND-to-movement errors. Keep the raw149 LSTM as the reference.
- 017 reviewed primary ICML/ICLR/CoRL papers and replayed static selector components
  without fitting. More correct UP supervision also admitted more wrong targets.
  Published adaptive thresholds, soft weights, influence weighting and calibrated
  admission already overlap our proposed ingredients; novelty is unestablished.

## Current result and next controlled work

018 completed15 paired fits of supervised160, online confidence and a paper-based
SoftMatch adaptation. Selected-development macroF1 .8456/.8575/.8721; soft passes
the predeclared feasibility gate and reduces selected background movement errors.
At final560, F1 .8370/.8950/.8818 and soft STOP recall83.33% versus confidence90%.
Two soft selected checkpoints predate SSL. The ranking reversal and seed variance
prevent a robust superiority claim; preserve both baselines and all checkpoints.
Published-method adaptation is not a new algorithm or a full paper reproduction.

The audit verifies common initial/warm-up/labeled streams, exact reproduction of
five prior supervised states, all30 checkpoint outputs and134400online targets/
weights. No96rescoring. Fifteen fits took48.74s in fitloops including devchecks,
65.39s subprocesswall; additional research/audit cost remains separate. SSL uses
2.5x supervised training forward samples, so equal updates are not equal compute.

Next is a public multi-person temporal-gesture benchmark compatibility/split/
license/pretraining/compute audit before download/training. Freeze a defensible
checkpoint evaluation rule before further algorithm selection; do not choose a
new rule on the96 endpoint. Retain native labels/modality and independent people/
sessions. Keep the head-dropout/statistics adaptations explicit in comparisons.
No new risk/utility mechanism, rescue fit or budget/threshold sweep follows018.
More expensive meta-learning, architectures and self-supervised pretraining need
fresh evidence and a declared comparison. Target purity alone remains insufficient:
soft training had more erroneous target mass even where selected F1 improved.

## Evaluation and robotics scope

No more personal validation/test recording requests. No validation-to-training
transfer. Existing48/96 have already guided research and remain retrospective;
96 is not a further selection gate or an untouched final benchmark. Keep sessions
together. Current single-user evidence cannot establish unseen-user reliability.

A later public benchmark must retain native labels/modality, use whole people/
sessions as split units and freeze an uninspected endpoint before method choice.
Check access, licensing, metadata, sensor compatibility and costs before download.
No public data has been added through018.

Recognition and robot action decisions are separate. CoRL KnowNo motivates
reporting action coverage/help along with failures, not a guarantee for gestures.
Continuous false activation, response latency and task success need appropriate
streams/actions; isolated clips alone cannot measure them. Physical robot and
broad-domain claims (cars, drones, surgery) remain unsupported.

A supervised → semi/self-supervised sequence is a hypothesis, not the goal or a
known optimal order. Existing staged temporal learning, adaptive pseudo-labeling
and robot uncertainty work must be credited and compared. Record negative results
and change direction when evidence warrants it; do not chase a favorable seed.
