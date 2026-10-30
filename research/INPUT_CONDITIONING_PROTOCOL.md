# Experiment 016: feature conditioning and explicit-clock control

Declared 2026-10-17, America/Chicago, before fitting the new conditions. The
user authorizes investigating missed training/validation details. Earlier
camera/hand experiments changed coordinate representations, not independently
randomized physical viewpoints. Motion benefits were mixed on the later96.
No physical camera-viewpoint improvement is established.

Hypothesis: numerical scales of continuous feature blocks may hinder the small
LSTM and its teachers. A separate explicit-clock control tests dependence on
elapsed/sample-duration channels. Neither feature normalization nor clock
ablation is a novel algorithm. This strengthens/checks the baseline supporting
the broader label-efficient research question.

Use pilot014's exact405 original inputs and five160-label samples (20/class),
with245 labels hidden. No new data, corrections, validation-to-training transfer
or538-label warmstart. All commands and UPstyles already present. Same149-input,
49,840-parameter LSTM, fresh initial states,560 Adam updates,.001,batch32,
28earliest-minimum-CE checks on the same48 development examples. No SSL targets
enter student training in this experiment. Each seed fits native-subset teachers
A/B (80labels each, opposite whole-session groups) and a full-input supervised160
student. Same initial state/batch/mask streams within each role across conditions.

Fixed2x2 conditions: raw_clock, rms_clock, raw_no_clock, rms_no_clock.60fits total.
The original raw_clock students and native-subset teachers must exactly reproduce
all15 parent014 model-state hashes before evaluation. Preserve parent015 too.

RMS conditioning uses seven continuous blocks: poseXYZ[0:63], wristXY[63:65],
log-scale[65:66], pose-velocityXYZ[67:130], wrist-velocityXY[130:132],
scale-velocity[132:133], orientation[134:143]. One positive scalar per block:
sqrt(mean across usable training clips of mean(valid samples/channels squared))).
Use geometry-valid143 for position/orientation, motion-valid133 for velocities.
Exclude clips with no valid sample from that block's scale estimate. Floor .05;
if no usable clip, scale1. No mean subtraction, reflection, rotation, new clipping
or label change. Scalars preserve zero/missing values and vector directions.
Fit each teacher's scales only on its own80 revealed training inputs; student's
scales only on its160. Never use the other teacher's anchors, hidden inputs,
development or endpoint statistics. Archive fitting row IDs/scales before fits.

no_clock zeros only146-input columns144/145 (elapsed seconds/sample interval).
Visibility/motion/geometry masks66/133/143 and appended availability bits remain
intact. Physical-time derivatives remain; this is not speed/time invariance.
All feature extraction and original raw recordings remain unchanged.

All60 checkpoints, normalizers and settings freeze before one separate
retrospective evaluation. Score20 students on the same96 historically inspected
clips (already development material); no fresh-test claim. Score each teacher
on the opposite known80 anchors and on245 label-hidden training inputs; only
then use oracle labels diagnostically. Average teacher probabilities on those
same hidden inputs to examine UP/NONE recognition and .95 agreement admissions.
No profile/ranking/gate change or new pseudo-label student run. Charge160 training
+48 development labels=208, endpoint96 separately; shared pretraining/earlier work
are costs. Hidden labels reused for analysis are not new annotation savings.

Primary: mean paired student96 macroF1 rms_clock minus raw_clock over allfive
seeds; report both sittings, all classes, STOP/NONE errors, NLL and fit costs.
Secondary: clock removal with/without RMS, full factorial interaction, opposite-
session teacher macroF1 and hidden teacher recognition/admission counts. Record
every unfavorable condition; no winning seed/condition activated or new floor
or learning-rate sweep. Checkpoint-selected48 scores are development evidence.

Inspect final teacher LSTM preactivation gate saturation on opposite anchors:
sigmoid gates outside[.01,.99], candidate tanh magnitude>.99. Reconstruct gates
from input/previous hidden projections and verify the equations synthetically.
This is descriptive, not proof that saturation causes classification errors.
Block scales alone also do not prove a training problem.

Hash/archive inputs, sources, scales, label sampling, all checkpoints/histories
and outputs; use separate ledgers, preserve all previous data/runs. Tests cover
validity-aware scaling, zero/mask/direction preservation, training-only statistics,
gate reconstruction and incomplete-fit evaluation blocking. Existing .venv;
no dependencies, recording requests, lesson files or camera activation.

Technical motivation: LSTM gates use sigmoid/tanh of learned input/recurrent
projections, as specified in [PyTorch's LSTM documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
Different input scales are a hypothesis about optimization, not a theorem of
failure. [MediaPipe's coordinate documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results)
distinguishes normalized image landmarks from world coordinates; existing
normalized XYZ should not be treated as calibrated physical3D camera views.
No novelty, SOTA, independent generalization or robot reliability claim.
