# Experiment 010 — explicit motion and hand coordinates

Declared 2026-10-15 before real feature extraction/fitting. User requests temporal
finger/hand-shape modelling, consistent references with command direction kept,
and investigation of early recognition across speeds/viewpoints/users.
Narrative feedback is available in chat; the named Documents/comments directory
was empty when checked. Do not claim per-clip exports imported without reading them.

## Finite first comparison

Keep the exact v3 538 training / 48 inspected validation cohort and human labels.
Use four prespecified representations in a 2×2 comparison:

| Condition | Local hand coordinates | Explicit derivatives |
| --- | --- | --- |
| camera_pose | Wrist-relative, camera-oriented | None |
| camera_motion | Same | Per-second joint/wrist/size changes |
| hand_pose | Same hand expressed in an anatomical basis | None |
| hand_motion | Same | Per-second joint/wrist/size changes |

All four share 3D estimated palm-length scaling, explicit palm orientation,
camera-oriented global wrist XY, apparent-size change, visibility, basis-valid
mask, elapsed seconds and sample interval. These are new feature version 2,
not the original 67-feature model. No camera-coordinate normalization or source
files from earlier frozen studies are overwritten. Current MediaPipe z is an
estimate relative to the wrist, not calibrated world depth or metric velocity.

Aspect-correct x/z, linearly interpolate visible coordinates to 32 timestamps
as before, center on wrist0, scale by max(norm(joint9−joint0 in xyz), .02).
Changing from the old projected xy scale is shared across the four conditions;
this study cannot separately attribute differences from the old model to that
scale change. The new camera_pose is not the old CNN/LSTM baseline.

For each sample, y-axis is wrist→middle MCP9; x-axis is index MCP5−little MCP17
orthogonalized against y; z=x×y, with a final orthogonal x=y×z. Basis columns
are camera-coordinate axes of the hand. Local row-vector shape is P×basis;
the camera pose can be reconstructed as local×basis-transpose before clipping.
Expose all nine basis entries in ALL four conditions, so command orientation is
not discarded. Camera wrist direction remains separate. A per-frame hand basis
may amplify tracker noise; basis validity/derivative masks and diagnostics count.
Mathematical rotation properties of ideal coordinates are not demonstrated
viewpoint robustness of MediaPipe measurements. No arbitrary rotations or flips
are declared label-preserving augmentations for directional commands.

Feature layout (32×146): shape63, wristXY2, log size1, visibility1;
velocity66, velocity-valid1; orientation9, basis-valid1; elapsed1, interval1.
Position conditions zero the velocity block/mask. Position/global channels clip
at ±10 and derivative channels at ±50 normalized units/second; report clipping.
No-hand inputs retain elapsed/interval only; geometry/basis/motion are undefined
and zero-masked. No silhouette or camera video is invented.

Differentiate unclipped sample shape+global signals using actual elapsed seconds.
A derivative is valid only when both samples have valid geometry and every raw
frame bracketing their interval is visible with no raw gap >.3s. Never difference
visibility or connect motion across missing tracking. Do not add acceleration,
smoothing, bone angles, graph attention or another architecture in this first
study; joint coordinates already permit learning relative relationships, and
explicit relationships require their own later controlled comparison.

## Shared temporal network and recipe

Reuse the LSTM/four-bin head pattern from experiment008 with 146 inputs and
47 hidden units, 49,276 parameters. Every condition has the exact same network,
initial weights and per-seed batch/dropout random streams. Zeroed channels in
pose-only conditions mean not every parameter receives informative input;
equal total counts do not make effective capacity or compute identical.
No claimed direct causal superiority over the earlier 62-unit/67-input LSTM.

Five seeds11/22/33/44/55; 560 Adam updates, .001, replacement batch32,
class-weighted training CE, unweighted validation CE every20 steps (28 checks),
earliest strict minimum. No augmentation, extra tuning or best-seed reporting.
Primary contrast: mean paired validation macro F1 camera_motion−camera_pose.
Secondary: hand_motion−hand_pose and hand−camera at each derivative setting.
Report STOP recall/false movement, NONE→motion/DOWN, F/B errors, all seeds and
actual fit time/forward latency. A higher average score with worse false commands
is not an overall reliability win. Save failed fits and all checkpoints.

## Exploratory prefixes, not early-action deployment

After full-clip fits, freeze all weights and evaluate inputs available within
.5,1.0,1.5 seconds of recorded clip start, plus the complete clip. Extract each
prefix independently, using no suffix for origin, scale, interpolation, basis,
derivatives or elapsed time. Retain observed durations/frame counts. Never reuse
full-clip features sliced after they incorporated later frames.

These models train on full clips; shorter inputs are distribution-shift probes.
Original full-clip labels may not be inferable from an early ambiguous prefix.
Report agreement with recorded full-clip intent, not validated early-command
accuracy. No confidence threshold, early-exit rule, prefix relabeling or unknown
class is tuned. Recorded starts are not independently annotated gesture onsets;
these cannot measure STOP latency from true onset, false movements/minute or
continuous command detection. Future continuous data need start/end annotations,
non-gesture intervals, explicit defer/coverage accounting and controller tests.

## Integrity, scope and relation to main research

Verify experiment008 first; match current snapshot/old arrays. Keep whole sessions
in their roles, no inspected validation enters gradients or unlabeled pretraining.
Archive raw fingerprints, feature arrays/config/source/masks, 20 models/histories,
full/prefix vectors and resource costs. Reload outputs and reproduce metrics,
minimum-loss selection and shared initial/batch states. No automatic app activation.

These are 538/48 already labeled single-user data, not new users, genuine speed
coverage, a final benchmark or measured annotation savings. Their duration range
does not provide a designed slow/normal/fast study. After developmental choices
freeze, genuinely new sessions/participants/viewpoints are needed. The main
label-efficient/audited learning policy remains a separate unimplemented study.

Existing ideas: [DD-Net](https://arxiv.org/pdf/1907.09658) models geometry and
temporal differences; [view-adaptive RNNs](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.html)
study viewpoint regulation and motion continuity. Hand coordinates/velocities
and an LSTM are not claimed inventions. Early decisions also require addressing
ambiguous partial observations, not merely processing a completed clip quickly.
