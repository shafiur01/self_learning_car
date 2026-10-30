# V3 development validation — 48 fresh clips

**Completed:**48/48 saved in session20261012T203344-955c84cd; all5 v3 models
trained and verified. This guide preserves the collection protocol; do not repeat
the cohort. See [finding006](findings/006_commands_v3.md) and
[Lesson10](LESSON_10.md) for selected validation results/limitations.

Prepared 2026-10-12 before recording or fitting. This is an internal plan, not
public preregistration or a final benchmark for paper/robot reliability claims.

The538 training clips teach the network. These48 different clips help choose
saved weights and diagnose errors without entering weight updates. Because
validation selects checkpoints, its accuracy is development evidence; later
generalization claims need new test sessions after choices freeze. At preparation
no v3 model had been trained or activated. Fitting has since completed, without activation.

## Start a new physical sitting

Close the training camera window and take a break. In the project terminal run:

```sh
.venv/bin/python -m research.experiments.commands_v3_validation record
```

Follow the displayed prompt, perform it for about two seconds, press SPACE to
replay, then the displayed label key to save. x repeats; q pauses/finishes.
Only the current prompt's key saves. At48/48 press q. If interrupted, rerun the
same command to resume the next saved prompt within the same physical sitting.
No model predictions appear. Landmarks/timestamps/visibility and metadata are
saved, not RGB video.

## Schedule and meanings

Six examples/class, interleaved across six rounds of all8 labels:

- FORWARD: beckon toward yourself with comfortable one/multiple/open-hand forms.
- BACKWARD: deliberate outward push-away gesture, varying fingers/angle/distance.
- LEFT/RIGHT: point to the indicated side of the mirrored camera preview.
- STOP: still open palm facing the camera, fingers extended; no closed fist.
- NO_COMMAND: ordinary resting/repositioning, relaxed fingers downward, and one
  no-hand clip. Give no deliberate command on these prompts.
- UP: two one-index-finger, two index+middle-finger and two thumbs-up clips,
  remaining fingers folded, indicating finger(s) upward.
- DOWN: deliberate downward pointing and whole-hand downward movement.

Use natural modest angle/position/distance changes as prompted. Replay-check the
actual gesture before saving. Prompt metadata is not independent verification
of what was performed. Poorly tracked command clips must be repeated. Legitimate
no-hand NO_COMMAND remains included, with its movement-tracking failure reported
rather than excluded to inflate accuracy.

## Files and integrity

`experiments/commands_v3_validation.py` is separate because the earlier UP
collector is hash-bound and permits only UP training. It reuses the existing
camera/replay and v3 recording schema without modifying frozen sources.

Under `experiments/data/commands_v3/`:

- `reused_training/`, `new_training/`: existing538 training files, unchanged.
- `validation_plan.json`, `.sha256`: fixed meanings/prompts, training fingerprints,
  historical/training session and raw-content exclusions.
- `validation_source/`: archived collector/tests and reused core source.
- `validation/`: new recordings marked split=validation.
- `validation_attempts/`: camera/replay/preparation/pause timing intervals.

Plan prepared2026-10-12T18:47:55.482611+00:00, SHA256
`98ae9ddc30e6df51b5177a9fd1da604225fe8ac3c7722b20d0791eba1719e94b`.
No silent training additions/label changes during collection. Validation rejects
training/history session reuse, prediction-assisted labels, wrong meanings/roles,
raw duplicates and missing/repeated task slots. Software IDs/provenance do not
independently prove physical-session identity.

Six synthetic checks passed: role/shape schedule, incomplete-training refusal,
foreign-test metadata rejection before coordinates, session/model-assisted/raw
duplicate refusal, no-hand retention and unchanged training fingerprints, resumed
prompts and overlay restoration after success/failure. Synthetic accuracy is not
gesture evidence. No dependencies added and no camera model activated.

All48 labels count as development-evaluation annotation cost, not training
labels. Attempt timing includes replay/preparation/pauses and misses unlogged
offline/unsaved-repeat effort.

## After collection

Audit the full cohort, then implement the separate schema3 fitting/checkpoint
loader. Reuse the existing features/temporal network, class-weighted loss and a
recorded five-seed budget. Validation labels do not enter gradients. Keep every
seed outcome and old weights; no automatic app activation. Report class errors
and checkpoint-selection limitations, then freeze before a later fresh test.
Six/class from one user is a pilot, not evidence of broad reliability, label
savings or the proposed SSL algorithm. Old inspected tests are not reused.
