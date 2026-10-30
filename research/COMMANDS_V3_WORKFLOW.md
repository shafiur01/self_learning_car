# Gesture-definition repair — replacement UP and open-palm STOP

Prepared 2026-10-12. User clarified that intended STOP is an open palm, whereas
the previous written v2 contract incorrectly described a closed fist. User wants
a more distinct UP gesture and offered 40–50 new training examples. This is a
supervised command-interface revision, not a new learning algorithm.

## Current status

**V3 models are trained.** All5 seeds saved in
`experiments/runs/commands_v3/20261012T205727638067Z`, using538 training and48
checkpoint-selection validation clips. Mean accuracy85.83%, macro F1 .8611.
UP recall90%; STOP66.67% with same two missed clips in every seed, mostly FORWARD.
Read [finding006](findings/006_commands_v3.md) and [Lesson10](LESSON_10.md).
No automatic app activation or final v3 test. Next review shared STOP/angled-thumb
errors and F/B coverage; preserve all weights and labels. Collection instructions
below are historical and complete, not a request to repeat saved cohorts.

**Both UP sittings are now complete.** Audited training total538 clips:
FORWARD124/BACKWARD85/LEFT68/RIGHT59/STOP57/NO_COMMAND61/UP50/DOWN34.
Prompted new UP shapes17/17/16. All538 pass tracking checks, with no exact raw
duplicates. Do not rerun completed collection. V3 validation48/48 is complete
and the separate fitting/checkpoint-selection run has finished. Class counts
do not need exact equality; use class weighting and diagnose
missing styles using validation. These are single-user pilot counts.

**Prepared and ready to record.** User chose all three UP shapes: one index
finger up, index and middle fingers up, or thumbs-up, with remaining fingers
folded. Hold the deliberate pose for about two seconds. STOP is a still open
palm facing the camera with fingers extended. Broad open-palm upward movement
is no longer UP under this replacement definition. The other command meanings
are unchanged.

User's STOP-history report: **"yes almost all were open palm, no fists"**. Reuse
the 57 STOP examples based on this report, preserving the "almost all" qualifier
and marking that no independent per-clip shape review occurred. Do not claim
every old STOP pose was verified. If later review identifies exceptions, record
specific corrections explicitly rather than silently treating guesses as facts.

Seven synthetic checks passed after extending the collector to the three shapes.
The real v3 version was prepared at2026-10-12T17:24:40.563788+00:00:488 retained
training files,24 old UP excluded, no new UP saved yet. Plan SHA256:
`f03e2d06de995a2afedcb97356b4338c8d4033645c3e2fcedd2080fb05adac63`.
Both sittings have since completed; preparation/recording instructions below are
the historical workflow, not a request to repeat them.

## What is removed and reused

`experiments/commands_v3_collection.py` creates a separate
`experiments/data/commands_v3` development version. It excludes all old UP
training examples from that version; old versions, recordings, weights and
unfavorable results remain preserved. Old validation and test cohorts do not
enter the new selection/evaluation pool. Earlier explicitly authorized recycled
training examples retain that original provenance.

The source cohort is the routine condition's 512 training files from experiment
004, whose fixed models led on average in experiment005. This is a post-test
development choice, not an unbiased selection or new test comparison. None of
the temporal model weights are imported or activated. The classifier will need
a separate fresh fitting plan after collection and validation.

The 24 old UP files are excluded from this new version. The 57 STOP files are
retained with the actual human-report basis recorded in their new metadata.
The base has488 examples: F124/B85/L68/R59/S57/N61/U0/D34. With50 new UP clips,
the candidate training pool would total538. The collector also supports excluding
STOP on mixed/fist history, but that was not the history the user reported here.

The new schema/contract version is 3. Version 2 loaders reject it, so a new
meaning cannot silently be mixed with an old checkpoint. Migration stores old
hashes, original metadata and unchanged coordinates. Changing a text definition
does not change a network's learned weights.

## Collection now

The agent already prepared the chosen configuration with this command (historical
setup record; do not rerun against the prepared destination):

```sh
.venv/bin/python -m research.experiments.commands_v3_collection prepare --up-style three_shapes --stop-history open_palm --stop-history-notes 'yes almost all were open palm, no fists'
```

Then the user records 25 training clips in the first sitting:

```sh
.venv/bin/python -m research.experiments.commands_v3_collection record --sitting 1
```

Follow the specific shape in each prompt: one finger, two fingers or thumb.
The50-clip schedule has17 one-finger,17 two-finger and16 thumbs-up clips.
Sitting1 has9/8/8 respectively; sitting2 has8/9/8. All three are the same UP class,
not three new commands. Modest position/angle/distance variations are prompted
while the indicating finger(s) remain visibly upward. Keep the entire hand in
view. Perform a new clip for roughly two seconds, SPACE to replay, u to save,
x to repeat, q to pause. Only u saves, without model predictions. A restart
resumes the next saved task; resume only the same physical sitting. At 25/25,
press q, take a break and later record another 25 in sitting 2:

```sh
.venv/bin/python -m research.experiments.commands_v3_collection record --sitting 2
```

All 50 are training examples, not evaluation labels. The recorder keeps two
sittings together and records camera/replay/preparation/pause intervals; offline
and unsaved-repeat effort is not fully measured. It rejects raw duplicates,
historical session reuse and poorly tracked UP saves. No RGB videos are saved.
Status audits development coordinates/quality; it does not run predictions.

## Training and evidence afterward

This collector does not train or activate a model. After data audit, record fresh
development validation for all eight classes using the corrected meanings and
whole sessions outside gradients. Then define fitting/selection and train new
temporal weights under the v3 contract. Do not reuse inspected experiment005 as
an untouched test. A later honest test needs new data after choices freeze.

Fifty clips is a practical pilot collection target, not evidence of enough data
for reliable unseen-user recognition or a conference paper. Improved accuracy
after narrowing UP would partly reflect a changed gesture task. It would not
show that the old broad movement task was solved, that motion cannot be learned,
or that the proposed SSL sequence saves labels. Keep the earlier result and
STOP-definition discrepancy in the research account. See [Lesson9](LESSON_09.md).
