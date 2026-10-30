# Collection plan — pilot versus publication evidence

Updated 2026-10-07. The user aims for ICML, ICLR, or a strong Q1 venue.
This plan is provisional beyond the next single-user validation pilot.

## Current step — eight-command upgrade

The user now authorizes recycling the completed 90-clip validation session into
training in a separate eight-class version. The earlier six-label collection
instructions below are historical. Follow
[`COMMANDS_V2_WORKFLOW.md`](COMMANDS_V2_WORKFLOW.md): new UP/DOWN training (both
pointing and motion), then fresh validation covering all eight, then train/review/
freeze before collecting fresh test. Initial suggestions are 20 new train clips
per vertical label across two sittings, 80 fresh validation across two sittings,
and later 80 post-freeze test. Counts are diagnostic pilot quantities, not a
publication power calculation. The old six-class files/reports remain preserved.

## Earlier six-label pilot

Keep the six labels: FORWARD, BACKWARD, LEFT, RIGHT, STOP, NO_COMMAND.
Collect 30 NEW validation clips/class across three new sittings:
10/class/sitting × 6 classes × 3 sittings = 180 new clips.
Start with one sitting of 60 clips. Restart the application for each new session.
Use `python -m gesture_commands.app --split validation`.

These clips compare the frozen models from experiment 001; they are not additional
training data. The original 72 validation clips stay identifiable as their own
session. Report session-specific results rather than treating all sessions as
identical repetitions. This pilot does not establish cross-person robustness.

**Progress update, 2026-10-07:** the first new sitting contains 15/class (90),
not 10/class (60). All 90 were retained and evaluated against experiment 002's
20 frozen models. See [the finding](findings/002_new_session_01.md). Its weaker
results require clarifying FORWARD/BACKWARD/NO_COMMAND meanings before the next
collection. Two further sittings of 10/class would now give 210 new validation
clips, 35/class across three sittings; this supersedes the earlier 180 total.
There is no reason to discard the extra clips or treat either total as a
guarantee. Independent sessions and representative conditions matter.

These inspected sessions remain development validation. Do not move their clips
into training while reporting results on them as held out. Additional training
diversity needs separately designated training sessions and an explicit protocol.
All frozen models remain preserved; final evaluation requires new sessions and,
for the primary research scope, genuinely new people.

**After the user's style clarification:** the next diagnostic collection is a
separate **training** sitting, five clips/class (30 total), following
[`SESSION_DIVERSITY_PLAN.md`](SESSION_DIVERSITY_PLAN.md). This tests new style
coverage at matched training-label counts. It does not replace or absorb the
90 completed validation clips, and the added human labels are counted. Further
validation sittings remain pending. Keep existing command meanings while varying
legitimate angles/styles; do not make the evaluation set easier to fit the model.

## Why 30 is a pilot, not an accuracy certificate

Illustration calculated using the Wilson score formula in
[NIST's reference](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/agcoulci.htm):

| Independent evaluation clips | Correct | Observed accuracy | Approximate 95% Wilson interval |
| --- | ---: | ---: | ---: |
| 30 | 27 | 90% | 74.4%–96.5% |
| 100 | 90 | 90% | 82.6%–94.5% |
| 300 | 270 | 90% | 86.1%–92.9% |

These are illustrative intervals, not measured results from our next session.
The simple binomial calculation assumes independent observations; many clips
from the same person/session are clustered and cannot be treated as independent
evidence of performance across people. Comparing paired model differences also
needs a different analysis from estimating one model's accuracy. A final sample
size needs a practically meaningful effect/operating point and a variability
estimate from independent participants/sessions.

## Broader study — indicative proposal, not collection authorization or venue minimum

The user delegated scope selection and emphasized fewer labels plus strong test
accuracy. Working choice: unseen-user recognition is primary; few-label
personalization is a separately reported extension. The same command classes
are used in training/testing; unseen action classes are outside this claim.
See `RESEARCH_DIRECTION.md`. Actual split sizes remain to be finalized.

As an initial resource-planning example, 20–30 participants with three sessions
each and ten clips/class/session gives 3,600–5,400 total labeled clips across six
classes. This is not a power calculation, not necessarily sufficient, and not a
request to start recruiting or recording this amount now. All clips would need
explicit participant/session/condition provenance. The current app's session IDs
do not by themselves identify participants.

Allocate whole participants/sessions according to the chosen claim before
training. Retain completely untouched final evaluation people/sessions. For
personalization, permit only a fixed adaptation/support budget for a new person,
then evaluate separately on that person's other sessions. Never adapt on their
evaluation recordings. Obtain independent variation in lighting, background,
distance, speed, hand/finger style, and ordinary non-command motion.

Use continuous non-command recordings for false movement events per unit time;
more trimmed NO_COMMAND clips cannot substitute for that measurement. Determine
required duration from the false-command target and uncertainty, rather than a
universal fixed number of minutes.

For a general method contribution, propose at least two appropriate public
benchmarks in addition to the custom study; this is our comparison design, not
an official publication requirement. Identify their representations and split
rules before deciding whether MediaPipe-landmark processing is directly usable.
Public labels help evaluation costs, but simulated label withholding is different
from demonstrating reduced annotation time in deployment.

## What publication claims must establish

The official [ICML reviewer instructions](https://icml.cc/Conferences/2026/ReviewerInstructions)
evaluate soundness, presentation, significance, and originality; the
[ICLR reviewer guide](https://iclr.cc/Conferences/2026/ReviewerGuide) similarly
asks whether claims are supported and the work contributes useful new knowledge.
Our interpretation: a larger dataset or high single-user accuracy alone will
not demonstrate the intended contribution. Novel insights can also arise from a
rigorous comparison of existing methods, if their value and distinction from
related literature are established.

Compare strong baselines, multiple seeds, varied labeled/unlabeled budgets,
component ablations, participant/session uncertainty, and labeling/training
costs. Proposed training budgets: 5, 10, 20, and 40 labels/class where eligible
data permit, with the same evaluation protocol. These are training budgets,
not instructions to reduce evaluation labels. Target selection and paired
analysis must be fixed before final testing. No acceptance or novelty guarantee.
