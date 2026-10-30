# Research workspace

[Repository map](../docs/REPOSITORY_LAYOUT.md) · [Commit/push guide](../docs/GITHUB_PUSH_GUIDE.md) · [Portable results](results/README.md)

**Git snapshot:** historical module paths are preserved. Personal notes and full
recording/run directories are local-only; links into them below require the
private archives. Each finding has a portable results link for GitHub readers.
No full historical rerun from exported tables alone is claimed.

Our objective is to measure gesture-recognition reliability per human labeling
minute and training cost, using the existing pretrained hand detector and small
temporal classifier. A publishable contribution is a long-term objective;
originality and success remain open questions.

**Current result:** [018: online weighting and checkpoint sensitivity](findings/018_online_softmatch.md)
completed15 fits on the same160/245 splits. Soft weighting passes the selected-
development gate (F1.8721 vs confidence.8575, lower background movement errors),
but final-checkpoint F1 reverses (.8818 vs.8950) and STOP recall worsens. Two soft
runs select supervised warm-up checkpoints. No robust superiority or new method
established. All30 states/1440vectors/134400online targets+weights verified;
9checks pass, parent controls reproduce exactly. No96rescoring. Fitloops48.74s,
workerwall65.39s; full costs/limits/seeds/failures saved. Next is a public benchmark
compatibility/split/compute audit and frozen evaluation rule, not a rescue sweep.
[Current research direction](RESEARCH_DIRECTION.md).

**Previous decision:** [017: literature-grounded review and zero-fit diagnostic](findings/017_literature_grounded_decision.md)
connects primary ICML/ICLR/CoRL methods with our negative/mixed findings. Static
adaptive/soft selector components recover correct UP targets but also amplify
wrong background targets; this is not full SoftMatch/FreeMatch reproduction or
new student accuracy. No fitting, inference or 48/96 payload access. Seven checks
and independent formula/parent-source verification pass. Its planned15-fit
comparison is now018 above; no new mechanism or threshold promoted.
Source access limits and costs recorded.

**Previous experiment:** [016: feature conditioning and clock control](findings/016_input_conditioning.md)
completed60 paired fits with the same160 labels. RMS raises accuracy86.25→87.08%
but worsens NO_COMMAND-to-movement28.33→43.33% and teacher recognition. Clock
removal alone keeps accuracy86.25%, with mixed seed/class outcomes. No condition
is promoted. All15 raw parent states match exactly;60models/17800scorevectors and
training-only scale statistics reproduced. No new data, lesson or activation.
[Fixed protocol](INPUT_CONDITIONING_PROTOCOL.md). Physical viewpoint gains remain
unestablished; standard normalization is not a novel algorithm.

**Previous diagnostic:** [015: coverage and ranking](findings/015_evidence_profile_diagnostic.md)
uses frozen models and hidden training inputs, with zero new fits or endpoint
rescoring. UP loses correct proposals at the .95 gate; NO_COMMAND teachers are
also weak. At fixed equal coverage, profile5/164 wrong labels versus confidence
7/164 and embedding1/164; at higher coverage confidence is stronger. All original
37 profile targets were already correctly recognized by the supervised student.
No reliable response-specific or causal training advantage established. Scores,
rankings and exclusions frozen/audited; no new lesson or recording request.

**Previous learning result:** [pilot 014: 160-label support](findings/014_evidence_profile_budget160.md)
completed all 65 fits with model/gate/training settings unchanged. Profile
accuracy 87.29% versus confidence 87.71%, embedding 88.54% and full405 93.33%.
It retained 37 seed–clip pairs (35 unique clips) with no observed label error,
but no UP/NO_COMMAND targets and no student advantage over the main baseline.
All eight classes had 20 supervised labels per seed; 245 labels stayed hidden.
48 development / 96 retrospective endpoint roles remain separate. Twelve
software checks and complete source/data/model reproduction pass. Parent
artifacts unchanged. [Frozen protocol](EVIDENCE_PROFILE_BUDGET_PROTOCOL.md).
No further run, recording request, lesson file or camera activation.

**Previous result:** the [80-label pilot](findings/013_evidence_profile.md)
did not beat confidence-only self-training: 74.38% versus 74.79% accuracy.
It retained only one pseudo-label across five runs. This motivates a controlled
support-budget test, not a claim of algorithmic success. Parent artifacts and
negative outcomes are preserved; no new lesson file or camera activation.
No novelty, useful improvement or actual label-saving claim is established.

The entries below preserve history; old collection/review instructions are not
new requests. Current work and constraints are in CONTEXT.md and finding018.

**Historical result:** [frozen models on two new sittings](findings/012_prospective_motion_sessions.md)
complete, explained in [Lesson16](LESSON_16.md). All96 clips jointly evaluated
against115 frozen checkpoints, no retraining. Camera-motion accuracy CNN93.33/
LSTM94.58/BiLSTM95.83/localset95/featureset92.92%. Earlier uniform motion gains
did not replicate: three families decline versus matched position inputs.
UP recall65–77%, NONE→motion remains nonzero. All12 STOP clips correct, not a
general stopping guarantee. One user/two sittings oneevening; pool now inspected
development. All11,040 vectors and384 raw-reextracted feature arrays verified.

Next review the failures and correct controls:

```bash
open research/experiments/prospective_motion_review.html
```

Start withNO_COMMAND andUP, watchbefore revealingguesses, save/exportobservations.
All96 clips/all23conditions/fiveseeds and89 nearby shape-reference replays.
19 browser checks pass. Original labels/weights unchanged; real humanreviewpending.
No further collection, refit or appactivation at this step.96 newlabels/31.25min
camera-replay-preparation-pause cost counted, notpureannotation orlabel savings.

**Previous:** [motion across models and controls](findings/011_cross_model_motion.md)
complete; [Lesson15](LESSON_15.md) explains why these controls matter. Same538/48
cohort,95 new fits +20 saved LSTM references. Camera-motion mean development
accuracy CNN99.17/LSTM97.92/BiLSTM99.58/local set98.75/feature set99.17%.
Strong averaging controls do not establish a recurrent-memory advantage; highest
BiLSTM accuracy includes an unwanted NONE→LEFT prediction. All115 checkpoints,
22,080 vectors and full failure tables verified/preserved. All scores are
checkpoint-selected development evidence, not fresh-test or robot reliability.

Audited all1,854 canonical files/942 raw recordings without rescoring old tests.
Legacy validation has90 clips already in training; oldUP differs from today's
gesture definition. Read the audit and caveats in finding011. The
[two-sitting plan](PROSPECTIVE_MOTION_PLAN.md) was executed as experiment012 above;
do not repeat the collection or one-time evaluation. No model activated.

**Previous experiment:** [explicit motion and hand coordinates](findings/010_motion_representation.md),
explained in [Lesson 14](LESSON_14.md), completed all20 fixed-budget trials.
Camera position/motion and hand position/motion mean inspected validation
accuracy84.17/97.92/91.67/92.08%. Camera motion has0 observed NONE→motion errors
and93.33% STOPrecall; hand motion has96.67% STOPrecall butloweroverallrecognition.
Same49,276-parameter LSTM, original538/48cohort, fivepairedseeds. No newly
collected labels. Prefix agreement probes are not validated early-command
accuracy or realrobot latency. New unseen sessions/people/speed/view evaluation
is required. [Protocol](MOTION_REPRESENTATION_PROTOCOL.md) preserves choices.
Narrative feedback and nine cumulative exports (found directly in Documents)
are saved separately: eight unique clips confirmed, original labels unchanged.
See [review reconciliation](notes/2026-10-15_review_reconciliation.md). No modelactivatedin camera.

**Review tool:** [development failure investigation and viewer](findings/009_failure_review.md)
is ready; the user watched failures and supplied narrative guidance. Run `open research/experiments/gesture_failure_review.html`
from the project terminal. Watch landmark animations by original category/model/
seed, compare training examples and successful controls, save observations and
export `gesture_review_notes.json`. [Lesson 13](LESSON_13.md) walks through it.
There are15 unique clips missed by at least one of25 models: F/B/L/R/S/N/U/D
2/3/0/1/2/5/2/0. No RGB video was saved. All labels/models remain unchanged;
exported observations are separate review evidence, not automatic relabeling.

**Previous result:** [experiment 008 temporal comparison](findings/008_temporal_architectures.md)
complete, explained in [Lesson 12](LESSON_12.md). CNN, pose set, feature set,
LSTM and BiLSTM trained with the same v3 538/48 cohort, five seeds and fixed
budget. Mean accuracy 85.83/85.83/85.42/89.58/88.75%. LSTM improves average
recognition but worsens NO_COMMAND→motion 13.33→33.33%; BiLSTM reaches 10% on
that error while STOP recall stays 66.67%. No overall reliability winner or
automatic activation. All 25 models and 1,200 vectors verified; no new labels.
See the [declared protocol](TEMPORAL_COMPARISON_PROTOCOL.md) and saved scientific
figure. Next investigate STOP/NONE development errors before further representation
selection and the separate label-efficient/audited-SSL study. New-session/person
evaluation is still needed; current scores are inspected development evidence.

**Previous step:** [temporal mechanism and algorithm plan](TEMPORAL_ALGORITHM_PLAN.md),
explained in [Lesson11](LESSON_11.md). Primary-source review plus
[fixed-weight motion probe](findings/007_motion_sensitivity.md) complete.
The original model is temporal Conv1d. Motion is input; learned
ordering reliance was weak/unproven in this fixed-weight probe. No new labels,
trained architecture, invented algorithm, test or app change occurred in that step. Separate same-task
representation comparisons from matched-budget audited-SSL research.

**Previous result:** [v3 supervised fitting](findings/006_commands_v3.md) complete,
explained in [Lesson10](LESSON_10.md). Five saved seeds,538 train/48 validation,
mean accuracy85.83%, macro F1 .8611. UP recall90%, STOP66.67% with shared
STOP→FORWARD errors. Validation selected these checkpoints; no untouched v3
test or app activation. Next review shared errors using existing data.

[Gesture-definition repair](COMMANDS_V3_WORKFLOW.md) collection is complete,
explained in [Lesson9](LESSON_09.md). Audited v3 training538 clips:
F124/B85/L68/R59/S57/N61/U50/D34. New UP allows one finger, two fingers or thumb
upward with remaining fingers folded. Both25-clip sittings are done; do not repeat.
Fresh validation completed before checkpoint-selecting training; the
[48-clip guide](V3_VALIDATION_PLAN.md) is retained as protocol. Six/class,48/48
complete. Base488 retained, old24 UP excluded; STOP57
reused based on user's "almost all open palm, no fists" report, not independent
verification. No model activation. Old data/results remain saved.

**Previous result:** [experiment005 fresh frozen test](findings/005_boundary_test.md)
is complete, explained in [Lesson8](LESSON_08.md). All15 saved candidates were
jointly tested on80 new same-user clips. Reference/routine/targeted accuracy
79/84.5/83%, macro F1 .7692/.8320/.8171; targeted advantage absent. NONE recall
improves34→86/86%, but UP is32/26/26% and mostly confused with STOP. Test data
are now inspected: do not repeat collection/evaluation of this pool or treat it
as untouched. No fitting or camera activation. Next proposed diagnostic is
UP/STOP development coverage and landmark motion; label-efficient SSL and
unseen-user/robot experiments remain later work. The frozen collection protocol
is retained in [the test plan](BOUNDARY_TEST_PLAN.md).

**Experiment004 is now complete:** routine/targeted additions average88.29%/
85.37% accuracy and macroF1 .8788/.8497 on the inspected41-clip monitoring set.
Expected targeted advantage absent; original reference86.34%/.8577. Both groups
repair the original shared resting-hand error, but other confusions/regressions
remain. Read [finding004](findings/004_boundary_data.md), [Lesson6](LESSON_06.md)
and its paired-seed plot. Its15 candidates now participate in experiment005.
Historical preparation and system state follow.

The completed training experiment used the
[class-matched additional-data comparison](BOUNDARY_DATA_PLAN.md).
Both added conditions received40 clips on the same472 base; all80 labels count
as collection effort. Fitting used original47 validation clips for selection and
second41 for separate monitoring. Existing system state follows.

Both sittings are complete and audited,40 each, with balanced groups and all
tracking checks passing. The collection instructions above are complete; do
not repeat collection simply to redo these same counts.

UP/DOWN each support pointing and hand movement. On user instruction, 90
inspected validation clips were copied into a separate v2 training pool, which
had 424 training clips initially. The new UP/DOWN collection adds 24 each, giving
472 training clips. Validation now has 88 clips in two sittings. All five models
are trained: first-session development accuracy 85.11%; second-session accuracy
86.34% on 41 new clips, with unchanged weights and no reselection. See
[first result](findings/003_commands_v2.md),
[second-session finding](findings/003_commands_v2_session_02.md), and
[Lesson5](LESSON_05.md). FORWARD recall64%, STOP80%, NO_COMMAND72%; all five models
confidently confuse the same resting NO_COMMAND hand with DOWN. User confirms
resting differs from a deliberate downward command; no labels changed.

The controlled comparison for resting versus deliberate DOWN and FORWARD/
BACKWARD confusion is prepared above. Keep these 88 validation clips
outside gradient updates and collect any new training examples separately.
The joint fresh test is now complete and inspected, and no camera model is
automatically activated. Old six-class recordings/checkpoints remain intact.
The linked workflow explains later final test. No additional second-session
recording is needed: that check is complete.

Start with [CONTEXT.md](CONTEXT.md) for project state, [PROTOCOL.md](PROTOCOL.md)
for the experiment plan, and [LOG.md](LOG.md) for decisions and results.
[LITERATURE.md](LITERATURE.md) distinguishes existing methods from our hypotheses.
For a beginner explanation of this first comparison, read [LESSON_01.md](LESSON_01.md).

First study completed: [finding 001](findings/001_augmentation.md). The initial
augmentation settings did not improve mean validation accuracy or macro F1.
The baseline remains the primary reference for the next comparison.

Working research scope and proposed learning mechanism:
[RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md). The primary target is label
efficiency on unseen people; personalization is a separate extension.

Recommended robotics question and its closest prior work:
[ROBOTICS_RESEARCH_IDEA.md](ROBOTICS_RESEARCH_IDEA.md). This proposal explains how
robot consequences could inform training-label review, how to test that claim,
and why simulation/task success cannot supply gesture meaning automatically.

Second study completed: [finding 002](findings/002_label_budget.md).
With 30/60/90/201 original training labels, mean validation accuracy was
78.06/85.28/90.56/93.33%. This is a simulated label-budget development comparison;
it does not establish actual annotation savings or robot reliability. Read
[LESSON_02.md](LESSON_02.md) to understand updates, withheld labels, and artifacts.

Run the second study from the project root:

```bash
python -m unittest research.experiments.test_label_budget -v
python -m research.experiments.label_budget
```

Its [configuration](experiments/configs/002_label_budget.json) fixes original-only
source eligibility, nested 5/10/15-per-class subsets, five seeds, 560 updates,
and matched validation opportunities. Every invocation preserves a new study;
it does not silently replace the camera model.

The first new validation session is now evaluated: 90 clips, 15/class.
Frozen model accuracy dropped to 58.00/60.44/67.56/70.67% for the same budgets.
See [new-session finding](findings/002_new_session_01.md) and
[Lesson 3](LESSON_03.md) for cross-session generalization and confident errors.
No retraining occurred; all recordings and weights remain intact.

The user reports varied angles and creative FORWARD/BACKWARD styles in that
session. The next [session/style diversity pilot](SESSION_DIVERSITY_PLAN.md)
compares original-only and mixed-session training at equal 60-label model
budgets. It needs 30 new training clips and records their additional human cost;
the comparison has not been implemented or trained.

```bash
python -m unittest research.experiments.test_evaluate_sessions -v
python -m research.experiments.evaluate_sessions
```

This strict evaluator identifies new validation sessions relative to study 002's
frozen manifest and saves a separate result under its `session_evaluations/`.
It compares every candidate and preserves the original training results.

## First experiment

From the project root:

```bash
source .venv/bin/activate
python -m unittest research.experiments.test_experiments -v
python -m research.experiments.run
```

The fixed [configuration](experiments/configs/001_augmentation.json) compares four
conditions over five paired seeds, with 80 epochs per trial. No new recordings or
labels are required for this first comparison. No camera is opened.

Each condition answers a specific question:

| Condition | Question |
| --- | --- |
| baseline | How does the current supervised procedure perform across seeds? |
| jitter | Does small, temporally smooth coordinate noise help? |
| time_warp | Do small changes of pace within a gesture help? |
| combined | Does applying both transformations help? |

The model, class weighting, Adam optimizer, learning rate, batch size, data, and
epoch budget are fixed. Each seed starts all four conditions from the same
weights, batch order, and dropout random stream. Transformation randomness has
its own generator. Validation receives no augmentation.

An **epoch** is one pass through the training set. A **seed** controls random
initialization and ordering. Repeating seeds checks whether a change depends on
one lucky training run. An **augmentation** is a transformed view of a recording;
it is not another independent example from a person.

## What is saved and why?

Each invocation creates a new directory under `experiments/runs/`:

| File | Purpose |
| --- | --- |
| `config.json` | Exact choices made before training |
| `manifest.json` | Recording identities, hashes, labels, sessions, and source counts |
| `development_features.npz` | Frozen train/validation arrays, independent of future data additions |
| `environment.json` | Package/platform versions, source hashes, Git state, and artifact hashes |
| `source/` | Actual experimental source and protocol used, including uncommitted code |
| `status.json` | Whether the study completed, failed, or was interrupted |
| `trials.json` | Every completed trial, including unfavorable outcomes |
| `summary.json` | Means, seed variability, and paired comparisons |
| `report.md` | Human-readable study summary |
| `CONDITION_seed_N/model.pt` | Compatible candidate checkpoint; not automatically activated |
| `CONDITION_seed_N/history.json` | Per-epoch training and validation losses |
| `CONDITION_seed_N/metrics.json` | Class-level metrics and confusion matrix |
| `CONDITION_seed_N/validation_predictions.json` | Exact predictions for auditing development errors |

Large run artifacts and copied data are Git-ignored. Durable context, protocols,
code, and written findings live outside that ignored directory. Preserve the
local run directories or back them up if full reproduction matters; Git alone
does not contain the recordings or weights. There is no upload in this workflow.

To reproduce the original input after recordings change, use the archived
`development_features.npz`, `manifest.json`, `config.json`, and source together.
The arrays have `train_x`, `train_y`, `validation_x`, and `validation_y` keys;
their row order follows the corresponding manifest records. The runner's
`train_trial()` accepts samples `(features, integer_label, file_name)` reconstructed
from those arrays. A normal rerun scans the current data and therefore creates a
new study rather than silently pretending to replay an old snapshot.

## Reading results

Accuracy counts correct clips. **Macro F1** gives every class equal importance
and combines precision with recall. **NLL**, negative log-likelihood, penalizes
confident incorrect predictions; lower is better. A percentage-point change is
an absolute difference: 90% to 92% is +2 points.

`NO_COMMAND → motion` counts raw clip predictions of movement classes on
NO_COMMAND clips: FORWARD/BACKWARD/LEFT/RIGHT in v1, plus UP/DOWN in v2.
It does not simulate the live gate or measure false starts per minute.
`STOP recall` measures the fraction of STOP clips recognized as STOP.

The standard deviation in a report is variability across training seeds on the
same validation session. It is not a confidence interval for new people or
sessions. Even a favorable pilot is only a reason to run the next independent
comparison. Consult the protocol before making claims or collecting more data.
