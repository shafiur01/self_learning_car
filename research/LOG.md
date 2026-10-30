# Research log


Entries are appended, including unsuccessful outcomes. Dates use America/Chicago.

## 2026-10-07 — experiment 001 planned

User authorized starting experiments and a persistent research workspace. The
long-term objective is a paper supported by evidence and a novelty review, not
an assumed universally best workflow.

Inventory: 334 training clips (three sessions; 201 original + 133 corrections),
72 validation clips (one session), and 52 historical test clips (one session).
Existing model comparisons used different training amounts; experiment 001
freezes the current data for all four conditions. No cross-split sessions found.

Plan saved before execution in `PROTOCOL.md` and
`experiments/configs/001_augmentation.json`: baseline, jitter, time warp, combined;
five paired seeds 11/22/33/44/55; 80 epochs. Augmentation hypotheses may fail.
This pilot uses validation only and does not demonstrate reduced labeling yet.

Implemented separate experiment package and artifact capture. Source copies are
needed because the project includes uncommitted work; a Git commit ID alone
cannot identify the executed implementation. Execution and findings will be
added after checks and training finish.

Research integrity checks passed: 11 tests, covering baseline equivalence to the
existing trainer, deterministic repeatability, monotone direction/endpoints,
visibility preservation, test-payload exclusion, split/duplicate leakage, metrics,
and complete artifact/checkpoint round-trip. These are software checks, not
recognition-accuracy evidence.

Initial related-work screening found Skeleton-Contrastive (2021), Hi-TRS (2022),
and SKETCH (2025). Abstracts inspected and recorded in `LITERATURE.md`; this is not
a completed novelty review or a full methods reading. The broad territory of
self-supervised skeleton learning and pretrained-backbone hand gestures already
has publications. A narrower contribution needs explicit comparison.

## 2026-10-07 — experiment 001 completed

Study ID: `20261007T050900637997Z`; completed 20/20 trials, 80 epochs each.
Frozen inputs: 334 train and 72 validation clips. All development recordings passed
the movement-tracking quality check. No test predictions used.

Mean validation accuracy: baseline 96.67%, jitter 96.39%, time warp 96.11%,
combined 96.67%. Mean macro F1: 0.9647 / 0.9623 / 0.9586 / 0.9631 respectively.
No mean accuracy or macro F1 improvement from these augmentation settings.
Jitter's lower raw NO_COMMAND-to-motion rate is exploratory on only 11 clips.
Results and reproduction hashes: `findings/001_augmentation.md`.

Decision: retain the unaugmented baseline as primary reference; do not promote
the highest-scoring individual seed. Next collect a new validation session in a
separate sitting, starting with five clips/class, and compare frozen candidates
by session. Later add session-aware label-budget curves. Novelty review,
unlabeled data collection, and semi-/self-supervised experiments are pending.

No active camera checkpoint changed. All previous recordings/models preserved.
Research run artifacts are locally stored and Git-ignored; durable summaries and
context are outside that ignored directory. Root `AGENTS.md` directs future work
to read/update these continuity files. Nothing was committed or published.

## 2026-10-07 — collection recommendation clarified

The user reaffirmed ICML/ICLR/strong Q1 publication ambitions and asked about
sample counts. Five clips/class was a quick-check minimum. The current pilot
recommendation is 30 NEW validation clips/class across three separate sittings
(10/class/sitting, six classes, 180 new clips), as already explained in chat.
This supersedes the five/class next-step recommendation without altering the
completed experiment's protocol or results. No new recordings are claimed here.

Official ICML/ICLR review guidance and NIST's Wilson formula were checked.
Illustrative independent-clip intervals were calculated: 27/30 yields
74.4%–96.5%; 90/100 yields 82.6%–94.5%; 270/300 yields 86.1%–92.9% at 95%.
These are not our empirical findings and ignore participant/session clustering.

`COLLECTION_PLAN.md` separates the next pilot from a broader indicative
20–30-participant resource plan. Those larger numbers are proposals, not a
publication requirement, powered design, or instruction to collect immediately.
Research scope (unseen-user recognition, few-label personalization, or both)
was requested and is pending. Participant metadata, final split design, strong
public benchmark comparisons, and continuous-video evaluation remain to plan.

## 2026-10-07 — scope delegation answered; working direction selected

The user delegated choosing the setting and clarified that the main ambition
is a mechanism producing strong test results with fewer labeled training data.
Working decision: unseen-user recognition of the same gesture classes is primary;
few-label user personalization is a separately evaluated extension. Test-user
recordings cannot enter primary pretraining or semi-supervised training. Single-
user data remain exploratory. The earlier clarification is no longer pending.

Created `RESEARCH_DIRECTION.md`, updated context/protocol/collection guidance,
and extended literature notes. TCL (CVPR 2021) method Section 3.3 and results
were inspected using the author's accessible copy after CVF returned 403.
FreeMatch, USB, and SHREC 2021 primary abstracts were screened; their full
methods/dataset checks remain pending. There is close published staged work;
this is not an established novelty or SOTA finding.

Proposed stages: pretrained visual features; optional self-supervised temporal
pretraining; fixed small-label supervision; supervised plus unlabeled consistency;
budgeted active review/auditing; optional distillation. These are hypotheses and
comparators, not newly implemented stages. Next algorithmic comparison: supervised
5/10/20-label-per-class curves where eligible original counts permit. Preserve
correction-selection provenance and upstream costs. The 180-new-validation-clip
pilot plan remains applicable. No extra training or evaluation was run here.

## 2026-10-07 — robotics main-idea review and working recommendation

The user requested a core robotics/ML/algorithm/sequence research idea, backed
by genuine evidence, with detailed planning to follow. Conducted a focused
primary-source review in parallel across robot feedback learning, uncertainty/
risk calibration, and robotics/gesture evaluation. Created
`ROBOTICS_RESEARCH_IDEA.md` and linked/updated continuity documents.

Recommended for discussion: consequence-aware audited semi-supervised learning
for temporal human-command interfaces, initially an indoor mobile assistant.
Candidate contribution lies in budgeted training-data audit/admission and
consequential confident errors, rather than a generic sequence of known stages.
Human intent and physically safe action are distinct; a safe wrong action is
not a correct pseudo-label. Low current risk must not loosen semantic validity.

Novelty objections recorded: SESAR already combines skeleton representation
learning and active selection; ThriftyDAgger/Risk-Aware Active IRL already query
according to robot consequences; CSST already uses cost-sensitive self-training;
AdaConG already calibrates guidance inside training; RCIP/KnowNo already manage
intent/action uncertainty and help decisions. Robot active/SSL perception and
incremental gesture semantics also have close predecessors. No confirmed gap,
new theorem, SOTA claim, or publication readiness is established.

Proposed evidence separates different trained models with one fixed execution
rule from one fixed model with different rules. Include simple random and
consequence-stratified review baselines; count all human labels/minutes and
compute. A standard random audit could bound frozen-pool contamination, but not
future robot safety; repeated adaptation and small audit budgets limit it.

Updated benchmark knowledge: revised SHREC 2021 removes POINTING (17 classes),
and its paper describes train/test people separated. FR-GESTURE (2026) supplies
static RGBD UGV commands, not temporal hand-motion data. No benchmark downloaded.
Public benchmark adapters, unlabeled/event recording, participant metadata,
robot states/controller/simulation, and independent final evaluations remain
unimplemented. Current pilot/label-budget next steps and unseen-user isolation
remain applicable. No training, installation, robot actuation, checkpoint
activation, recording modification, commit, or publication occurred here.

Technical review clarified the candidate operational rule: separately allocate
targeted acquisition and a uniform frozen-pool audit; admit pseudo-label batches
only against prespecified semantic and consequence error ceilings. Human-label
ambiguity, changed residual pools, stress strata, and repeated rounds require
explicit treatment. These are proposal refinements, not implemented guarantees.

## 2026-10-07 — experiment 002 planned before execution

The user said to start. First concrete step: supervised label-budget reference
before implementing the robotics-aware audit mechanism. Snapshot still contains
334 training / 72 validation clips; the 180-new-validation-clip collection has
not appeared. Original training counts are 61/38/31/26/29/16 in label order,
from one session, with no exact raw duplicates. Corrections comprise 133 clips.

Use 5/10/15 original labels/class plus all 201 originals, paired seeds
11/22/33/44/55, 560 optimizer updates and 28 validation checks per trial. The
earlier conditional 20/class budget is unavailable with only 16 original
NO_COMMAND examples. Correction pool excluded from this primary study; its
prior model-selection costs remain recorded. All recording files preserved.

Created fixed-update experiment runner/configuration and beginner lesson.
Protocol records sampling, class weighting, nested/session-aware selection,
shared validation, provenance, and limits. Software verification and real-data
execution pending at this entry. No new dependencies needed; matplotlib is
already present in the existing .venv for standalone plots.

Experiment 002 verification passed: 16 research checks (5 new budget/artifact
checks plus 11 existing research checks). Synthetic fixtures verify nested
session-aware subsets, correction exclusion, insufficient original budgets,
fixed update/checkpoint opportunities, deterministic repeats, test-payload
exclusion, and predictions regenerated from saved checkpoints. These checks
establish software behavior, not recognition accuracy. Real trials start next.

## 2026-10-07 — experiment 002 completed

UTC study ID `20261008T005037258990Z`, all 20/20 real-data trials completed.
Original pool 201, corrections excluded 133, validation 72 in one session.
Frozen design: 30/60/90 labels plus all-original reference; five paired nested
subset/model seeds; 560 updates with batch 32/replacement sampling, 28 validation
checks, no augmentation. All checkpoints, subsets, predictions, and source/input
hashes archived. No test predictions or application checkpoint activation.

Mean accuracy: 78.06/85.28/90.56/93.33%; macro F1:
0.7620/0.8442/0.9012/0.9297. Raw NO_COMMAND-to-motion mean:
29.09/18.18/10.91/10.91%; STOP recall 68/84/96/100%. These reuse eleven
NO_COMMAND and ten STOP validation clips across models. They are not independent
population rates, executed motion, or physical stop tests. Lower-label errors,
ties in individual seeds, and absence of raw false-motion improvement from
90 to 201 labels are retained. Five/class is weak in this supervised comparison.

Exploratory frozen-prediction inspection found one STOP-to-motion prediction
scoring >=0.95 under five/class; none under other budgets. All-original
NO_COMMAND-to-motion errors scored below 0.8, so raw error rates cannot be
presented as failures of the live command gate. No continuous replay was run.
Withheld-label comparisons are not measured annotation savings. Validation
selection and single-user dependence preclude a final benchmark claim.

Created durable `findings/002_label_budget.md` with all seeds, limits, hashes,
and artifact links; plot inspected visually. Updated context/direction/README.
Next: first new validation sitting, ten clips/class (60); two further separate
sittings for 180 total. Then evaluate frozen models by session and define the
matched-budget SSL/controller/auditing protocols. These robotics stages remain
pending. Prior recordings/checkpoints preserved; no dependencies/commit/upload.

## 2026-10-07 — first new-session evaluation of experiment 002

User reported 15 clips per gesture. Inventory confirms 90 original validation
clips, exactly 15 per six labels, in new session `20261008T005936-27dde0e5`.
All pass existing tracking quality; no exact duplicate raw contents within the
new cohort or with previous development recordings. All 406 old development
recording hashes match. Current inventory: train 334, validation 162/2 sessions.
Preserved all clips rather than reducing the sitting to the suggested 60.

Recorded an evaluation addendum before new inference. Implemented
`experiments/evaluate_sessions.py` with new-session/content/provenance guards,
separate archived evaluation inputs/source/predictions, and no optimizer or
activation. Five new substantive integrity checks passed before execution.
Runtime checks reproduced old prediction vectors for every frozen checkpoint
and confirmed all old/new development files and weights stayed unchanged.

All 20 frozen models evaluated in `20261008T013739665513Z`, beneath study
`20261008T005037258990Z/session_evaluations/`. Accuracy means for 30/60/90/201
training labels: 58.00/60.44/67.56/70.67%; macro F1:
0.5558/0.5840/0.6657/0.6942. All-original mean dropped 22.67 pp from old-session
93.33%; old and new results are reported separately, without checkpoint selection.
Primary paired macro F1 differences versus all-original:
−0.1383/−0.1102/−0.0285 for small budgets.

All-original recalls FORWARD/BACKWARD/LEFT/RIGHT/STOP/NO_COMMAND:
52/81.33/90.67/93.33/82.67/24%. FORWARD errors become BACKWARD. Mean raw
NO_COMMAND-to-motion rates by budget: 80/80/70.67/76%, on the same 15 clips
per model. All-original STOP-to-motion counts by seed: 4/2/2/2/2 out of 15.
Exploratory score inspection after primary evaluation: 50 repeated model–clip
STOP/NO_COMMAND-to-motion predictions score >=0.8 and 41 >=0.95 under the
all-original condition. No live-gate, continuous-stream, or robot evaluation.
Do not interpret repeated predictions as independent events or adjust thresholds
to make this inspected session look good. No causes or fixes established.

Asked the user to describe FORWARD/BACKWARD/NO_COMMAND and whether hand/styles
matched the original recordings; reply pending. Numerical quality cannot verify
semantic labels, and RGB is not stored. Created durable finding and Lesson 3;
updated continuity/README/collection plan. Remaining two 10/class sittings would
give 210 new validation clips; this is a provisional pilot quantity, not a
powered publication sample. Extra training diversity must use separate training
sessions. SSL/controller/audit comparisons still need fixed protocols and code.
Current inspected sessions are development data, not a final untouched test.
No retraining, new dependencies, app activation, commit, upload, or paper claim.

Post-execution artifact verification passed: archived evaluation input/source
hashes, all 20 checkpoint hashes, complete status, and accuracy regenerated from
all 1,800 saved model–clip predictions. `git diff --check` passed. No failed
real evaluation runs occurred; the unfavorable accuracy findings are retained.

## 2026-10-07 — user describes varied gesture styles

User reply: slightly varied hand angles, more precise directions, creative
FORWARD/BACKWARD motions somewhat different from training; also noted less
NO_COMMAND/STOP training data. Verified study-002 original counts:
FORWARD 61, BACKWARD 38, LEFT 31, RIGHT 26, STOP 29, NO_COMMAND 16. All small
conditions are class-balanced. Existing loss weighting compensates class
importance but supplies no new gesture styles. No causal diagnosis established.
Handedness and exact individual styles remain unknown; no recordings relabeled.

Recorded `SESSION_DIVERSITY_PLAN.md`: next diagnostic is original 10/class vs
original 5/class plus new varied-style 5/class at equal 60 training labels.
Needs 30 new human-labeled clips in a separate training sitting, six classes.
Plan reuses frozen original-only models and original nested subsets, with the
same initialization seeds, updates, sampling, and original 72-clip checkpoint
validation. The new 90 clips remain separate development evaluation. New
collection is targeted following inspected errors, so provenance and incremental
human effort must be recorded. This tests session/style mixture jointly rather
than proving an angle effect. No additional models or implementation executed.

Maintained existing gesture meanings, including closed-fist STOP; ordinary
non-command visible-hand motion belongs in NO_COMMAND. Arbitrary rotation/mirror
augmentation is not assumed label-preserving. Final unseen-user/robot/label-
savings claims remain unsupported. Remaining validation sittings are pending.

## 2026-10-07 — eight-command upgrade prepared; inspected validation retired in v2

User explicitly asked to retrain using the new validation data, obtain new
validation/test, and add UP/DOWN. Clarifications: both pointing and whole-hand
movement, varied indications; future DOWN actions could be drone descent or
humanoid sitting. Perception labels retained separately from robot-specific
action mapping. No controller/motor connection or physical outcome claim.

This supersedes the pending six-class 60-label-mixture pilot as immediate work.
Recorded protocol before real preparation. Added `experiments/commands_v2.py`
with a separate eight-class schema/contract, migration, camera profile adapter,
fixed-update multi-seed training, candidate freeze and fresh-test ledger.
Reused legacy features/GestureNet, replacing final layer 6→8 (48,520 parameters).
Old data/model sources/checkpoints unchanged. Shared camera main accepts a
profile explicitly, with legacy default behavior preserved. UP/DOWN error/gate
logic uses explicit movement labels, avoiding six-class index assumptions.

Nine new v2 integrity checks and 14 legacy pipeline checks passed. Following
test-preflight/ledger improvements, the nine v2 checks passed again. Checks use
synthetic data/control camera fixtures, not real recognition outcomes. Includes
old loader rejecting eight-class weights, copied validation provenance, preserved
source files, same-session/duplicate guards, training excluding test payload,
saved prediction reload, post-freeze-only test and repeat-test rejection.

Actual preparation completed: `experiments/data/commands_v2/` has 424 train
copies, 334 legacy + all 90 from `20261008T005936-27dde0e5`. New counts in label
order: 114/75/68/59/57/51/0/0. All 424 arrays match original xyz/times/visibility
exactly; original source hashes match migration; no raw development duplicates.
V1 original split metadata is untouched. V2 copies record origin hashes,
original metadata/roles and `recycled_inspected_validation` acquisition.
Old 72 validation and 52 historical test are excluded from fresh v2 evaluation.
New validation/test both empty; no real v2 checkpoint exists. Training preflight
refused missing UP/DOWN (0/5 each) before allocating a real study or optimizer.

Created `COMMANDS_V2_WORKFLOW.md` with stepwise collection/retraining/freeze/test
instructions and teaching explanations. Next: 10 UP + 10 DOWN training clips,
half pointing/half moving per class, then another sitting for 20/class; fresh
validation for all eight, preferably 2×5/class (80), before training. Final test
only after choices freeze. These are pilot suggestions, not publication power
calculations. All added labels/reviews and inherited correction costs count.
No accuracy improvement, SSL, staged-learning advantage, label savings, novelty,
unseen-user or robot reliability result is claimed. No dependencies/commit/upload.

## 2026-10-07 — 48 UP/DOWN training recordings audited

User: "yes done with labelling for now". Inventory finds UP 24, DOWN 24, all
training/original recording source in `20261008T021615-a0b59124`. Full v2 train
total 472; labels 114/75/68/59/57/51/24/24. New clips pass tracking quality,
duration 1.731748–1.999981 seconds. Imported copies still match migration hashes;
no exact raw development duplicates. These are one additional training sitting,
not the two suggested sittings. Retain all 48, with no forced reduction or extra
vertical collection merely to match a quota. Hand reports unspecified/notes
empty, so style coverage cannot be confirmed from provenance.

Saved collection audit `experiments/data/commands_v2/collection_audits/
20261008T023708128618Z.json`, including all new input hashes/quality measurements.
Fresh validation and test both empty; no new weights or accuracy result produced.
Next: first independent validation sitting of five/class across all eight
labels (40), then another sitting for the proposed 80 total before retraining.
Keep final test genuinely new and after choices freeze. Updated continuity and
workflow guidance. No code/dependencies/checkpoints changed in this turn.

## 2026-10-07 — LEFT correction applied; first v2 training starting

User identifies `e38cd85291d840689e82676107d709f6.npz` as LEFT, mistakenly
marked FORWARD, and says the rest of the validation data are reviewed. Located
in v2's fresh validation session `20261008T024032-d194a121`. Corrected the
label on explicit user authority, kept an exact original file under
`label_revisions/20261008T025236017041Z/`, and recorded before/after file hashes
and metadata label history. All arrays, raw-content identity, split and session
are unchanged. This is label review, not a model-driven relabeling operation.

Inventory after correction: train 472; validation 47 in one session, class
counts 6/6/7/6/6/6/5/5; test 0. All fresh validation recordings pass tracking
quality. No development duplicates/session leakage; all imported copies still
match migration hashes. Actual counts meet the fixed minimum five/class.
Begin initial five-seed/560-update run, keeping the originally suggested second
validation sitting pending rather than requiring extra clips before any pilot.
Protocol execution addendum records this one-session limitation before fitting.
No extra dependencies or algorithm changes; real outcomes pending.

## 2026-10-07 — first five-seed eight-command training completed

Study `experiments/runs/commands_v2/20261008T025353704748Z`, all five trials
complete. Same fixed recipe: 472 train, 47 validation/one session; weighted CE,
Adam .001, 560 full batch-32 replacement updates, no augmentation, 28 checks.
All candidate weights start fresh and are saved separately. No automatic
camera model activation, six-class checkpoint replacement, or final freeze.

Mean validation accuracy 85.11% ±2.61 pp run SD; macro F1 0.8532 ±.0288; NLL
.3380. Seeds 11/22/33/44/55: 38/40/41/40/41 correct out of 47; saved steps
260/560/280/560/460. Mean class recalls F/B/L/R/STOP/NONE/UP/DOWN:
80/73.33/100/93.33/96.67/53.33/84/100%. LEFT support seven and UP/DOWN five
each, so perfect small-pool recall does not establish general reliability.

Raw NO_COMMAND-to-motion counts 4/2/3/3/2 of six, mean46.67%. Across the five
models' shared pools: NONE→BACKWARD12, NONE→UP1, NONE→FORWARD1; STOP→UP1.
Exploratory post-fit score inspection found one false-motion score ≥.8 and ≥.95,
max .9829. No margin/repeat gate or continuous stream was simulated; counts are
repeated model–clip predictions, not independent physical events. Validation
selected saved steps; no final-test or robot/SSL/savings/novelty result.

Post-fit verification passed: source/input hashes, all 519 development files,
label correction in frozen manifest, schema/session provenance, and all 235
prediction vectors regenerated from the saved weights. Training array shape
472×32×67 and validation47×32×67; no test payload included. Saved verification
record includes all checkpoint hashes. All unfavorable outcomes are retained.
Created durable finding003 and Lesson04; updated continuity/workflow.

Next: second new eight-label validation sitting (five/class,40), compare the
already saved candidates without retraining/reselection on it, then decide
whether to freeze or develop further. Final test collection stays after choices
freeze. No new source/dependency changes, upload or commit in this execution.

## 2026-10-08 — second v2 validation sitting evaluated without fitting

User reports second independent validation complete. Found 41 new original
validation recordings in `20261008T050854-92265627`, five/all eight classes
except RIGHT six. All pass tracking checks; no raw duplicates/split leakage;
all old519 study inputs remain unchanged. Current train472/five sessions,
validation88/two sessions, test0. Retained all41 rather than reducing to a quota.

Recorded evaluation protocol before predictions. Added separate helper
`experiments/evaluate_v2_sessions.py` and six meaningful software checks, all
passing. Checks cover cohort freshness, unchanged files/contract, class counts,
tracking failures retained, separate old/new aggregation, source/checkpoint
hashes and synthetic saved-weight inference with optimizer use forbidden.
Original training source remains unchanged for reproducibility/future freeze.

Evaluation `20261008T052213777935Z` under study
`commands_v2/20261008T025353704748Z/session_evaluations/`: all five saved seeds
complete, with no updates, checkpoint reselection, gate tuning or activation.
Accuracies90.24/90.24/82.93/87.80/80.49%, correct37/37/34/36/33 of41;
mean86.34% ±4.43pp run SD. Macro F1 .8577 ±.0447; NLL .4690 vs old .3380.
Mean recalls F/B/L/R/STOP/NONE/UP/DOWN64/96/96/96.67/80/72/84/100%.
NO_COMMAND→motion mean28% (counts2/1/1/1/2 of5); STOP→motion0/1/1/1/1.
Mean accuracy is1.24pp above the old session, but fixed weights imply no learning
improvement. FORWARD/STOP recall is worse and seed rankings changed. Preserve
all results separately; the205 model–clip predictions share41 human recordings.

Exploratory score inspection: all five call NO_COMMAND clip
`7601353c641440f9bca0b5c65cfed650.npz` DOWN with scores .9961–.99999.
Asked about ordinary lowering. User says the hand was lowered/kept naturally
resting; DOWN normally means properly pointing downward, and resting is not DOWN.
Preserved NO_COMMAND and all recordings. Earlier deliberate whole-hand forms
remain supported; no contract change. This is a shared confident error, not
proof of the learned cue or cause. DOWN mean precision80.95% despite recall100%.
No continuous gate/robot false-start outcome measured. A high score or model
consensus does not establish a trustworthy pseudo-label.

Verified artifact/source/checkpoint hashes,41 new input files and205 predictions;
all235 original score vectors reproduced. Saved verification.json, separate
finding003_session_02 and Lesson05; updated context/protocol/workflow/direction.
Next design a controlled training-data comparison for resting/intentional DOWN
and track FORWARD/BACKWARD errors, using separate training sessions and counting
all effort. Both88 validation clips stay outside gradient updates. No new fit,
comparison execution or final freeze. Test remains uncollected; unseen-user,
robot reliability, SSL, label savings, novelty and publication claims unsupported.
No dependency, commit or upload introduced.

## 2026-10-08 — experiment 004 guided collection prepared

User requests starting the next step. Read context/protocol/latest log and
prepared `BOUNDARY_DATA_PLAN.md` before new collection or fitting. Question:
same 472 base + 40 routine clips versus same 472 + 40 targeted clips, with
the saved five 472-label models as a smaller-budget reference. Each added
condition has 10 F/B/NO_COMMAND/DOWN; total new acquisition is 80 human labels.
Two new physical sittings each supply 5/class/group (40). Both groups occur in
both sittings; reverse group order in sitting 2. This is a bundled instruction/
coverage comparison, not random acquisition or a new algorithm. All effort counts.

Fixed intended fitting policy: existing model/features, five fresh paired seeds,
560 updates/batch 32/weighted CE/Adam .001/no augmentation/28 checks. Only the
original 47 validation clips select checkpoints; the inspected second 41 supply
separate development monitoring. Neither enters gradients. Primary comparison
mean macro F1 targeted-minus-routine on the second 41; report every seed, all
class precision/recall, STOP regression, NO_COMMAND false-motion, F↔B and costs.
Align base and appended class/sitting/repeat rows. Fitting runner follows the
new-data audit; no fitting or new accuracy result occurred in this preparation.

Added `experiments/boundary_pilot.py`: isolated preparation/recording/status,
counterbalanced prompts, prompted-key confirmation, prefix/duplicate/session/
hash guards, resumable progress, visible-hand NO_COMMAND guidance and attempt
interval logging. Reuses the existing camera/replay and v2 save format; scoped
overlay adapter restores on all exits, without editing old app/model sources.
No gesture checkpoint is loaded during recording. Quality failures are retained
and reported. Private intent alone cannot disambiguate identical observed motions.

Nine meaningful synthetic checks passed in 1.91 seconds. Covers group/class
matching and order, idempotent preparation, preserved validation and reference
hashes, changed plans/weights rejected, wrong role/group/model/session rejected,
duplicate/missing/cross-sitting slots, quality failures retained, guided saves/
pause/resume, completed sessions avoiding camera reopen and failure cleanup.
These are software checks, not recognition outcomes; no physical camera opened.

Actual preparation completed in `experiments/data/boundary_pilot/`: plan.json,
plan.sha256, empty recordings directory, archived collector/tests/plan/protocol,
and preparation_verification.json. Verified all 560 reference inputs and five
original candidate hashes; no new recordings yet. Status sitting 1: 0/40,
sitting 2: 0/40. Original data/weights/source intact. Current human step:
`.venv/bin/python -m research.experiments.boundary_pilot record --sitting 1`.
Review and confirm 40 prompted clips, then q; audit before the later sitting.
Updated context/protocol/workflow/README. No new dependencies, model activation,
final freeze/test access, upload or commit. No savings/novelty/robot claim.

## 2026-10-08 — sitting 1 restarted on explicit user request

User reports a recording mistake and explicitly requests deleting the marked
sitting-1 clips without preserving them. Found 16 saved clips (the recorder's
next prompt was clip 17). Deleted all sitting-1 recordings and its attempt logs,
with no backup, overriding preservation for this explicitly requested subset.
Audit confirms sitting 1 is 0/40 and sitting 2 is 0/40. Prepared plan, source,
reference recordings and checkpoints unchanged. Restart the sitting-1 command
from clip 1; close any previous recording window first. Discarded acquisition
is additional research effort, not successful training data or label savings.

## 2026-10-08 — restarted sitting 1 completed and audited

User reports done. Found40/40 new clips in `20261008T063028-1f0a4f92`:
routine20 and targeted20, each five F/B/NO_COMMAND/DOWN. All pass tracking checks;
all task slots/metadata match the plan, no raw duplicates or historical-session
reuse. All560 reference inputs and five saved candidate hashes still verify.
Saved full current file/hash/quality audit under the isolated collection's
`collection_audits/`. No labels edited or models fitted. Numerical tracking and
provenance do not independently verify each gesture's meaning.

Current usable collection40; sitting2 remains0/40. Next record sitting2 in a
genuinely later physical sitting; targeted first, routine second,40 more clips.
Previous16 discarded saves remain additional acquisition effort; current40 do
not erase that cost. Planned comparison still waits for80 usable examples and
their audit. No new accuracy result, test access, checkpoint activation or
dependencies. Updated context/protocol/current workflow.

## 2026-10-08 — sitting 2 audited; matched training about to execute

User reports done. Both sittings are now40/40; second capture session is
`20261008T065105-55ca7759`. Total80: routine40,targeted40, each10 F/B/NO_COMMAND/
DOWN. All pass tracking checks; no raw duplicates/session reuse, all prepared
input/candidate hashes unchanged. Logged completed camera intervals34.51min,
including replay/preparation/pauses; discarded16 saves/offline time excluded.
Numerical/provenance checks do not independently verify gesture semantics.

Added `experiments/boundary_train.py`, leaving archived core/collector source
unchanged. Six synthetic checks passed in2.95s. Covers complete-cohort admission,
canonical class/sitting/repeat row pairing, no evaluation rows in training,
foreign-test metadata rejection without payload access, fixed-recipe equality,
matched initialization/sampling/class counts, end-to-end candidate reload,
baseline preservation, all prediction/metric reproduction and tampering detection.
Synthetic outcomes are software evidence only. No new dependencies.

Recorded execution protocol before real fitting. Run five seeds for each added
condition: same472+40=512, unchanged560-update recipe, original47 selecting
weights, second41 separately monitored. Preserve five original references and
all ten new weights/errors. No final freeze/test or camera model activation.
Real outcomes pending; record every unfavorable result.

## 2026-10-08 — experiment 004 completed; targeted advantage absent

Study `experiments/runs/boundary_data/20261008T072040903427Z`, all10 new fits
plus5 reference evaluations complete. Each added model uses512 clips, with
matching 124/85/68/59/57/61/24/34 class counts. All initial-weight and batch-index
hashes match within routine/targeted seed pairs. Original47 selection only;
second41 monitoring supplies post-selection inspected development outcomes.
No automatic camera upgrade, combined80-label fitting, new dependencies or test.

Monitoring reference/routine/targeted mean accuracies86.34/88.29/85.37%, runSD
4.43/2.67/0pp; macroF1 .8577/.8788/.8497 withSD .0447/.0278/.0030. Primary
targeted-minus-routine−.02905; paired seeds11/22/33/44/55 differences
+.00278/−.06862/−.03113/−.02475/−.02353. Expected advantage not supported here.
Routine improves mean accuracy1.95pp over reference (0.8more correct/41 per
model); no statistical/population superiority claim. All targeted models35/41
but different errors/weights/F1. Selection accuracies85.11/87.66/86.38 separate.

Monitoring recalls reference/routine/targeted:
F64/64/60, B96/88/84, L96/100/100, R96.67/100/93.33,
STOP80/96/88, NONE72/84/76, UP84/72/80, DOWN100/100/100.
Raw NONE→motion28/16/24%; NONE→DOWN24/8/20%; repeated STOP→motion counts4/1/1.
Routine UP/BACKWARD regress; targeted F/B remain weak. NLL .4690/.2671/.3439,
not proof of calibration. Class pools5 each exceptRIGHT6; predictions shared.

Exploratory post-primary inspection: all10 new models correctly recognize
original resting clip7601353c641440f9bca0b5c65cfed650 asNO_COMMAND. Another
NO_COMMAND clip2cf792b82a104d959940e7183c996466 goesDOWN in2routine/all5targeted.
Preserved all human labels. False-motion score ≥.8/≥.95 counts6/5reference,
1/0routine,4/1targeted; max targeted .9952. No threshold/gate/robot simulation.
Saved exploratory_diagnostics.json; do not elevate a repaired case to reliability.

Acquisition80 usable +16known discarded saves;34.51min logged completed camera
time includes replay/preparation/pauses, excludes deleted/offline intervals.
Mean fitting3.8461s routine and3.8260s targeted perseed, excluding other effort.
Reference weights reused with zero incremental fitting, not zero historicalcost.
Saved discarded_effort.json as a count only, without recovering deleted clips.

Verification passed640 original/current input hashes,15 checkpoint hashes,
1320 reproduced saved score vectors, archived sources/input arrays, exact old
reference outputs, matched start/sampling/counts and all aggregated results.
No failed real runs occurred. Preserved every model and unfavorable outcome.
Wrote finding004, Lesson06, standalone paired-seed/error plot (existing
Matplotlib3.11.2) and plot source; visually inspected the exported PNG. Updated
continuity/workflow/direction. Next proposed joint freeze/policy and genuinely
new evaluation; no final freeze, test access, upload or commit here.
## 2026-10-08 — experiment005 joint freeze; fresh human collection pending

User authorized the next step and asked to recall the research goal. Reaffirmed
ML education and evidence-backed cost/label-efficient robotics learning. No
novelty, publication readiness, SSL result or label savings presumed.

Added `experiments/boundary_test.py`, `experiments/test_boundary_test.py`,
`BOUNDARY_TEST_PLAN.md` and `LESSON_07.md`. Archived training/collector/core code
unchanged. New module handles three-condition joint freezing, metadata-only
progress, human-only prompted test collection and one access-ledger-protected
evaluation. Two physical sittings40 each,5/all8 classes/sitting; schedule
interleaves classes and reverses second-sitting label order. Natural resting and
no-hand NONE plus deliberate pointing/moving UP/DOWN styles retained. Same-user
specified-style clip pilot, not an unseen-person or robot final benchmark.

Primary specified before data: pooled80-clip macro F1 targeted minus routine,
paired by all five seeds and averaged. All15 candidate and per-class/per-sitting
outcomes preserved, no ensemble/threshold/seed tuning or early sitting1 scoring.
Labels count as evaluation cost; attempt timing includes replay/preparation/pause,
excludes unlogged offline effort. Poor tracking is reported/retained in scoring.

Seven synthetic safeguards passed in5.614s: complete/all-label schedule and
prompt-only keys, incomplete-pool refusal without landmark reads, historical/
pre-freeze/prediction-assisted metadata rejection, checkpoint tamper detection,
duplicate failure after an access ledger exists, joint all-candidate evaluation
with no-hand quality failures retained and repeat access blocked, and camera
adapter restoration on success/failure with no model loading. Synthetic accuracy
is not gesture evidence. No new dependencies or real test payloads used.

Real freeze at2026-10-08T16:42:36.341543+00:00 in
`experiments/data/boundary_test/freeze.json`, SHA256
`ab96e006890a4b6e3d032e76d3d5ee402a19b13f19cb218095f5619043d02398`.
Reverified640 original study inputs,15 checkpoint fingerprints and1320 archived
development score vectors plus matched starts/sampling/counts. Saved immutable
rule/source/input/candidate hashes and preparation verification. Metadata-only
status confirmed0/40 both sittings, complete=false, inspected=false, no ledger.
No new fitting, test scoring, deletion, activation, upload or commit. Human
sitting1 recording is the pending next step; sitting2 must follow a real break.

## 2026-10-08 — sitting1 interrupted; resume at clip21

User reported "Camera frame unavailable" after the twentieth saved clip.
Metadata-only status confirmed20/40 sitting1,0/40 sitting2, a valid saved task
prefix, complete=false and inspected=false. Existing recorder derives its next
prompt from saved progress; no code change, deletion, re-freeze or test payload
access required. Next is clip21 STOP (comfortable usual fist), save key s.
Resume with the same sitting1 command within the same physical sitting.
Prior clips and the frozen policy/checkpoints remain preserved.

## 2026-10-08 — first test sitting complete; second sitting pending

User reported completion. Metadata-only status confirms40/40 sitting1 with
five clips for each of all eight labels, sitting2 at0/40, complete=false and
inspected=false. Frozen source/input/checkpoint and saved task provenance checks
passed; this does not verify gesture semantics or prediction accuracy. No test
landmark reads, scoring, fitting, deletion or app activation. Next collect a
genuinely new physical sitting after closing the first window and taking a break,
using `boundary_test record --sitting 2`; all15 models are evaluated only after
both sittings finish under the existing frozen protocol.

## 2026-10-08 — experiment005 completed; gains plus unresolved UP failures

User completed sitting2. Metadata-only audit admitted40/40 both sittings,
five/all8 labels/sitting, complete=true, inspected=false. Executed the single
joint evaluation with existing frozen sources/weights and no fitting. Ledger
access at2026-10-08T17:58:07.022351+00:00, state complete. All15 candidates scored
on80 fresh clips; these sessions are now inspected. Original test artifacts
preserved in `experiments/data/boundary_test/evaluation`.

Reference/routine/targeted pooled accuracy79/84.5/83%, seedSD2.40/.68/2.09pp;
F1 .769173/.832048/.817127, SD .025683/.006862/.025472; NLL .999176/.495619/
.496283. Primary target-minus-routine F1−.01492094. Seed11/22/33/44/55 changes
−.00910102/+.02370089/−.04589897/−.00641274/−.03689285. Expected targeted advantage
absent, four/five negative; no significance/population claim or best-seed choice.
Routine gains5.5pp vsreference, targeted4pp, but both add40 training labels: no
label savings demonstrated. Existing inspected validation negatives retained.

Mean class recalls F/B/L/R/S/N/U/D reference94/86/100/100/86/34/32/100,
routine94/86/100/100/86/86/26/98, targeted98/90/100/100/78/86/26/86.
NONE→motion66/14/14%; NONE→DOWN60/10/8%; STOP→motion10/14/14%.
Sitting1 accuracy81.5/87/87%, sitting2 76.5/82/79%. Low UP existed forreferences
and persists/regresses after additions; preserve labels, no implicit UP→STOP fix.

Post-primary inspection grouped saved guesses by prompts: six unique UP clips
(whole-hand motion and angled pointing) are STOP for every15 model. Prompted
style is not independently verified performed style; cause remains a hypothesis.
All two no-hand NONE clips classified correctly despite quality failures; retained
in80 denominator. High-score false motion ≥.8/≥.95 instances34/32 reference,
7/4 routine,7/3 targeted, among38/14/14 total repeated-model errors. Maxscores
.9999939/.9996859/.9942541. No live gate, motor, threshold tuning or calibration
claim. Repeated predictions not independent clips/subjects.

80 evaluation labels;3751.27s=62.52min logged camera/replay/preparation/pause
intervals. Sitting1 capture IDs20261008T164636-0a26e7aa and20261008T171301-18937971
with20 each kept as one sitting; sitting2 ID20261008T173315-7a4cd252,40clips.
Offline restart/unsaved repeat effort incompletely measured. Raw duplicate and
session/provenance checks passed. No new dependencies or app activation.

Added post-evaluation `experiments/report_boundary_test.py`; it verifies original
artifact hashes,80 raw fingerprints,15 memberships/weights,1200 vectors from
archived features, labels/predictions and all pooled/per-sitting metrics/summary,
without reopening raw landmarks or another test invocation. Saved verification,
exploratory diagnostics and derived fingerprints in evaluation/analysis. Generated
standalone scientific class-recall/paired-seed figure using existing Matplotlib;
viewed PNG, clean layout. Wrote finding005, Lesson08 and continuity updates.
Frozen evaluator/core/collector code, all recordings and checkpoints unchanged.

Next proposed step: UP/STOP development-coverage/landmark diagnosis before more
collection or fitting. Grand goal remains cost-matched supervised/SSL/audited-SSL
and optional self-supervised/order comparisons plus participant/robot evidence;
none established by this supervised same-user pilot. Future choices informed
by this test require another genuinely untouched final benchmark.

## 2026-10-12 — replacement UP requested; STOP contract discrepancy identified

User says intended STOP is an open palm, not a fist, and authorizes removing old
UP from current learning and recording40–50 new training examples of a more
distinct gesture. The earlier written v2 STOP contract/prompts incorrectly said
closed fist. Documented this discrepancy in finding005 and current continuity;
old metric arrays/sources/labels/checkpoints remain unchanged. Whether old STOP
was uniformly open palm is not inferred. Asked two asynchronous clarifications:
exact replacement UP (recommended index finger up, others folded; thumbs-up
alternative/freeform) and historical STOP consistency. No answers received yet.

Built `experiments/commands_v3_collection.py` and its six synthetic checks.
Preparation requires explicit UP/style and STOP-history arguments. It selects
only all routine-condition training files from study004, excluding24 old UP;
it copies retained coordinates with full origin hashes/metadata into a new
v3/schema3 contract. With uniformly open-palm STOP reported, reuse57 STOP and
base488; with mixed/fist history exclude STOP too (base431) pending review.
Current validation/test cohorts never imported; old authorized recycled training
keeps provenance. All50 planned new UP clips are training: two25-clip sittings,
prompted modest shape-preserving angle/position/distance variations. No loaded
model, automatic fitting, fine-tuning, deleted source files or app activation.

First six synthetic checks failed on a manifest assumption: added routine rows
contain collection group rather than a split field. Fixed by verifying saved
split/label/session metadata explicitly before copying, and testing evaluation
exclusion from the manifest's named roles. All six then passed in7.039s:
old-UP exclusion/source preservation/schema incompatibility, uncertain-STOP
exclusion, evaluation separation/all50 training/no premature readiness, foreign
test metadata rejection before coordinates, raw duplicate/quality rejection,
resume/current prompt/no compatible old model and overlay restoration on success
and camera failure. Synthetic outcomes establish software safeguards only.

Verified real prior freeze15 candidates intact and study004's640 input hashes,
1320 archived development vectors/matched provenance. No real v3 destination was
prepared because human details remain pending. No new real training or test
access, no dependency additions. Recorded `COMMANDS_V3_WORKFLOW.md`, Lesson09,
protocol and context updates, including that motion is already in preprocessing.
Later improvements after changing a gesture task cannot establish the original
broad UP motion task was solved, label savings, SSL novelty or robot reliability.
Fresh all-class validation is needed before a separate new fitting plan.

## 2026-10-12 — user chose three UP shapes; real v3 prepared

User selected one or two fingers up with remaining fingers folded, or thumbs-up.
Defined one index, index+middle, or thumb upward, deliberately held about2s. All
three map to UP. Existing STOP report: "yes almost all were open palm, no fists".
Proceed with57 STOP retained on that qualified human report, recording exact
text and independent_review=false rather than claiming uniformly verified open
palms. No permission/clarification repeated. No automatic correction of STOP.

Extended v3 collector to `three_shapes` and per-sitting shape schedules.50 total
prompts:17 index,17 index+middle,16 thumb; first9/8/8, second8/9/8. Prompts mix
shapes and height/angle/distance, and metadata stores prompted shape without
claiming independent performed-shape verification. Seven synthetic checks passed
in8.160s, adding combined-shape balancing/counts and exact qualified STOP-report
preservation to prior exclusion/schema/role/quality/resume safeguards.

Executed real preparation with three_shapes/open_palm and exact report text.
At2026-10-12T17:24:40.563788+00:00, new `experiments/data/commands_v3` prepared
488 reused training clips from routine512, with24 old UP excluded. New schema3
keeps original metadata/byte hashes; all prior recordings/checkpoints preserved.
No current evaluation files migrated, old authorized recycling provenance retained.
Plan SHA256 `f03e2d06de995a2afedcb97356b4338c8d4033645c3e2fcedd2080fb05adac63`.
Preparation reverified source640 development inputs and1320 saved vectors; postcopy
verified all488 schema/contracts/train labels/no UP and original fingerprints.
Metadata/quality status confirms0/25 both sittings, newUP0, oldUP0, no training
readiness yet because fresh all-class validation/fitting plan are still required.

Updated continuity/protocol/workflow and teaching. Next user runs
`commands_v3_collection record --sitting 1`, completes25, q, then a new sitting2
after a break. No actual new collection, fitting, old file deletion, test access,
dependency additions or camera model activation performed by agent here.

## 2026-10-12 — first25 replacement-UP training clips complete

User reports initial25 done. V3 audit confirms sitting1 25/25, sitting2 0/25,
newUP25, oldUP0 in the new training version. Prompted shape counts9 index,
8 index+middle,8 thumbs-up. Provenance/prefix/session, tracking quality and
exact-duplicate checks passed, with488 reused inputs/original fingerprints and
prepared source/contract hashes intact. These checks do not independently verify
performed gestures or establish accuracy; no fitting or predictions occurred.
All existing data/weights preserved. Next collect a new physical sitting after
a break via `commands_v3_collection record --sitting 2`; remaining25 complete
the50 training target. Fresh all-class validation/fitting plan remain later work.

## 2026-10-12 — replacement-UP collection complete; training inventory538

User reports second25 completed, asks per-direction counts and whether data are
adequate to start training. Existing collector audit confirms25/25 both sittings,
newUP50, oldUP0. Prompt counts17 index/17 index+middle/16 thumb, independent
performed-shape verification not claimed. New training sessions
20261012T174044-0244fa51 and20261012T175809-93ca5838 contain25 UP each.

Audited every reused/new v3 train file: correct schema3/contract/role, no raw
duplicates, all538 pass numerical tracking quality. Original/migrated input
fingerprints and collector/contract hashes unchanged. Saved a full inventory in
`experiments/data/commands_v3/collection_audits/2026-10-12T182109507647+0000_complete.json`.
Counts F124/B85/L68/R59/STOP57/NONE61/UP50/DOWN34 across9 training capture IDs;
all prior data/checkpoints preserved. This is no claim of accurate human labels,
adequate gesture diversity, independent participants or population reliability.

Judgment: attempt a controlled pilot with current training counts; do not acquire
additional labels solely to equalize all classes at124. Planned class-weighted
loss reuses the prior recipe and adjusts frequency contributions, not coverage.
DOWN34 is lowest; validation should reveal whether more styles are necessary.
Missing prerequisite for checkpoint-selecting training: v3 validation0. Recommend
a separate physical validation sitting48 clips (6/class, UP2/each shape), all
outside gradients. Collection/fitting helpers not implemented yet. All48 would
count as development evaluation cost, not additional training labels or an
untouched final test. No training, predictions, code edits, dependencies or
camera model activation performed in this inventory turn.

## 2026-10-12 — v3 validation helper ready; human48-clip collection pending

User authorized next step. Added separate `experiments/commands_v3_validation.py`
and tests, keeping hash-bound UP collector/core unchanged. Preparation snapshots
all538 training files/labels/sessions/raw identities and hash-pins cohort/contract.
Schedule48 interleaved prompts,6/class: UP2/index,two-finger,thumb; STOP open palm;
ordinary/resting/repositioning/no-hand NONE; DOWN pointing and movement retained.
Validation saves only prompted human labels with no trained model predictions.
Different physical sitting outside training; restarts preserve prefix within that
same sitting. Training/history reuse, exact duplicates, role/schema/contract
mismatch and model-assisted labels refused. No-hand NONE failure kept/reported.

Six synthetic checks passed in8.470s: training snapshot/schedule/shape/STOP rules,
incomplete-UP refusal, foreign-test metadata refusal without coordinate access,
training-session/model-assisted/raw-duplicate refusal, complete48 cohort with
no-hand retention and unchanged training bytes, resumed key/prompt and overlay
restoration after normal/camera-error returns. No synthetic accuracy inference.
Tests used temporary fixtures; no new dependency or real model fitting.

Actual preparation2026-10-12T18:47:55.482611+00:00. Root commands_v3 now contains
validation_plan.json/.sha256 and validation_source archive. Plan hash
`98ae9ddc30e6df51b5177a9fd1da604225fe8ac3c7722b20d0791eba1719e94b`.
Verified real538 training inputs retained and six/all8 prompts with UP2/shape.
Status0/48, complete=false, quality failures0 (no validation collected yet).
Wrote V3_VALIDATION_PLAN.md and continuity/protocol/workflow updates. First doc
patch failed context matching and applied nothing; corrected patch succeeded.

Next user runs `commands_v3_validation record` after closing training camera and
taking a break;48/48 then q. No human recordings or model training/activation
executed by agent here. After completion audit and separate v3 fitting/loader
implementation follow. Validation will select weights, so later reported accuracy
is development evidence, not untouched test reliability/SSL/label-saving proof.

## 2026-10-12 — v3 models trained; UP90% but shared STOP failures remain

User reports48 validation completed. Audit confirms6/all8 labels,UP2/each shape,
session20261012T203344-955c84cd, one no-hand NONE quality failure retained.
Training538 unchanged. Saved before-fit validation audit and declared exact
five-seed/560-update training procedure in PROTOCOL before real fitting.

Added separate `experiments/commands_v3_train.py` and four tests; reuses the
prior fitter/features/network, strict schema3 contract loader, role manifests,
full source/config/features/environment archive and preserved histories/weights.
Tests initially passed6.181s; after tightening development/session/plan provenance
and saved loss equality, four passed6.144s. Cover incomplete/foreign data refusal
before fitting/output allocation, split/features/no-hand retention, tiny complete
run/old-loader incompatibility/wrong-contract rejection/source preservation and
tampered predictions. No inference from synthetic accuracy. Frozen core/collectors
unchanged; no new dependencies.

Executed `commands_v3_train train`; study
`experiments/runs/commands_v3/20261012T205727638067Z`, all5 complete, no failed
real fits. Seeds11/22/33/44/55 validationcorrect41/42/42/40/41 of48, accuracies
85.42/87.5/87.5/83.33/85.42%, F1 .8567/.8772/.8756/.8373/.8587. Savedupdates
420/480/300/360/500, each selected by minimum unweighted validation loss.
Means accuracy85.83% SD1.74pp,F1 .861106 SD.016301,NLL .415667 SD.103115.
Every seed retained; no winner activated or inspected test reused.

Mean recalls F/B/L/R/S/N/U/D73.33/70/100/100/66.67/86.67/90/100%.
NONE→motion13.33%,NONE→DOWN0,STOP→motion30%. STOP4/6 correct in every seed;
sameclips31f155fc13db4984b9b5108026540f79 (different-height prompt) and
906b8197e62b4f2d80809490b0cd8028 (angle prompt) missed by all5. First alwaysF;
secondF in4,NONE inseed33. Maxwrongscore .9939603. Retained humanlabels.
UP index/two-finger10/10 repeated predictions each correct;thumb7/10, one
angled thumbsclipbda917a4c0f148ab876d11514343287d goesLEFT in3seeds. One ordinary
repositioningNONEclip ee21e3af784b4c648f9a29f2c1528904 LEFT in4seeds,maxscore.94982.
Post-primary falsemotionSTOP/NONE13 instances,8≥.8,4≥.95; no threshold tuning,
live gate or motor-action evaluation. Repeated guesses not independentclips.

Verification passed586 original input fingerprints,5 checkpoint hashes,240
reproduced score vectors/predicted names and all metrics/summary/minimum-loss
provenance. Saved every model and negative result; camera unchanged. Fitloops
25.65s total (~5.13s/seed). New50 training+48 validation labels, camera logged
24.04min UP/17.45min validation, includesreplay/preparation/pauses, incomplete
offline/unsaved-repeat effort. Reused488 labels/pretraining/correction costs retained.

Wrote finding006, Lesson10, post-primary diagnostics and curve/class-recall
scientificplot source withderived hashes. First direct plot invocation lacked
project module path; corrected to module invocation. First visualinspection
found edge labels clipped; moved them insidebars, regenerated and re-viewed clean
PNG. One continuitypatch context failed atomically; corrected patch applied.
No further models/tests run because of these documentation/plot corrections.

Next review shared STOP/thumb-angle/F-B errors usingexistingdata beforemorelabels
or app activation. Selectedvalidation is development, not finaltest; changedtask
precludescontrolledv2 accuracyclaim. No SSL/label-saving/novelty/robot-reliability
result established. Allcontinuity documents now reflectcompletedcollection/fitting.

## 2026-10-12 — deeper temporal/label-efficient research and motion probe007

User asks whether movements/order are represented, aboutLSTM/BiLSTM/3D-CNN and
starting research for a new/improved algorithm. Readcontinuity andactualcode.
Confirmed32×67 (63localXYZ,2wristXY,1logsize,1visibility), twoConv1d(k5), pool4,
8-classhead48,520 params. No explicitderivatives/dt/duration/recurrentmemory;
timestampscontrolinterpolation. Localfield9samples, finalheadwholeclipaccess.
Architecturecapacity isnot proof of learned reliance/meaning. xyz≠Conv3d.

Delegated parallelprimary reviews/codecritique under active delegation setting.
Three agents saved onlytheir notes: temporal_prior_art, label_efficiency_review,
mechanism_protocol dated2026-10-12. Reviewed closest DD-Net/DG-STA/hand graph,
MMANet/SkeleMoCLR/SESAR/AS3L, cost/riskcontrol/PPI and2026 screens with readingdepth
andfailedfetchesexplicit. Root inspected primaryTCN/C3D/MMANet/DD-Net/DG-STA text.
Skeleton-volumeConv3d qualification added; all3DCNNneedRGB wouldbeincorrect.
Knownpose/motion/attention/SSL/staging/robotgate components cannot beclaimednew.
No dataset/dependencydownload, newarchitecturefit or novelty determination.

Declared fixed-weight probe before execution in PROTOCOL. Added motion_probe.py
andthree meaningfulsyntheticchecks; passed .021s forchannelisolation/inputimmutable,
wholeframepermutation andproperraw reversal/reanchoring. Variantsbaseline,zeroXY,
zerosize,centerlocalshape,centerlocalshape+zerotrajectory(visibilityretained),fixed
shuffle(seed20261012),rawreversal/re-extraction. Only48 inspectedv3 validation and
all5 fixedweights; source/input/hashsnapshot archived; nofitting/newlabels/test.

Run `experiments/runs/motion_probes/20261012T221029684780Z` complete. Baseline240
vectorsreproduced. Toplabel changes across240 repeated-model outputs0 wrist,
1size,17staticlocal,17static+zero,0shuffle,1rawreverse. Original-label agreement
85.83/85.83/86.25/82.08/82.92/85.83/86.25%. These arecounterfactual agreement,
notvalidnewaccuracy, andsomegesture meanings/OODinputs differ. BACKWARDagreement
70→36.67% staticlocal; STOP66.67% inallvariants. Scorechanges max.09079 wrist,
.08011size,.97464staticlocal,.96076staticzero,.05751shuffle,.15418rawreverse;
so motion/orderchannelsnotdemonstratedunused. Inferencesmallpose-heavy sample
more sensitive tochangingarticulation thanprobedorder/globaltrajectory. No causal
shortcut/provenrobusttrajectory inference. Allfavorable/unfavorableviewsretained.

Independently reproducedall1680 savedprobevectors, metrics/fliprates/JS/summary,
andcheckedartifactfingerprints,weightsunchanged. Savedindependent_verification.json.
Wrotefinding007,Lesson11,TEMPORAL_ALGORITHM_PLAN andupdatedliterature/direction/
robotproposal/context. Candidatehypothesisbudgetedrandomconfidentauditsguideactual
pseudo-labeladmission undersemantic/consequencecost, matchedtoSSL/active/riskbaselines.
Notfinishedalgorithm: estimator/pool/sampling/round/cutoff assumptionsneedformalizing.
Policycan'tusephysicalfeasibilityasintenttruth orbiasedauditsasvalidpoolbounds.

Nextsmallfinitearchitecture/representation comparisonwithsamev3task/pools/budgets
andstaticbaseline, thenfrozenrepresentationlowlabel/SSL/audit study. Whole-session/
participant exclusionsandcostsretainpriority. Noautomaticappactivation, newmodel
training,testpoolreaccess,oldrecord/weightchanges, or publication/label-saving claim.

## 2026-10-13 — experiment 008 prepared before real fitting

User authorizes next work and requests a reminder of the research objective:
learn ML while developing label-efficient robotic gesture recognition, with
novelty, reliability and publication ambitions supported by genuine evidence.
Read continuity and code; reproduced the prior 586 input hashes, five weights
and 240 validation score vectors. No historical test data accessed.

Declared `TEMPORAL_COMPARISON_PROTOCOL.md` and PROTOCOL addendum before fits.
Five conditions: existing CNN 48,520 parameters; pose_set 48,792; feature_set
49,176; LSTM 48,944; BiLSTM 49,256. The two set models use frame MLP + mean
pooling and cannot retain frame order; pose_set also excludes wrist/size channels.
They still observe changing poses as a collection. First-frame-referenced wrist
features may encode trajectory even without order. DD-Net/graphs/derivatives
remain a separate representation experiment, avoiding bundled feature changes.

Added separate `temporal_comparison.py` and tests, preserving frozen sources.
CNN calls the existing fitter; other architectures copy the exact numerical
recipe with explicit model construction. Same arrays, labels, seeds, replacement
sampling streams, class-weighted training, unweighted validation selection and
560 updates. All 25 models will be retained; no family-specific tuning or winner
seed. Fresh sequential workers measure actual time/process RSS; window timings
exclude acquisition/tracking. No new dependencies or additional human labels.

Checked official PyTorch 2.14 LSTM documentation and primary Deep Sets reference.
Nine checks passed in 7.681 seconds after adding worker-source verification and
early CNN reproduction checks: five new checks cover gradients/counts, set
invariance/trajectory isolation, malformed comparisons, complete tiny paired
archive with tamper detection, and changed-cohort refusal; four imported prior
fitter checks also ran. Temporary synthetic fits are not accuracy evidence.
Real comparison results pending; negative outcomes will be recorded.

## 2026-10-13 — experiment 008 complete; accuracy improves with command-error trade-offs

UTC study `experiments/runs/temporal_comparison/20261014T011524632211Z` (October13
Chicago). All 25 declared fits completed in fresh sequential CPU workers; no
real failed fits or replacements. Same exact 538/48 arrays, all original roles,
class weights and paired per-seed sample streams. Preserved every checkpoint.

CNN/pose_set/feature_set/LSTM/BiLSTM mean accuracy 85.833/85.833/85.417/89.583/
88.750%; F1 .861106/.859964/.856773/.890919/.886943. LSTM/BiLSTM primary mean
paired F1 gain +.029814/+ .025838, positive in 3/5 and 4/5 seeds. Accuracy seed
SD 1.743/1.743/1.473/4.886/4.796 percentage points. No best-seed selection.
Recurrent F/B recalls improve; pose baselines approximately match CNN. This does
not isolate useful time-order reasoning or establish a universal family winner.

LSTM STOP recall 76.67% versus CNN66.67%, but NONE→motion 33.33% versus13.33%
and NONE→DOWN20% versus0. BiLSTM NONE→motion10%, STOP recall66.67%; both same
STOP clips missed by all five seeds. Raw STOP/NONE false-motion repeated counts
CNN13,pose15,feature14,LSTM17,BiLSTM13 of60 predictions each. Scores≥.8 counts
8/6/4/10/7; ≥.95 counts4/2/1/3/2. Highest false motion scores .99396/.97735/
.96658/.99506/.97600. Saved post-primary error audit and its standalone helper;
no label changes, ensemble voting or threshold/gate evaluation followed.

Fit loops total47.8005s; full fresh-worker wall75.5435s. Mean fit seconds
3.847/.435/.409/1.949/2.920; local preprocessed-window CPU median times
.061/.038/.033/.211/.333ms. Process peak RSS includes interpreter/Torch/data/
optimizer, not isolated network memory; background/fixed-order timing limitations
documented. Human new labels0; historical538 training/48 evaluation effort counts.

Original CNN initial states/batch hashes/chosen steps/final states/scores exactly
reproduced. All586 raw fingerprints,25 new checkpoints,1,200 saved vectors,
metrics/summary and minimum-loss provenance verified, with old five checkpoints
unchanged. Repeated standalone CLI verification passed. Sources/settings archived
before fitting. Scientific PNG/PDF generated and visually inspected clean;
Matplotlib used its writable temporary cache after the default home cache warning.
The derived plot/error-audit scripts and outputs are preserved with fingerprints.

Wrote finding008 and Lesson12; updated continuity/protocol/README/algorithm plan.
No new dependencies, final-test access, recording replacement or camera activation.
Next investigate existing STOP/NONE errors before further representation/session
evaluation and the separate matched-label-budget learning-policy experiment.
Current evidence is inspected single-user checkpoint-selected development, not
SOTA, novelty, label savings, robot reliability or publication readiness.

## 2026-10-14 — development failure analysis and offline review viewer ready

User authorizes next step and asks to inspect failing clips by category, to
contribute useful observations. Read continuity, declared FAILURE_REVIEW_PROTOCOL
before descriptive coverage work, verified experiment008/586 inputs/25 models/
1,200 score vectors. Added separate failure_review.py and offline HTML/JS assets,
preserving all frozen source files, recordings and checkpoints.

All48 inspected validation clips analysed and replayed, with53 distinct nearest
training references. Raw timestamps/visibility/mirrored convention preserved;
centered view reuses palm/aspect normalization without inventing gap frames.
Exact archived-feature agreement required for each of101 exposed raw replays.
Pose-profile RMS used for per-class neighbours, wrist RMS reported separately;
no combined optimized metric or neighbour classifier. No-hand pose undefined,
therefore no misleading nearest-pose reference shown for that legitimate clip.

Artifact `experiments/runs/failure_review/20261014T053532241062Z`: total15 unique
failure clips out of48 across any of25 models. F/B/L/R/STOP/NONE/UP/DOWN
2/3/0/1/2/5/2/0 of six each. Family/seed filters distinguish occasional errors
from shared failures; zero-failure categories and all correct controls retained.
Both STOP failures tracked100%, un-clipped features; nearest STOP .600/.733
versus correct .053/.086/.086/.096. Shape RMS variation .069/.078 versus
correct .011–.033. Pose/tracker/normalization/coverage hypotheses unproven.
First failed STOP wrist path .449 overlaps successful STOP range .139–.477.

Repositioning NONE ee21e3af784b4c648f9a29f2c1528904 is LEFT18/25, maxscore.99506,
despite nearest NONE .128 versus LEFT .200. Other relaxed/resting/fingers-down
NONEs become DOWN in several recurrent seeds; private intention not directly
observable. No-hand NONE onlyBiLSTM11→LEFT score.3104; allother24 correct.
No label correction, confidence threshold tuning, gate/motor outcome or new fit.

Viewer starts with model guesses hidden, original labels/prompts, camera/centered
views, pause/slow/scrub and training comparisons. Separate saved assessments/
notes/proposed labels export as gesture_review_notes.json; versioned importer
checks bundle/clip identities, refuses duplicate/foreign notes, changes no data.
Human feedback pending. Approximate foreground viewer time includes pauses;
review/confirmation/correction effort must count in future label budgets.

Five backend checks passed .007s;15 real Chrome UI checks passed for filters,
mirroring/reveal, zero-failure/missing-hand handling, exported observation data,
frame timing/scrub and preserved labels. Sandbox headlessChrome launch aborted
SIGABRT; requested required escalation, approved run succeeded using temporary
profile. Nonfatal macOS display-link/allocator warnings preserved; no user's
browser profile accessed. QA notes are synthetic, not user annotations. Screenshot
viewed clean. Added a short local HTML launch link inside experiments.

Wrote finding009/Lesson13 and continuity/protocol/README. Source/data/artifact
fingerprints and CSV/JSON/browser QA preserved. No dependency installation,
test-pool access, camera activation, claimed novelty/label saving/reliability.
Next user reviews STOP/NONE first, other failure categories and correct controls,
exports observations, then agent inspects/imports separate feedback before any
evidence-based intervention. Main audited-SSL algorithm remains future work.

## 2026-10-15 — temporal user feedback and experiment010 preparation

User watched failures and emphasizes sequence-level finger relationships,
physical-time speed/direction, maintained-shape+motion STOP, hand references
preserving meaningful orientation, early decisions and future users/views/speeds.
Saved narrative guidance as a separate dated note. Named Documents/comments
exists but all readable listings are empty, including after user reply that notes
were just pasted. No JSON feedback imported, review-time total invented or labels
changed. A narrow Downloads filename search did not finish and was interrupted;
no unrelated file contents were read. Continue from the available narrative.

Declared MOTION_REPRESENTATION_PROTOCOL/PROTOCOL before new features or fitting.
Four camera/hand × pose/motion conditions, shared3D estimated-palm scaling,
orientation matrix/globalXY/time retained; derivatives masked across raw tracking
loss/gaps and invalid basis. New32×146 schema separate from frozen original67.
Reuse LSTM/four-bin head pattern at146 inputs/47 hidden,49,276 parameters,
identical initial weights/batch/dropout streams, five seeds and560-update recipe.
Prior67-input/62-hidden model is context only, not a matched causal reference.

Five checks passed .631s: translation/scale properties and reconstructible
orientation, real-time stretch/velocity scaling, missing/gap derivative masking,
no-hand/degenerate basis/prefix future isolation, paired tiny fits with gradients
and strict checkpoint feature-version refusal. Primary contrast camera motion
minus camera pose macroF1; all secondary/safety errors retained. Prefixes.5/1/1.5s
are recomputed using observed data only after full-clip fitting; no prefix loss
selection, early-exit threshold or onset-latency claim. Real results pending.

## 2026-10-15 — resumed completed experiment010 and review imports; verified artifacts

At resumption, CONTEXT/PROTOCOL/finding010 already recorded a completed 20-model
motion/reference comparison and nine imported JSON exports. Checked actual files
rather than repeat training or duplicate cumulative imports. Original files are
directly in Documents; comments directory is empty. Verified original/archive/
versioned-import equality, bundle/raw clip hashes, eight unique original-label
confirmations, nine edited versions and43 cumulative snapshot entries. No relabel.
Read the actual notes about inward curl, open-palm frames inside FORWARD,
stroke/recovery speed/rhythm, held STOP and relaxed/down-facing NONE. Treat
perceptual/phase explanations as hypotheses, not proof or hardcoded label rules.

Repeated motion_representation verify reproduced all20 saved weights and3,840
full/prefix score vectors/metrics/minimum-loss selection, identical initial
states and sample streams. Re-extracted all2,920 features from586 recordings
and matched archived tensors, geometry audits and prefix observations exactly;
previous25 checkpoints verified. Existing scientific figure visually inspected.

Recorded outcome camera_pose/camera_motion/hand_pose/hand_motion means
84.17/97.92/91.67/92.08%, primary F1 difference+.144212 inallfive seeds.
Camera_motion NONE→movement0%,STOP recall93.33%; hand_motion STOPrecall96.67%
but overall92.08% and NONE→movement6.67%. Preserved all negative trade-offs.
Shown old STOP906b819... is now correct3/5 camera_motion seeds; other shared
STOP and repositioningNONE are correct5/5. Existing data heavily inspected;
no across-person/view/speed reliability, sequence-memory, label-saving or novel
algorithm conclusion. Prefix agreement is not validated early recognition.

Saved a dated audit JSON in the existing study and a plain-language follow-up
note; updated continuity/protocol. Kept numerical fitting archives unchanged.
Next priority same-motion-feature order-invariant/causal controls, then frozen
settings/new grouped session-person-speed-view evidence. WAIT/defer and onset
annotation are distinct from NO_COMMAND. Main audited label-efficient policy
remains future work. No new fit, recording request, external data download,
threshold selection, final-test access or automatic app activation in this audit.

## 2026-10-15 — experiment011 other models, audit, controls; experiment012 ready

User requests same motion results for other families, all test datasets validated,
then controlled steps. Asked optional scoring-versus-audit clarification. User
asks us to choose proper validation without false remarks. Recorded audit-only
historical performance policy before any such scoring; retained original reports
and test access ledgers. No historical test performance recomputation occurred.

Predeclared CROSS_MODEL_MOTION_PROTOCOL before new real fits. Added new research
adapters/fitter/strict loader/verifier cross_model_motion.py, dataset_validation.py
and five synthetic mathematical/provenance/tiny-fit checks. Passed .426s before
real fitting; initial test expected an error string from a deliberately malformed
fixture, fixed fixture to valid metadata before wrong-version rejection, not a
failed real fit. No original frozen core/model/collector source modified.

Study cross_model_motion/20261015T164901242592Z completed80 core +15 control new
fits,20 exact experiment010 LSTM states reused and original files copied/hashed.
Same538/48 version2 arrays,560 updates/28 checkpoint checks/five seeds/~49kparams.
Initialization paired withinfamily; replacement sample streams acrossfamilies.
All115 models retained; no architecture tuning or app activation. Fitloops95new
197.3645s total, excludes earlier fits/software/audit/human effort.

Mean camera_position/camera_motion/hand_position/hand_motion accuracy:
CNN89.58/99.17/85.83/87.08%; LSTM84.17/97.92/91.67/92.08%;
BiLSTM89.17/99.58/91.67/92.50%; localset84.58/98.75/81.67/87.08%;
featureset85/99.17/85.42/89.58%. Primary withinfamily motion−position macroF1
+.095059/+.144212/+.104811/+.140303/+.141876, allfivepairedrunspositiveeach.
Camera-motion BiLSTM33 NONEee21e3af...→LEFT(.52578), 3.33% repeatedNONEfalsemotion;
others0observed. LSTM STOPrecall93.33%, others100%. Not a universal robotwinner.

Predeclared camera-motion controls causalCNN98.33%/F1.9832; no_elapsed set98.75%/
.9874; no_elapsed LSTM97.92%/.9790. Set scores remain strong without recurrent
memory; no established advantage for recurrent chronology in this cohort.
Velocities/firstreference/intervals retain local timeinformation; no raworder
irrelevance claim. Causal stack test andprefix agreement do not establish early
robot response or onsetlatency. All negative trade-offs preserved.

Audited1,854 canonicalfiles/942uniqueraw, malformed0, conflictingcanonicallabels0.
Legacyvalidation162 has90alreadycurrenttraining,72outside. Legacytest52 sixclass;
v2validation88 excludes10oldUP→78common; boundarytest80 excludes10oldUP→70common;
currentv3validation48. HistoricalSTOP qualifiedreport, notallvisuallyverified.
One obsolete labelrevision archive excluded. Initial expected20v2UP documentation
arithmetic corrected toactual10; audit always classified actualmetadata.
All80boundarytest file/content hashes match originalevaluation; freeze/access
ledgers matchaudit. Legacytest52 currently valid/hashsnapshotted; no complete
original per-file bytehash manifest located,0/52 in migration's development-only
contentpool, no unsupported historicalbytepreservation claim. All1,854currenthashes
rechecked. No testpool used for fitting or unlabeled pretraining.

Added PROSPECTIVE_MOTION_PLAN/prospective_motion.py and six synthetic checks;
passed .411s (initial foreign-role fixture refused by strictwriter, then tested
malformed metadata directly). Covers balancedprompts/resumption/restoredUI hooks,
newsession/duplicate/modelassistedrefusal,incompletebeforepayload/ledger,complete
96jointsyntheticroundtrip/retirement/tamper. Synthetic data are not user clips.
Frozen all115 candidates/full executable source chain/settings/cohort rules into
experiments/data/prospective_motion_011. Real freeze verifies source/fitarchives,
weights/115outputs. Status0/48+sitting2 0/48, no camera opened. Do not mutate
frozen .py sources; documentation may evolve separately from archived snapshots.

Two genuinely new physical sittings of48, six/class/sitting, human promptedlabels
without guesses, newraw/session only, bothcompletebeforeonejointperformanceaccess.
Full115candidatefamily declared, no refit/luckyseed/gate/earlyexit/ensemble choice.
Cohort becomes inspected development after scoring, failed access retired too.
This tests new sessions for currentuser, not unseenpeople/factorialspeeds/views/
continuousrobotcontrol/finalpaperclaims. Future mainlabel-efficientpolicy remains
separate; no SSLgain, SOTA, novelty, publicationreadiness orlabel savings proved.

Wrote finding011/Lesson15, fullrunREPORT, 115model/920class CSVrows and464failure
instances plusuniquepercategorycounts, integrity/costaddenda andscientificPNG/PDF
(inspected after title-layout correction). Old HTML viewer remains25experiment008
models; new failureCSVdoesnotimplyitwasupdated. UpdatedCONTEXT/PROTOCOL/READMEs.
No new dependencies, recording/label changes or checkpointactivation. Next user
runs .venv/bin/python -m research.experiments.prospective_motion record --sitting 1.
All96nextlabelsandattemptintervals will count as additional effort.

Final completion checks: all11 comparison/prospective tests pass together in
.915s after full source-chain freeze; reproduce115checkpoints/22,080vectors and
586inputs again, no missing new documentation links. Frozen collector status
remains0/96. Derived artifact hashes/completion_verification.json saved; no
source/weight/score/audit/freeze changed.

## 2026-10-15 — experiment012 sitting1 completion audit

User reports48 gesture recordings complete. Read current context/protocol/log,
then used prospective_motion.status for metadata-only checks. Sitting1 48/48,
sitting2 0/48; six/class, capture session20261015T230148-8e32d742, all48 stored
tracking-quality flags true. Schedule/contract/task prefix/roles, duplicate/raw
identity/session guards and pinned source/checkpoint/study hashes pass. No raw
xyz/times/visibility payload decoded or model inference performed for evaluation;
no evaluation_access ledger exists. Do not describe this as independent semantic
label verification or fresh numerical payload checking. Those checks/scores wait
until both sittings complete under the declared policy.

Saved versioned metadata collection audit in prospective_motion_011/collection_audits,
including48 original file hashes/attempt ledger. One completed attempt from
23:01:47.691197 to23:19:56.716351UTC,1088.98446675s (18.15min) camera/replay/
preparation/pauses, not pure annotation time. All48 acquired human labels count.
Updated CONTEXT/PROTOCOL/READMEs; numerical study/freeze/checkpoints/recordings
unchanged. Next user takes a break and records sitting2's48 prompted clips as a
genuinely new physical sitting. Same gesture meanings; no sitting1 fit/tuning/
performance review. Both96 required before joint frozen115-model evaluation.

## 2026-10-15 — experiment012 complete collection; joint evaluation begins

User reports remaining48 complete. Pre-access metadata audit confirms96 total,
48/sitting andsix/class/sitting; captureIDs20261015T230148-8e32d742 and
20261015T232301-711ada77.95/96 stored tracking-quality flags pass; the one flag
failure is in NO_COMMAND (per status only NONE permitted), retained under the
predeclared policy. No payload decoded by this collection audit; no performance
access ledger exists yet. Frozen source/study/115checkpoint guards all pass.
Versioned both-sittings metadata audit saved with96filehashes/attempts.
Two completed attempts total1874.806464s (31.25min), camera/replay/preparation/
pauses, not pure annotation. All96 new human labels count; no labels replaced.

Proceed now with the existing frozen prospective_motion evaluate command:
one access ledger before raw payload decoding, all96 checked/extracted, joint
full-window scores for all115 checkpoints, pooled+eachsitting. No retraining,
threshold/ensemble/seed selection or camera activation. All prior scores/weights
preserved; any access retires this cohort as inspected development. Outcomes
pending at this entry; same-user pilot only, no general robotics/SOTA/label-saving
claim. Next derive a report from saved scores, preserving all unfavorable results.

## 2026-10-15 — experiment012 complete: mixed replication, review ready

Executed existing prospective_motion evaluate once after both sittings complete;
access ledger written before payload decoding, joint115models×96clips=11,040
vectors, all pooled/sitting metrics verified. Completed23:36:44UTC, no refit or
settings/seed/threshold/ensemble choice, no checkpoint activation. Now inspected
material; future verify uses archive, do not restart evaluate as untouched.

Allfour-mode mean accuracy(camera_position/camera_motion/hand_position/hand_motion):
CNN94.17/93.33/93.54/89.17%; LSTM90.21/94.58/95.62/91.88%;
BiLSTM96.25/95.83/93.33/91.46%; localset94.17/95/94.58/91.67%;
featureset94.17/92.92/95.83/90%. Primary camera_motion−position macroF1
−.01162298/+.04689243/−.00771537/+.00657145/−.01683010, positive2/3/2/3/1of5.
Motion improves meanaccuracy in sitting1 but reduces it in sitting2 acrossallfive
families. Uniform earlier developmentgain didnotreplicate. No causal viewpoint/
speed/order/overfit attribution or universalarchitecturewinner. Handmotionbelow
handpositionallfamilies. Controls causalCNN91.46%, setnoelapsed93.54%, LSTMnoelapsed
95.21%. Long-rangememory advantage not isolated, noearlyaction/onset/prefixclaim.

Corecamera_motion UP recalls65/70/73.33/76.67/65%; NONE→motion3.33/6.67/5/5/5%.
All115 recognize12STOPclips,1380repeatedguesses on12uniqueclips, notguaranteed
robotstopping. Unionfailuresacross115:46/96, F8/B6/L5/R4/S0/N11/U11/D1;
730repeatederrors, not singlemodelerrorrate. Full probabilities/classconfusions/
seedpairedcontrasts/sittings preserved. No unfavorable rows discarded/relabelled.

Added separate prospective_motion_analysis.py forpost-access derivedreports and
replays, frozen sources unchanged. Re-extracted all384features exactly from96raw,
checked identity/session historicalexclusion, currenthashes, actualtrackingquality
flags and no clipping.95 qualitypasses+1validallnohandNONE, preservedall96.
One NONE685196... promptedhands-out yettracked100%,13/25coreCMwrong(max.9473),
so prompt/execution/tracker consistency needs humanreview, notautomaticlabelchange.
UP37729d... index20/25coreCMwrong and38b0...thumb15/25(max.9973). Shape-onlynearest
reference descriptions are not intenttruth or explanations.

First derived-analysis attempt failed at regex replacement because JSON Unicode
escapes were interpreted as regex backreferences. Saved partial/source/failure
under analysis_attempt_20261015T234014630154Z, fixed newhelper with callable
replacement, regenerated onlyderivedreports. No evaluation restart/scoremutation.

New offlineviewer contains96clips/all23conditions/5seeds/89verifiedtraining
shape-nearestreferences, hiddeninitialguesses, allcorrectcontrols, originaltimes/
mirroring/replay/slow/scrub andseparatenotes. Distinctshortcut
research/experiments/prospective_motion_review.html; old25modelviewer unchanged.
19realChromeUIchecks pass using approvedisolatedtemporaryprofile. NonfatalmacOS
CVDisplayLink/allocator/taskpolicywarnings preserved. Screenshotclean andscientific
figurePNG/PDF inspected; movedplotlegendoffbartops. SyntheticQA notes nothuman
feedback andnotimported. Actualhumanreviewpending.

Saved summary/model(345rows)/class(2760rows)/error(730rows)/clipindex CSV/JSON,
posthocUPstylebreakdown(4uniqueclips/style,not20independent),rawfeature/cost audits,
reviewbundle/source/figure andbrowserQA. Wrote finding012/Lesson16 andupdated
CONTEXT/PROTOCOL/READMEs. Sameuser/twosittingsoneevening~3.07minapart,notnewpeople/
crossday/factorialspeedview/robotproof.96newhumanlabels,1874.8065s(31.25min)
camera/replay/preparation/pauseintervals,notpureannotation; prior/offline/review
costsadditional. Newfits0, nodependencies/label-savings/novelty/SOTAclaim.
Next user reviewsNONEandUP, saves/exportsobservations; then selectonecontrolled
hypothesis beforeanynewfitting. Keep96inspected, separatenewsession/personfinal
cohorts forlater decisions. Mainauditedlabel-efficientlearningremainsfuturework.

Final experiment012 completion verification:96 rawhashes unchanged,384featurearrays
matched,115models/11,040newvectors reproduced; original586inputs/22,080vectors
and pairing verified again. All CSVrowcounts/documentlinks pass; posthocstyle
helper reproduces CSVbyte-exact. Finalderivedsource/artifacthashes/completion
verification saved. Frozen evaluator/source/model/config and originaljointscores
unchanged; no additional fitting or population/performance claims.

## 2026-10-15 — user stops new validation recording; raw algorithm design

User: no more validation gestures, use existing forvalidation/test; invent a
strong improvementalgorithm with smallerdata, then discussnextstep. Persisted
no-more-personalvalidation/testrecordings inAGENTS/CONTEXT/PROTOCOL/READMEs.
Inspectedoldendpoints remain retrospective; no falseuntouched claims. User'snew
instruction supersedes any earlier expectation to requestmoreclips or mandatory
humanreview beforeindependentdesign. No camera/collectionasked.

Read requiredcontinuity andpriorlabelbudget/research/robotics plans. Parallel
primarysource/algorithmcritique/fixed-data reviews saved three datedagentnotes.
Noagents fit/evaluated/modifiedfrozencore. Boundedsearch confirmsprioringredients:
ModDrop gesturemissingchannels, FixMatch confidenceselftraining, rationaleguided
selftraining, pose/articulation/globalmotionseparation and gestureMoE. Latest
audited/domain-awarepseudo-label androbotgestureadaptation work alsoclose; exact
reading/accesslimitations documented. Novelty unestablished, no broadbest/SOTA/
publicationguarantee. Root recheckedoriginalthreepaperidentities: CReLU2016,
FixMatch2020, activationfunctionsurvey2021/2022; noactivationchoiceinvented.

Centralcandidate Evidence-Profile Self-Training: trainnative missing-evidence
conditionalCE teachers (no full/subsetKL); signedclasslogoddsresponses to omitting
pose/localvelocity/globaltransport; compare candidates againstsamefrozenteacher's
disjointsession sameclassanchor neighborhoods beforeadmittingsofttargets.
Fourviews, explicitavailabilitybits, supportsuncertainpartialviews andmultiple
observedstyles; nohardcodedUPstatic/Fdynamicrules, no rawreversal/rotationlabels.
Provisionalbothfullteacheragreement/confidence.95 plusresponsecompatibility;
min3correctanchorsclass, heuristicMAD/LOOradius, unsupportedadmissiondeferred
notlabelledNONE. Onefrozenteacherround/freshstudent. Maincontributionhypothesison
admission, notnewfusion/masking/causality/theorem. Sharedwrongprofilescanpass.

Independentmetadataaudit reproduced538=405original+133model-selectedcorrections;
primary405 counts86/63/46/41/44/41/50/34. GroupA250 counts61/38/31/26/29/16/25/24,
groupB155 counts25/25/15/15/15/25/25/10, wholecapturegroupsall8. UPonlytwo
singlelabel sessions preventsfullynestedgrouped8-classCV. SavedfeasibilityJSON
withprovenance/counts, notlearnerinput. No originalraw payloads orendpointinference
neededtoday. Proposed80traininglabels (5/class/group),325hiddenexistinglabels;
label-efficiencysimulation, not80totalinputs oractualannotationeffortsaved.
Current48developmentforcheckpointselection; current96quarantinedfromfurther
tuning now forfutureseparateversionedretrospectivejointendpoint. No012evaluator
restart orledgeroverwrite. Existing538-trainedweightsforbiddentolowlabel
initializers/teachers/embeddings; freshmodelsmandatory exceptsharedMediaPipeprior.

Added ideas/013_evidence_profile README/ALGORITHM/EXPERIMENT_PLAN(draft)/feasibility
andLesson17. Definesfixed-budget samplerguards,confidencematched/native-subset
controls,embedding/random/matchedcoverage/weightmass baselines,totalcompute,
rareclass rejectionandcostaccounting. StrongSSLadaptationprotocolrequiredbefore
broadclaims. Exactimplementation/freezebefore realfits; no massmodelsearchnow.

Added puremath research/experiments/evidence_profile_demo.py; eight handconstructed
checks pass, archive evidence_profile_design/20261016T010003871383Z(UTC,localOct15).
Checks conditional50/50ambiguity,146channelpartition,signs,equalconfidence/different
profileadmission,staticshapesupport,sparseanchorunsupported,andexplicitconfident
shared-errorcounterexample. Finite-poolidentity Rw-R=Cov(w,error)/mean(w) verifies
whenweightinghelps; it isstandardalgebra,notnewprooforrealaccuracyresult.
Demo readsno realrecordings/hiddenlabels/models. No actuallearningprocedurefitted,
no newdependencies, data/label/weight/frozencoresourcechanges orcameraactivation.
Next implementsmallcontrolledpilot andfreezelearner/dataaccess/config beforefits;
retainnegativeoutcomes, no more uservalidationdatarequest.

Finalcandidate013audit: consolidateddesign independentlychecked forconditionalCE/
signature/anchor/distance/normalization consistency; no mathematicalblocker, exact
matchedmass/computecontrols still tofreeze beforefits. Added separate80/160/405
total-input draftcurve atfixed80labels; UemptyhaszeroSSLsignal. All586 priorfit
rawhashes+96prospectiverecordings unchanged,115frozen source/modelguardspass,
newdocumentlinksresolve. Designstatus/source/artifacthashes archived. No new
accuracy result; eight syntheticmath checks only.

## 2026-10-15 — pilot013 implementation authorized; existing data first

User authorizes next step, requests focused paper-relevant tracking and offers
optional30/class new checking clips. Choose the existing-data mechanism pilot
first; no new recordings or validation-to-training transfers. Corrected premise:
motion's uniform benefit did not replicate in012;325 are label-hidden TRAINING
inputs, not checking/test clips. All8 classes guaranteed10 revealed labels/seed.

Implemented new isolated learner/runner/auditor and a concise execution protocol.
65 planned fresh fits:5seeds×(4teachers+8sparse variants+1full405reference).
Controls freeze matched predicted-class count/weight multiset, nearestembedding,
ordinary/full versus subset teachers, sparse80 and1680-update supervised80.
The long control matches optimizer updates, not full compute. Teachers shared
physically but charged per method. Native masking, partial ambiguity, cross-session
anchors, one frozen target round and no hidden-label selection remain as proposed.
Seven synthetic software checks pass in.495s: missingness/context/nonmutation,
logodds signs/invariance, forbidden hidden-label API, exact matching, unsupported
anchor refusal, paired tiny fits/zero-coverage equivalence/strict checkpoints,
and incomplete-fit endpoint refusal. No real results viewed yet.

Next prepare hashes/config/sampled rows/source archive before real fitting.
Existing48 selects checkpoints;96 retrospective endpoint and hidden-label purity
wait until all65 candidates/targets finish. Preserve experiment012 ledger. No
new lesson file: authoritative code/config/complete result tables plus concise
LOG/current context will track this work. No dependency or camera activation.

All65 fits completed, including separate full405 references after the60 low-label
fits. No threshold/source/config changes. Native teachers agree on182–204/325
inputs per seed, but confidence>=.95 admits0/2/1/0/0; profile admits0/1/0/0/0.
These are training-prediction/revealed-anchor diagnostics, not hidden-label truth.
Target tensors and every checkpoint now hashed in fit_freeze.json. Proceed with
one separate retrospective96 endpoint access and post-freeze oracle diagnostics;
retain low-coverage failure. No experiment012 ledger changes.

Pilot013 complete:45 students scored on existing96 only after all65 fits/targets
froze. Accuracy supervised80 74.17%, long80 74.79%, subsetconfidence74.79%,
profile/matchedconfidence/matchedrandom74.38%, full405 93.33%. Primary profile−
subsetconfidence macroF1−.002873,0/5positive(fourties). NONE→motion profile68.33%,
full405 8.33%. No useful advantage, no model activation.

Gate admits almost nothing: native confidence0/2/1/0/0, profile0/1/0/0/0. Post-
freeze oracle: profile retains one correctDOWN, rejects confident NONE→DOWN and
one correctBACKWARD. One retained case cannot establish precision advantage;
emptyprecision null. Matchedselectors tieclassification despite differentchosen
case; negligible unlabeled loss and active/no-active dropoutstreams limit causal
interpretation of tiny differences. No threshold changed or favorable seedchosen.

Verified65checkpoints/7440savedvectors, all5target/selectorbundles,549reextracted
rawfeaturearrays. Sevensoftwarechecks pass.65 numericalfitloops152.90s total;
standalonechargedprofile~6.24s versus sparse/full supervised~1.93s, excluding
setup/IO/oldcosts. No memory/inferencesaving measured. PNG/PDF inspected; all
class/error/purity/cost tables and a posthoc stylecoverage audit saved. Original
weights/data/012ledger preserved. Added one concise finding013, no lesson file;
currentcontext condensed, history retained here and inpriorfindings.

Recommended next declared factor:160 revealedlabels (20/class) fromsame405,
245labelhidden; nest80subset andholdotherchoicesfixed. Notexecuted. User'soptional
new30/classcheckingoffer unused; morecheckingwon't supplymissingtraining signal,
validationtotraining wouldobscurecomparison. No significantfinding/novelty/SOTA/
label-saving/reliability successclaimed fromthisnegativepilot.

Completion audit: new65model/selector/endpoint/source guards and prior115freeze
intact; zero-admission studentweights exactlymatch supervised80. CSV45model/360
class/30diagnosticrows anddocumentlinks checked; artifacthashmanifest/completion
verification saved. Posthocseedcoverage: all8labels but seed22 lacksindexUP in
its10UPs; severalteacher5s lackvariants, recordedwithoutretroactiveselection.
Same fixedpermutations at160 retain80subset andcoverall3UPstyles ineachteacher
for5seeds; feasibilityonly, no160fit. No newlesson or cameraactivation.

## 2026-10-16 — pilot 014 declared before fitting

User authorizes the next controlled step. Execute the nested 160-label support
comparison using the same 405 original inputs: 20/class, 245 label-hidden training
inputs. Each teacher/anchor group has 80 labels, 10/class. Same five seeds,
architecture, 0.95 confidence gate, masks, selectors and training recipe as 013.
The change strengthens student/teacher/anchor coverage together; it does not
isolate teacher quality. No new recordings or validation-to-training transfer.
48 development labels select weights; 96 previously inspected clips remain a
retrospective comparison. Training plus selection accesses 208 labels.
65 fits planned; all low-label fits before reference access, all fits/targets
frozen before endpoint/oracle diagnostics. Preserve/hash parent artifacts and
use a separate run/ledger. Twelve software checks passed before real fitting.
Primary: profile minus same-teacher confidence macro F1 across all five seeds;
record every control, admission stage, pseudo-error denominator and cost.
Frozen declaration: EVIDENCE_PROFILE_BUDGET_PROTOCOL.md. No gate rescue, new
lesson, model activation or claim of gain before results.


Pilot014: all60 low-label fits completed before reference405 access; all65
models/targets now frozen. Profile admissions1/0/32/4/0 across fixed seeds.
No cutoff, source or sampler changes. Full405 reference weights reproduce all
five parent state hashes exactly. Proceed with separate retrospective endpoint
access and post-freeze oracle diagnostics; no existing ledger is restarted.

## 2026-10-16 — pilot 014 complete

All65 fits and separate retrospective96 scoring complete; no gate/source/data
changes. Profile accuracy87.29/macroF1.8699 versus native confidence87.71/.8710,
embedding88.54/.8814, matched confidence87.08/.8672, matched random87.92/.8736,
supervised16086.25/.8580, full40593.33/.9326. Primary paired macroF1 -.001118:
one positive, two negative, two tied. No supported main advantage.
Profile admissions1/0/32/4/0; zero recorded-label errors across37 seed–clip pairs
(35unique clips). Confidence5/89errors, embedding0/39, matched confidence1/37,
matched random2/37. No UP/NONE pseudo-label accepted. Target purity alone did not
establish student benefit; empty precision undefined, repeatedpairs dependent.
Profile NONE-to-movement25%, UPrecall66.67%, STOP100% on12uniqueclips only.
12softwarechecks pass;65checkpoints/7440vectors/5selectors/549rawarrays verified;
all parent artifacts intact and full405 states match in all five seeds.
65fit loops174.89s; standaloneprofile~7.06s vs supervised160~1.94s, excluding
setup/I/O/prior work; no efficiency or annotation-saving claim. Figure inspected.
All controls and unfavorable results in findings/014_evidence_profile_budget160.md
and the frozen run report. No new recording, lesson, dependency or activation.
Next candidate mechanism/coverage diagnostic is not declared or executed; no
further budget/gate sweep. Existing endpoints remain retrospective.


Pilot014 completion audit: all frozen guards/parent artifacts intact; 19
zero-active sparse fits exactly equal supervised160; active unlabeled sampling
paired. CSV counts,35unique/37profilepairs and documentation links verified.
Complete artifact hash manifest and completion verification saved. No further fit.

## 2026-10-16 — diagnostic 015 declared before computing scores

User authorizes the next coverage/information diagnostic. Use saved pilot014
teachers/anchors and all five245-hidden-input training pools. No new training,
recordings, dependencies or 48/96 endpoint feature/score access. Original .95
selector is reproduced, not modified. Frozen plan:
EVIDENCE_PROFILE_DIAGNOSTIC_PROTOCOL.md. Designed after013/014 outcomes, hence
retrospective development, not an independent preregistered test.

Trace each true/predicted class through agreement/confidence/anchor/radius gates.
Compare dense profile, embedding, confidence and fixed-seed random rankings on
identical agreed/common-supported candidates. Primary fixed25% ceil quota per
predicted class; .10/.50/1.0 descriptive only. Additional25% control matches
predictedclass x fixedconfidencebins; report exclusions and all denominators.
Infer saved supervised160 models on hidden training inputs only, for a
redundancy proxy. Recorded-label ranking and already-correct examples do not
establish causal learning utility. No fitted correctness calibrator or gate sweep.
Seven diagnostic software checks passed. Freeze scores/selectedIDs/student
probabilities before separate oracle analysis; all parent files remain pinned.
No next student fit or successful algorithm claim authorized by this diagnostic.


## 2026-10-16 — diagnostic 015 complete

No new fit, label, dependency, checkpoint change or endpoint rescoring. Saved014
models/targets examined only on1225hidden-training seed–clip pairs/385uniqueclips.
Original selector weights reproduced. Rankings/score/student probabilities froze
before separate oracle analysis; prior outcomes already known, retrospective only.
UP:100/150ensemble correct,61correctagreements,0confidence admission (correct
agreement scores .2378–.9120). NONE:25/105correct,5correctagreements;4confident
wrongDOWN proposals rejectedbyprofile. Agreement846,sharedsupportedpool627;
219agreed exclusions reported. Missing support still limitsUP/NONE coverage.
Fixed25%-classquota: confidence7errors/164,profile5/164,embedding1/164,random17/164.
Coarseconfidencebincontrol:21/192,20/192,14/192,22/192. At50%, confidence15/320
beatsprofile23/320 andembedding17/320. All10/25/50/100points retained, no chosen
favorable coverage. No reliable response-specific ranking or training advantage.
All37originalprofile targets already correctly recognized by supervised160,
35highconfidence; this is only a redundancy proxy, not causal utility proof.
Seven softwarechecks pass;5scores/anchor bundles,180selections,1225studentvectors
reproduced. Runtime NumPy allowlist blocked oracle/dev/endpoint during scoring;
training-call trap active. Parent files intact. PNG/PDF figure inspected.
Source/scores/config/data-access audits/fulltables in frozen diagnosticrun;
findings/015_evidence_profile_diagnostic.md is the concise paper-relevant record.
Next candidate direction is class coverage and learning value versus strong
confidence/embedding baselines; no additional gate/budget/student run started.
No novelty, SOTA, independent generalization, robot or label-saving claim.


Diagnostic015 completion audit: independently recomputed all180 selection
error counts and pooled summaries, stage partitions and CSV rowcounts. Parent
source/artifact guards intact; all local report links resolve. Seven final
software checks pass. Artifact hashes/completion verification saved; no new fit.

## 2026-10-17 — experiment 016 declared before fits

User asks to investigate missed internal training/validation methods and recalls
viewpoint gains. Correct distinction: earlier camera/hand coordinate representations
and motion, not randomized physical camera viewpoints; later gains were mixed.
Test baseline conditioning before further pseudo-label algorithm changes.
INPUT_CONDITIONING_PROTOCOL.md freezes2x2 raw/RMS x clock/no_clock conditions.
Same160labels/20class,245hidden, teacher80each,405original inputs,48development,
96retrospective endpoint. Seven positive per-block scales fit only own labeled
training clips, validity-aware/equalclipweight/floor.05/no centering. No held-out
statistics, rotations or changed direction labels. Remove only clock144/145;
velocities remain physical-time features.60fresh fits, same149-input LSTM/49,840
weights,560updates and paired role init/batch/mask streams. Original15raw states
must reproduce014 before evaluation. All60 freeze before new separate evaluation.
Primary RMS_clock-minus-raw_clock student96macroF1, allfive seeds/sittings/classes;
secondary clocks, teacher heldout recognition, hidden admissions and gate saturation.
No SSL student, learningrate/gate sweep, new labels/dependencies/activation/lesson.
Seven softwarechecks passed before fits, including LSTMgate reconstruction and
training-only scale fitting. Generic normalization is not an invented algorithm;
results and unfavorable factors remain in the record. Novelty/generalization,
physicalviewpoint effects, independent testing and real cost savings unproved.


Experiment016: all60 fits and normalizers frozen before new retrospective
evaluation. All15 raw teacher/student controls exactly reproduce014 states.
Runtime fitting allowlist loaded only five learner bundles and48-selection data;
no hidden oracle/endpoint payload. No factor, floor, gate, source or optimizer
change. Proceed with the separately ledgered evaluation; all old runs intact.

## 2026-10-17 — experiment 016 complete; accuracy/background trade-off

All60 fits/evaluations complete. Seven softwarechecks passed;60checkpoints,
17800vectors,15own-label-only scale estimates and paired role streams verified.
All15raw states exactly reproduce014; parent files intact. Runtime train allowlist
loaded only five learner bundles and48-selection arrays, no oracle/endpoint.
Student96accuracy raw_clock86.25%,rms_clock87.08%,raw_no_clock86.25%,rms_no_clock
85.42%. MacroF1 .8580/.8636/.8594/.8488. Primarypaired RMS_clock-minus-raw_clock
+.005575,3/5positive; sitting1-.003072,sitting2+.015408. StudentNONE→movement
28.33/43.33/23.33/48.33%; slight accuracy gain is not a robotics upgrade.
RMS_clock UPrecall75vs66.67%, NONErecall56.67vs71.67%; teacheranchorF1 declines
.7265→.6968,hiddenmeanF1.7429→.7118. No predictedUP/NONE confidenceadmissions
in any condition. Clockremoval alone macroF1+.001359,only1/5positive,accuracytied.
Gates not uniformly less saturated; no causal saturation explanation established.
60fitloops116.97s excluding other costs; no efficiency/annotation-saving claim.
Allfour factors/seeds/classes/sittings plus264errorrows/1440pairedcaserows saved.
Figure inspected. Findings016 is the concise record; no lesson/dependency/data/
activation. No next fit declared/executed. Background recognition/class-specific
learning value remains the next candidate direction. Viewpoint gains notisolated,
normalization notnovel, all96scoresretrospective and no generalreliability claim.


Experiment016 completion audit: independently checked all admission/class counts,
sitting metrics, raw-control endpoint/teacher probabilities and CSV summaries.
60checkpoints/17800vectors/15scales/15raw-parent states and parent hashes pass.
Source/config/model/normalizer freezes intact; report links resolve, figure
checked. Full artifact hash manifest/completion verification saved. No further fit.


## 2026-10-17 — review/diagnostic017: primary literature before more fits

User requests primary-paper backing, careful compute and focus on low-label robot
command learning. Reviewed relevant ICML/ICLR/CoRL methods and explicitly marked
abstract-only/inaccessible sources. SoftMatch/FreeMatch provide simple comparison
targets; Ren2018, MPL, PabLO2025 and IA-SFT2026 constrain novelty for usefulness/
error-aware admission. KnowNo/CRC inform decision metrics, not guarantees here.
R3M supports reuse but does not justify a large new pretraining run on245clips.

Declared six static prediction-only selector components before new weights and
oracle summaries. No training, forward passes, rawclips or48/96prediction reads.
Run published_selector_replay/20261017T072612284949Z used allfive saved014/015
seeds/1225hidden-training pairs/385uniqueclips. Originaldual95 89mass/5wrong,
ensemble95 135/8, global656/56, class984/184, Gaussian878.73/112.15, GaussianUA
796.64/107.22. CorrectUPmass0/0/31/99/66.86/72.23; classsnapshot admits90actualNONE
including58movement targets/7wrongSTOP. GaussianUA trueNONE mass51.87 includes
34.54movement. More coverage also brings noise; extra correct targets on student
mistakes are only a utility proxy. No ranking of trained algorithms is possible.

Seven checks passed. Initial equal-confidence analytic check caught nearzero
variance roundoff before study scoring; documented limiting rule fixed it. Scalar
independent recomputation matches all1225pairs×6rules; original5masks/6parentinputs/
32olderPython sources unchanged. Weights/code/plan/input hashes freeze before new
oracle access, but data/outcomes alreadyinspected: retrospective. Replay .039s
excludes implementation/review/audit. Allperseed/class results/access saved.

No threshold or new algorithm promoted. Next declared gate: landmarkview/method
implementation audit, then capped15studentfits(3methods×5seeds) for supervised160,
commononlineconfidence andSoftMatchadaptation.560updates/twoCPUthreads/no sweep;
notstarted. Use48development only for continuation: meanmacroF1gain,3/5positive
paired differences,no meanNONE→movement increase orSTOPrecall decrease. Report
fixedfinal and selected checkpoints/fullcosts. No96selection, validationgradients,
newrecordings/dependencies/cameraactivation/novelty or costsaving claim.

Finding017 holds evidence/limitations/nextgate instead of a lesson file. Updated
CONTEXT/PROTOCOL/README and rewrote stale RESEARCH_DIRECTION.md to current scope;
prior262-line direction text preserved in run017/prior_research_direction.md.

017 final preflight addition: the published SoftMatch m=.999 retains57.10% of
initial statistics after560updates versus its2^20-update image recipe. This
closed-form check motivates a documented statistics-timescale audit before any
fit; no static diagnostic rule/weight changed. No new model trained.


## 2026-10-20 — experiment018 declared before fitting

User authorizes training as needed. Fixed ONLINE_SOFTMATCH_PROTOCOL.md:15fresh
fits, supervised160/onlineconfidence/SoftMatchadaptation across5paired seeds;
raw149LSTM, same014160/245splits/20labelsperclass,560Adam.001updates,batch32,
twoCPUthreads. First140supervised plus420joint,140ramp. Identical landmark inputs;
deterministic weak/head-dropout strong; private label/strong dropout RNG avoids
extra-pass stream confound. This is a controlled adaptation, not exact papers.
Paper Algorithm1 rawstatistics ordering chosen explicitly over currentUSB
aligned-hook ordering. Bootstrapfirstbatch; onehiddenpool-pass half-life
m=2^(-32/245), not oracle-tuned; no parameterEMA for anymethod. Nine checks pass,
including zero-mask gradients, fullbatchnormalization, rawtargets, finiteweights,
inputbytes, RNGisolation and forbiddenpayloads. Initial scalar test expectation
was corrected for intended float32 underflow before fitting, no algorithm change.

All15fits/30selected+final checkpoints/traces freeze before separate training-
oracle analysis. Trainingallowlist learner+48selection only; no96rescoring.
Continuation gate from017 uses48selectedmacroF1/3positive/NONE/STOP, with matched
supervised and fixedfinal checks reported separately. Source/config/parent hashes,
perworkerRSS/time and actualforward counts recorded. No sweeps/newlabels/deps/
activation or noveltyclaim. No fit started at this declaration.


## 2026-10-20 — experiment018 complete; selected gain and final reversal

Run online_softmatch/20261020T205543410735Z completes all15fits, no retries/sweeps.
All30selected/final states and trainingtraces froze before separate oracleerror
analysis; no96payload loaded/rescored. Same160/245/all8commands/no405warmstart.
Selected48 accuracy sup85.00/conf86.25/soft87.50%; F1.8456/.8575/.8721.
SelectedNONE→movement30.00/20.00/13.33%, STOP86.67/86.67/90.00%.
Predeclaredgatepasses: F1+.014560 vsconfidence, paired differences
[-.045688,+.073443,0,+.021212,+.023830],3positive/1negative/1tie; NONE-6.67pp,
STOP+3.33pp. Versussupervised +.026436 but2positive/1negative/2ties only.

Final560 reverses F1: sup.8370/conf.8950/soft.8818, accuracy84.17/89.58/88.33%.
SoftfinalSTOP83.33vsconf90%; NONE13.33vs16.67%. No endpoint cherry-picking or
retroactive gate change. Softselectedsteps320/560/100/160/120; seeds33/55 predate
SSL. Seed55 selectedgain retains supervisedcheckpoint, not a better classifier
learned frompseudo-labels. Sixuniqueclips/class limit inference; no significance,
independence/robotreliability/novelty/realannotation-saving claim.

All8pseudo-targetclasses receive mass acrossfiveseeds. Confidence vssoft training
mass57475/62255.97, wrong7074/9283.35 (12.31/14.91%), correctUP5524/5906.43;
actualNONE→movementtargetmass1606/2168.46. Allphases/class axes and ramp-adjusted
mass retained. Repeatedexposures/evolvingdifferentmodels preclude causal selector
utility or fixedcandidate-quality ranking. Higherpurity not sufficient for gain.

Ninepreflight checks pass; all5supervisedselected states exactly014; pairedinit/
warmup/labeldraw/dropout streams. Verification reproduces30checkpoints/1440dev
vectors/134400onlinetargetweights+hidden draws; independentmanual30confusions/
67200chronological scalarsoftweights agree. TrainingNumPyallowlist logged only
matching learner+48selection; allparent014 files pinned/unchanged. Inputs/code/
plan/config/preflight/model/trace freezes intact. OlderPython sources untouched.
Fitloops48.74s incldevchecks, workerwall65.39s; setup/review/audit additional.
Peak340.25MiB. Trainingforward samples89600supervised versus224000eachSSL group;
equaloptimizersteps not equalcompute, no generalefficiencyclaim.

Saved finding018, allseed/class/failure CSVs, scientificPNG/PDF (visuallychecked),
fullhistories/traces/access and hashes. Updated CONTEXT/PROTOCOL/README/direction;
prior direction archived in018run. No lesson/dependency/newrecording/activation.
Research consequence: establish online baselines and checkpoint sensitivity,
retain both confidence/soft. Next publicmulti-person benchmark compatibility/
split/compute audit and evaluationrule freeze before further methodselection.
No additional localfit or96rescoring to rescue the outcome.


## 2026-10-30 — GitHub preparation, source modules and research outcomes preserved

User requests necessary code/folder organization and explicitly will perform the
commit/push after guidance. Actualprojectindex/branch/remotes left unchanged;
no projectcommit, GitHubpush, artifactupload or new researchtraining/evaluation.

Added docs/GITHUB_PUSH_GUIDE.md andREPOSITORY_LAYOUT.md, researchrequirements,
separate tools/export_research_records,check_repository,run_software_checks and
packagingtests. Historical experimentPython/paths notrefactored: pinnedsource
hashes remain valid. .gitignore protects allrawtensors/weights/runtime/private
notes; .gitattributes preservesLFsource and exact hashedexportbytes. Root/research
README and20finding pages link portable records; oldlocalarchivelinks distinguished.

Exported20historical experiment/evaluation records (634files/8,671,020bytes) to
research/results. Configs, allavailable seedmetrics, negativeoutcomes, approved
plots andvalidationloss histories retained. 594exported payloads compared against
original JSON/CSV/figures: numericvalues preserved; only localpathstrings made
portable. Original source/freezes pinned bySHA; copies are not replacement frozen
inputs. Fullreproduction still requires privatearchives/originalpath assumptions.
Localignored artifactinventory covers5613files; alloriginal hashes unchanged.
Nooffdevicebackup configured/performed (storage_uri null). Full privateerror lists,
biometricrecordings/viewerbundles andweights are not included inordinaryGit.

Clean-copy tests initially exposed3legacy mockedcamera tests depending on the
presenceofdownloadedhandmodel. Added temporary test-only assetfixture and real-
device guards innewrunner; nooldapplication/testsource edits, no fake model placed
inmodels/. Fixed a packagingunit test for symlinkedtemp-rootpaths and a scanner
falsepositive onredactionregex before finalverification. All198syntheticchecks
nowpass in a copywithnoprivatedata/runs/notes/detectorweights. Tests contain tiny
syntheticfits, not newpersonaldata researchruns. Initialfailurelogs retainedlocal.
ScratchGitstaging checks verify actualindexbytes, forced-exclusion policies and
whitespace; actualprojectGitindex still empty, branchSRB-hand-detection unchanged.
Basic credential/homepath/exporthash scans pass; stageddiff review remainsuser's
finalstep. Publicorigin/private-remote option and backupinstructions documented.
Mainresearch objective, modelstatus andnextpublicbenchmarkaudit unchanged.
