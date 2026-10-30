# Experiment 011 — extend motion representations across models, then controls

Declared 2026-10-15 before new fitting or historical performance scoring.
User requests other models first, all test datasets validated, then the next
controlled steps. Optional clarification offered: score compatible old cohorts
as retrospective development checks (default), or audit integrity only.

**Clarification before fitting/scoring:** The user delegates the choice to us,
prioritizing proper validation without false claims. Choose audit-only for old
test performance: check all historical raw datasets/roles/overlap/semantics,
preserve their original scores, and prepare genuinely new prospective sessions
after candidate definitions/weights are frozen. Do not rescore old test cohorts
in this experiment. The retrospective-scoring design below remains a documented
unexecuted alternative; all new comparison scores use only the original v3
development validation. This supersedes the initial default.

## Fixed comparison, data and claims

Reuse experiment010's exact version2, 32×146 arrays/538 training/48 validation
rows and all20 saved LSTM checkpoints. Four camera/hand × position/motion modes.
No raw data/label changes, new dependencies, architecture-specific search,
best-seed choice or camera activation. These are adaptations of established
families to the new input; experiment008's old67-input models are context only.

| Family | Fixed architecture | Parameters |
| --- | --- | ---: |
| CNN | Conv1d146→36→48, k5/pad2; pool4; dense192→72→8 | 49,484 |
| LSTM | Exact saved experiment010 146-input/47-hidden/four-bin network | 49,276 |
| BiLSTM | 25 hidden/direction; pool4; dense200→72→8 | 49,656 |
| Local set | Frame MLP146→128→131; mean; dense131→96→8; global channels zeroed | 49,163 |
| Feature set | Same frame MLP, all146 inputs; no recurrent/convolutional order | 49,163 |

Local set masks position63:66 and derivative130:133 (global wrist/size) but keeps
local articulation/motion, orientation, validity and clocks. It is not the old
pose-only set. Both set families are permutation invariant over already computed
rows, but explicit velocity, initial-reference displacement and elapsed time can
encode temporal information. Do not call them motion-free or order-information-free.
ReLU follows hidden dense/conv layers, dropout .2 precedes final output. Shared
560 Adam updates/.001, replacement batches32, class-weighted training CE;
every20 updates choose earliest strict minimum unweighted v3 validation CE.
Seeds11/22/33/44/55, CPU threads2. Equal updates/parameters is not equal compute;
record actual fit/forward times and total process RSS with its limitations.

80 new core fits (four families×four modes×five seeds),20 exact saved references.
Reuse the unchanged numeric fitting procedure with an explicit model factory;
pair sampling streams. Initial states match across modes within a family; they
cannot match across different architectures. Preserve all outputs and failures.
Primary: camera-motion minus camera-position paired macroF1 within each new
family, all five seeds. Report all four modes, class metrics, STOP/NONE errors,
sample SD, paired differences, training costs and frozen-prefix agreement.

## Dataset audit and retrospective evaluation

Inventory metadata across canonical raw-data locations. Exclude label-revision
archives and dataset copies from independent cohort counts. Hash and validate
shapes/times/visibility/convention; compare raw identities and session lineage
against all538 training clips. A formerly-validation session reused as training
is ineligible for evaluation, even if a particular row wasn't sampled.

Default evaluation cohorts, with per-session outcomes retained:

- current v3 validation48, all8 classes, checkpoint-selection development;
- original legacy validation, only sessions/identities outside current training;
- historical legacy test52, original six classes, already inspected;
- v2 validation88, exclude all old UP rather than infer which match new UP;
- inspected boundary test80, exclude all old UP (10); keep physical sitting1's
  interrupted capture IDs grouped, sitting2 separate.

Use canonical current records, never the obsolete label-revision copy. Historical
STOP reuse relies on the qualified human report of mostly open palms, not a new
independent verification of every old label. Report this semantic caveat. Other
old classes are common-label retrospective checks, not full v3-task final tests.
No validation/test enters training or unlabeled pretraining. Raw/session overlap,
excluded counts/reasons, original labels, quality failures and duplicate lineage
must be explicit. Preserve one-time test ledgers and their past results unchanged;
new scores live in a separate retrospective archive authorized by this request.

Inventory correction: v2 validation contains10 old UP clips, leaving78 common-
label records; an earlier expected count of20 was a documentation arithmetic
error. The audit always classifies actual metadata labels and was unaffected.

Score all saved comparisons only after core fits/settings are frozen. Do not tune
on these outcomes or label them untouched. There are no genuinely new v3 final
test sessions in the current inventory. If audit-only is selected, skip performance
scoring of historical cohorts and retain integrity/compatibility reports instead.

## Then the predeclared controls (15 fits, camera_motion only)

1. Causal temporal CNN: left-padded k3 layers with widths40/48/48 and dilations
   1/2/4, pool4, dense192→96→8 (49,632 parameters). Past-only layer receptive
   field15 samples, final dense head combines all bins. This is a small adapted
   baseline, not a claimed full reproduction/new TCN method. Causal layers do not
   make prefix resampling incremental/cached or validate early robot actions.
2. Feature set with elapsed-time channel144 zeroed, interval145 retained.
3. LSTM with the same elapsed-time mask. Compare these two masked families to
   investigate sequence memory without explicit per-frame clock positions.

Same five seeds/recipe, no extra tuning. Elapsed-masked models still receive
derivatives and first-reference trajectories; this does not remove all local
temporal information. Test set permutation invariance and causal feature-stack
prefix independence mathematically, without treating synthetic accuracy as evidence.
All family/condition definitions above fixed before viewing retrospective scores.

## Integrity and next physical data

Snapshot code/config/features/protocol before fitting. Strict new research-only
checkpoint loader. Reload every saved model and reproduce full/prefix vectors,
all row labels, metrics, selection minima and pairing; preserve reference weights
and all originals. Prefix .5/1/1.5s inputs are independently recomputed version2
arrays from experiment010: original-intent agreement, not validated early accuracy.

After these controls, freeze a small declared candidate set before genuinely new
development sessions/person/view/speed collection. Continuous early-decision work
needs onset/offset/non-command intervals, WAIT/defer and coverage/false-actions/
STOP-delay metrics. Main low-label/audited learning study is still separate; no
novelty, SOTA, robust generalization or annotation-saving promise follows here.
