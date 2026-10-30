# Draft next experiment — existing-data feasibility of response-based self-training

Prepared2026-10-15 after the user rules out further personal validation/test
recordings. **Draft design, not an executed experiment or final preregistration.**
Next implement the sampler, label-access boundary and native-mask learner; freeze
its exact configuration/source before fitting. Do not run a large search now.

## Scope and roles

- Primary pool:405 original recordings from the current538 training manifest.
  Counts F/B/L/R/S/N/U/D=86/63/46/41/44/41/50/34.
- Keep133 model-selected corrections out of the primary pool. They can supply a
  clearly separate full538 practical sensitivity later. No deletion or relabeling.
- Primary budget10/class=80 labels. Remaining325 clips enter as inputs with
  labels hidden. This tests **label efficiency**, not80 total input clips or actual
  annotation money saved. A later input-budget curve must separately cap clips.
- Existing48 current-contract development clips select checkpoints under one
  fixed rule. Their labels count separately, so80 training+48 development labels
  are128 supervised labels before endpoint evaluation costs.
- Existing96 inspected clips: stop further tuning/feature/retrieval/pretraining
  access now and use one joint **retrospective fixed endpoint** after the next
  finite candidate set freezes. All96 already influenced this research. They
  are never described as untouched, even if quarantined from now onward.
- Older class-incomplete/old-UP cohorts retain historical reports and may be
  separate retrospective checks only if explicitly declared before scoring.
  Do not pool them into the eight-class primary result after seeing outcomes.
- No new human reviews or recordings required. Optional future oracle audit
  queries must reveal existing training labels and charge every unique label.

The future runner must write separate access/provenance ledgers; it must not
overwrite or restart experiment012's one-time evaluator. The user's latest
instruction authorizes existing-data development/testing; it does not erase
historical exposure or permit misleading independent-test language.

## Whole-session support

Session groupA=original201 + DOWN24 + firstnewUP25:250 originals,
counts61/38/31/26/29/16/25/24. Full captureIDs:

```text
20260930T062558-33e35afb
20261008T021615-a0b59124
20261012T174044-0244fa51
```

GroupB=remaining155 originals:90 recycled-training + two20-clip routine halves
+secondUP25, counts25/25/15/15/15/25/25/10. Full captureIDs:

```text
20261008T005936-27dde0e5
20261008T063028-1f0a4f92
20261008T065105-55ca7759
20261012T175809-93ca5838
```

Five revealed labels/class/group at primary80; deterministic paired seeds
11/22/33/44/55. Keep every recording/capture session in its assigned group.
No style quota in the primary sampler: up_shape and task hints remain inaccessible
to the learner; style is diagnostic metadata for analysis only. Stratified class
sampling uses oracle labels for benchmark construction, not a claim of automatic
annotation acquisition. Preserve actual sampled row identities before fitting.

UP has only two training sessions. Fully grouped nested eight-class CV is
impossible here; do not quietly replace it with clip-random folds. The grouping
is a fixed class-supported cross-fit design, not independently replicated people.

## Initial finite comparisons

Use a shared small LSTM/four-bin pattern. Adding three availability bits to the
146-input/47-hidden pattern increases49,276 parameters by564 to49,840, subject
to implementation count verification. Use the same149-input architecture in
all direct variants; student availability bits are111. Existing feature extractor
remains frozen; no arbitrary raw-gesture transformations or class semantic changes.

1. Sparse supervised:80 human labels, full inputs only.
2. Ordinary confidence self-training: two full-only teachers, same80 total human
   labels and one student, full-teacher agreement/confidence gate.
3. Native-subset teachers plus the identical confidence-only self-training rule.
4. Candidate: same teachers as3, add the response-profile admission/weight rule.
5. Nearest-embedding admission with those same teachers/anchors.
6. Matched-count and matched-weight-mass confidence/random selection controls,
   preserving each candidate-predicted class's admitted count. Match using
   predictions only, never hidden class truth. Score random controls across the
   same fixed seeds, not repeated until favorable.
7. Fresh full405 supervised reference using the same representation/architecture;
   it is a higher-label reference, not a fair80-label competitor.

These are proposed controls; finalize exact optimizer/sampling/selector exposure
before training. For the minimal pilot, retain560 updates/batch32/Adam.001 per
teacher or student,28 checks on development48. Each teacher receives only its40
revealed labels; student receives all80 plus its allowed targets. Subset masks,
initialization and clip draws pair where compatible. Proposed confidence.95,
anchor minimum3/correct fullprediction>=.5, scale/radius rule and one frozen
target round specified in ALGORITHM.md, not tuned from hidden325 labels.

Teacher resources are shared/reused exactly across3–6 so gains isolate admission.
Teacher/student checkpoints/histories/seed target lists and every failure retained.
Mask regularization and extra compute are established ingredients/possible
confounds. Compare equal-update and equal-total-compute outcomes; do not call
three560-step models as cheap as one560-step model. Record all input views,
teacher forwards, fitting/selection/pseudo-targeting time, memory and student
single-window inference. Prototype overhead can be measured before expanding.

Before any broad SSL/SOTA statement, include carefully specified strong existing
SSL methods such as FixMatch/FreeMatch and a relevant hand-skeleton approach.
Agree their landmark augmentation/pretraining mapping and resources first; a
full-input identity-view self-trainer is not an exact FixMatch reproduction.
Adding methods later is a new declared comparison, not retroactive preregistration.

## Label-access boundary

The sampler/oracle may see current labels to build fixed balanced subsets.
Training code receives sanitized features plus labels for L only. U labels use
a separate inaccessible oracle structure, never a NO_COMMAND placeholder.
Do not expose task_hint, up_shape, filename-encoded commands, session/class
metadata, original_label or target-conditioned selection cues to the learner.

Never initialize with any full538-trained encoder, teacher, student or embedding
bank. That would leak the hidden labels into an80-label claim. Fresh initialization
is mandatory apart from the common upstream MediaPipe detector, whose prior
training/resources are explicitly disclosed. No new package is currently needed.

Hidden325 labels may be used **after targets and model choices freeze** for
diagnostic pseudo-target error/coverage; they cannot choose a cutoff, feature or
winning variant without being charged as revealed labels and changing the study.
Each teacher's anchor inputs/labels are excluded from its own training; no teacher
self-supervised pretraining in the initial pilot. Later self-supervision is a
separate matched ablation, never uses development/endpoint inputs.

## Outcomes and stopping rules

Primary contrast: mean macroF1 candidate minus same subset-teacher confidence
baseline on the fixed retrospective96, allfive seeds and both48-clip sittings
reported separately. Report development48 separately; it selected checkpoints.
No population p-value from repeated clips or seeds.

Secondary: ordinary sparse/full references, perclass precision/recall, UP style
coverage, STOP/NONE→motion, confusion/NLL, fraction/weight of admitted targets,
effective pseudo-label count perclass/session, pseudo-target errors after oracle
release, parameters and **total** training/teacher/inference costs.

Test whether the response weight is actually negatively associated with wrong
pseudo-labels, as required by the finite-pool identity. Compare precision at
matched coverage, not simply cleaner labels after nearly everything is rejected.
Zero accepted cases makes precision undefined; never report it as100%.

Go forward only if a useful benefit remains over same-teacher and matching
confidence/embedding controls, is not solely one endpoint sitting, and does not
increase important false-motion errors. If gains vanish at total-compute matching,
claim no efficiency improvement. If no gain or rare-class support collapses,
record failure and revise the hypothesis in a new protocol; do not reshuffle
endpoints or silently increase labels until a desired story appears.

Later label curves can add20/class160, then smaller/larger nested budgets once
support constraints are audited. At5/class a two-part teacher has2–3/class and
often no admissible anchor bank; it is a deliberate stress test, not a calibration
setting. At30/class, groupB has only10 DOWN so15/15 allocation is impossible.

## Separate total-input efficiency from label efficiency

After the primary mechanism comparison, a separately declared input-budget
check can hold the same80 revealed labels fixed and use nested total pools:

| Total distinct training inputs | Revealed labels | Additional label-hidden inputs |
| --- | ---: | ---: |
| 80 | 80 | 0 |
| 160 | 80 | 80 |
| 405 | 80 | 325 |

Choose the80 hidden-input subset by a fixed random seed without opening its
labels or style metadata. No additional inputs may enter pretraining, retrieval,
teacher fitting or normalization for the capped condition. At80 total inputs
there is no unlabeled self-training signal; define its unsupervised term as0
and report that degenerate case instead of claiming pseudo-label improvements.
Total compute and chosen input identities must be frozen for each condition.
This tests the user's smaller-*dataset* objective separately; the main80-label/
405-input experiment cannot establish an80-input achievement. Do not quietly
expand this secondary curve after endpoint inspection and call it preregistered.

## What a successful pilot would and would not show

It could show that one concrete admission rule uses revealed labels better on
this fixed single-user task, at specified resources. It would not establish
actual saved annotation minutes, a smaller total input dataset, guaranteed robot
behavior, cross-person robustness, a universal training sequence or new SOTA.
General image-network claims are also out of scope: these inputs are pretrained
landmarks, not RGB. A later compatible public benchmark can supply additional
existing-data evidence without new personal recordings; access/protocol/license
must be verified before using one. Novelty remains a separate bounded-search
and experimental question.
