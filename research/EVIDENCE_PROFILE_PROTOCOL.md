# Pilot013 execution protocol — fixed before real fits

2026-10-15, America/Chicago. User authorizes starting the idea and asks for
focused, auditable progress. Continue existing data first; the offer of30/class
new checking clips is optional, not collected or needed for this pilot. No
validation recording enters training. No new lesson files; retain code, input
identities, frozen settings, complete outcomes and concise current context.

The405 original training clips contain F86/B63/L46/R41/STOP44/NONE41/UP50/DOWN34.
Exclude133 model-selected corrections from primary pool; preserve all files.
For each seed11/22/33/44/55 reveal five/class in each of the two fixed whole-
session groups (250/155). Thus80 labels,325 label-hidden training inputs, every
command in both teachers. This is not80 training plus325 test. Hidden labels
are kept in a separate oracle artifact and excluded from the learner API.
Existing48 choose checkpoints; existing96 supply a later joint retrospective
endpoint. Count all144 evaluation/development labels separately;80+48=128 labels
available for training/selection. All endpoints already inspected historically.

Reuse exact camera-motion146-feature arrays and preprocessing from experiment011,
pin source/raw hashes and re-extract during verification.149-input LSTM47 with
four ordered bins/64-unit head,49,840 parameters, same direct-comparison model.
No538-label checkpoint or representation is loaded as initialization. MediaPipe
pretraining is common upstream knowledge, disclosed. No new dependencies.

65 fits: five seeds×(four teachers+eight80-label student variants+one405-label
reference). Four teachers are two session groups×full-only/native-subset input.
Each teacher has40 labels, anchors are the opposite40. Native masks full1/2,
omitP/A/G each1/6, explicit availability bits, shared clocks/validity. Same-side
teacher initializations and labeled batch streams pair across full/subset fits.
Same student initialization and labeled draws across compatible student variants;
active-SSL variants share unlabeled draws. Dropout streams need not pair between
supervised-only and active-SSL forward-call patterns. Zero-admission SSL skips
its unlabeled forward and reproduces the sparse supervised fit exactly.

Standard recipe560 Adam steps,.001,batch32, weighted labeledCE,28 full-input
unweighted selection-loss checks every20, earliest strict minimum. No augmentation
or search. Student loss labeledCE+lambda*mean325(weight*softCE); replacement
unlabeled batches32 estimate that fixed-pool mean, rejected weights0. Ramp0→1
over140 updates. Two teachers+targets frozen for one student round. Allseeds saved.

Students: supervised80; supervised80_long; selftrain_full_conf;
selftrain_subset_conf; profile; embedding; matched_conf; matched_random.
All use80 labels. Full405 reference is fitted only after all low-label targets
and students finish, so its label read cannot feed any low-label decision.
The long supervised control uses1680 updates,28 checks every60: it matches
teacher+student optimizer-update count, NOT exact FLOPs, runtime or selector
exposure. SSL student has additional unlabeled forwards; total costs reported.

Confidence gates both full teachers agree and each max>=.95; average soft target.
Profile uses signed one-vs-rest log-odds changes full-minus-omit(P/A/G), same frozen
teacher on candidates and opposite-session anchors. Retain correctly predicted
anchors p(true)>=.5, minimum3/class/teacher. Global perteacher MADscale floor.05;
nearest-class squared standardized distance, median leave-one-anchor-out nearest
distance radius floor.0025. Unsupported means weight0, not a NONE label. Both
teachers must pass; weight=min(exp(-D/R)), otherwise0. No partial-view agreement.
Embedding baseline uses penultimate64 features and the identical scale/radius/
anchor rules, isolating response-profile versus embedding support.

Matched controls start from native confidence-eligible candidates. For each
*predicted* class, take the profile variant's exact positive-weight multiset.
matched_conf assigns it in descending order to the same count of highest mean-
teacher confidence candidates (rowID breaks ties). matched_random assigns that
multiset to a fixed seeded random candidate order. No hidden truth determines
counts, ranking, weight or pseudo-target. Zero class admission gives both0.
This matches count and mass/class, not a new annotation-budget advantage.

Teacher fit costs reused physically but charged to each standalone method.
Measure target generation/selector envelope and student times, all steps/forward
budgets, parameters and process memory limitations. Selector prototype computes
some unused embeddings/views for shared diagnostics; common subset-selector
envelope is a conservative measured charge, not an exact perselector FLOP claim.
No training-efficiency conclusion from update counts alone.

Freeze code/config/sampled IDs and input arrays before fitting. After all65 fits
and target hashes freeze, a separate new ledger permits96 endpoint scoring and
oracle pseudo-target diagnostics. Do not restart experiment012 evaluator or
overwrite any previous score. No endpoint/hidden-label cut-off tuning or stopping
to select favorable seeds. Preserve failures, empty admission and regressions.

Primary: mean macroF1(profile−selftrain_subset_conf) on96, paired allfive seeds;
report each48 sitting separately. Also compare matched_conf/random, embedding,
sparse/long/full references, everyclass, UPstyles after oracle release, STOP/NONE
false movement, NLL, admission/weight and pseudo-target error at matched coverage.
No population p-values from shared clips/seeds. Empty precision is null, not100%.
Higher pseudo-label purity alone does not establish better student learning.

This is a mechanism pilot, not an exact FixMatch/FreeMatch reproduction, SOTA,
novelty finding, actual annotation saving, unseen-user reliability or robot trial.
No student activated in camera. Additional label/input budgets and public-benchmark
baselines require a separate declaration after this complete outcome is recorded.
