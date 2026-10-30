# Pilot 014: nested 160-label support experiment

Declared 2026-10-16, America/Chicago, before any real 160-label fit. The user
authorizes the next controlled step after discussing the research objective.
Parent: pilot 013, `evidence_profile/20261016T030844123147Z`. Preserve it exactly.

Question: with greater labeled support, does the unchanged evidence-response
selection rule admit useful pseudo-labels and outperform the same-teacher
confidence rule? This is a diagnostic budget experiment, not a new algorithm.

Use the same 405 original training inputs, order and whole-session groups.
Same five seeds and stratified permutations; reveal the first 10/class/group
instead of five. Each teacher has 80 labels (10/class), its opposite-group
anchor bank has 80, and the student has 160 (20/class). The original 80 labels
must be nested in each 160-label set; 245 other labels remain hidden. No style-
based resampling. Audit UP styles descriptively. Exclude the same 133 corrections.

This intervention changes student support, teacher support, anchor coverage and
the labeled/hidden composition together. It cannot isolate teacher quality from
anchor quantity or style coverage. The 405 training inputs remain fixed. The
unlabeled loss remains an average over the entire hidden pool, now 245 rather
than 325; do not renormalize over accepted targets.

Keep pilot 013's preprocessing, 149-input/49,840-parameter LSTM, fresh weights,
560 updates, Adam 0.001, batch 32, ramp 140, native masking, partial-view targets,
0.95 dual-confidence gate, anchor confidence 0.5, minimum three correct anchors,
MAD floor 0.05 and radius-squared floor 0.0025. Do not change the confidence gate
after seeing admissions. One frozen pseudo-target round. Same mask/feature
limitations and active/inactive dropout-stream caveat as pilot 013 apply.

Five seeds times four teachers, eight 160-label students and one full-405
reference = 65 fresh fits. Students: supervised160, supervised160_long (1680
updates), full-teacher confidence, native-subset-teacher confidence, profile,
embedding support, and predicted-class count/weight-matched confidence/random.
The long supervised run matches optimizer-update count, not FLOPs or runtime.
Charge teacher and selector costs per method, even when shared physically.

Use the same 48 development clips for checkpoint selection (earliest minimum
cross-entropy, 28 checks). Thus training plus development accesses 208 labels,
not 160 total human labels. The same 96 previously inspected clips are a fixed
retrospective development comparison, never an untouched final test. No new
recordings or validation-to-training transfers. Upstream MediaPipe pretraining
and earlier development work are common prior resources, not cost-free learning.

Freeze code, data, configuration and sampled rows before fitting. Learner bundles
contain only revealed labels. All 60 low-label fits finish before full-405
reference label access. All 65 fits and targets freeze before the new run's
endpoint scoring and oracle pseudo-label diagnostics. Use a separate run and
ledger; never restart or overwrite the parent or experiment 012's evaluation.
Archive parent artifact hashes and verify they remain intact. Reproducing the
unchanged full-405 reference should yield identical weight-state hashes.

Primary comparison: profile minus native-subset confidence macro F1, paired
across all five seeds on the 96 clips; also report each sitting. Report all
controls, classes, NONE-to-movement, STOP recall, NLL and compute. Diagnostics:
teacher agreement, confidence admissions, profile admissions/weight mass,
per-class anchor support and post-freeze pseudo-label errors. Empty precision
is undefined; report pooled error counts as well as averages across supported
seeds. Matched coverage/weights comparisons do not guarantee matched outcomes.

Compare 80 versus 160 descriptively; higher accuracy after revealing more labels
is not evidence of algorithmic gain or saved labeling. Positive or negative
outcomes, zero admissions and rare-style rejection remain in the record. Do not
tune the gate, select a winning seed or launch additional budget/threshold runs
as an unreported rescue. No novelty, publication, broad generalization or robot
reliability claim. No camera activation, dependencies or new lesson files.
