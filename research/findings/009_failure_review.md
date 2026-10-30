# Development error investigation and landmark review viewer

> Git-tracked records: [configuration, results and provenance](../results/009_failure_review/README.md). Original run/data links below require the local private archives.

2026-10-15 update: user reviewed clips; nine cumulative exports were found directly
in Documents and imported separately. Eight unique clips confirm original labels,
no label changes. See [review reconciliation](../notes/2026-10-15_review_reconciliation.md)
and [experiment010](010_motion_representation.md). The pending status below is the
original 2026-10-14 preparation state, not the latest feedback status.

Automated analysis/viewer completed 2026-10-14 (America/Chicago); human review
is pending. No new fit, label change, threshold tuning or final-test access.
Protocol: [FAILURE_REVIEW_PROTOCOL](../FAILURE_REVIEW_PROTOCOL.md).
Artifact: `experiments/runs/failure_review/20261014T053532241062Z/`.
Inputs are the unchanged experiment 008 training/validation cohort and its
25 saved models. Old recording bytes and all weights are preserved.

## Which categories contain failures?

There are 15 unique validation clips missed by at least one of the 25 models,
out of 48 clips. This union includes occasional single-seed mistakes; it is not
the error rate of a particular deployed model or a vote/ensemble evaluation.

| Original label | Unique failures / six clips | CNN | Pose set | Feature set | LSTM | BiLSTM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FORWARD | 2/6 | 2 | 2 | 2 | 2 | 2 |
| BACKWARD | 3/6 | 2 | 3 | 2 | 3 | 3 |
| LEFT | 0/6 | 0 | 0 | 0 | 0 | 0 |
| RIGHT | 1/6 | 0 | 0 | 0 | 0 | 1 |
| STOP | 2/6 | 2 | 2 | 2 | 2 | 2 |
| NO_COMMAND | 5/6 | 1 | 1 | 1 | 4 | 3 |
| UP | 2/6 | 1 | 1 | 1 | 0 | 2 |
| DOWN | 0/6 | 0 | 0 | 0 | 0 | 0 |

Family columns count a unique clip missed by at least one of that family's five
seeds, not all five. RIGHT's single failed prediction is BiLSTM seed 44.
Zero failures on six observed clips does not establish population reliability.
Full 1,200 guesses and all correct clips remain accessible for comparison.

## STOP: tracked, but different in normalized pose coordinates

Both shared failures have 100% tracking visibility, no feature values at the
clipping limit, approximately two-second durations and maximum frame gaps near
.072 seconds. Missing-hand frames do not account for these particular failures;
this does not independently establish that the tracker estimated every joint well.

| STOP file | Prompt | Wrong / 25 | Nearest STOP pose RMS | Shape variation RMS | Wrist path (palm units) |
| --- | --- | ---: | ---: | ---: | ---: |
| `31f155fc13db4984b9b5108026540f79.npz` | Different height | 23/25 | .600 | .069 | .449 |
| `906b8197e62b4f2d80809490b0cd8028.npz` | Modest palm angle | 24/25 | .733 | .078 | .793 |

The four STOP clips correctly recognized by all 25 models have nearest-STOP pose
distances .053/.086/.086/.096 and local-coordinate variation .011–.033.
The first failure's closest FORWARD distance is .619, versus STOP .600; the
second's FORWARD .729 is slightly closer than STOP .733. These are descriptive
distances, not confidence scores or a validated classifier.

A plausible investigation concerns different projected hand geometry, finger
articulation, palm orientation, tracker error or normalization sensitivity. It is
not proof of missing coverage, incorrect labels or the cause of neural predictions.
The first failure's wrist path falls inside the .139–.477 range of successful
STOP examples; a simple wrist-motion distinction is not supported as a complete
explanation. Human review should compare both camera-position and centered shapes.
Only one session/four controls were examined; no population inference follows.

## NO_COMMAND: resting/repositioning and a no-hand case

- `ee21e3af784b4c648f9a29f2c1528904.npz`: ordinary relaxed repositioning; LEFT
  in 18/25 guesses, including LSTM seed 33 score .99506. Fully tracked, with low
  shape variation .011 and wrist path .211. Nearest NONE pose .128 is closer
  than nearest LEFT .200. Pose-neighbour proximity alone does not explain or
  repair these confident neural errors. Another relaxed-height NONE example has
  no errors despite comparable nearest NONE/LEFT distances .150/.197.
- `7aa24e4618164e0ab75033ab4b8567ac.npz`: naturally lowered relaxed hand;
  DOWN in three LSTM seeds and BiLSTM seed 55. Wrist endpoint moves downward
  .271 initial-palm units during the recorded window. Measured movement does
  not establish a deliberate DOWN command; preserve the human NONE label.
- `22ed7684a75b4be69198bdc38165e7f4.npz`: relaxed fingers facing down; LSTM
  seeds 33/44 predict DOWN, despite small wrist endpoint displacement. Posture
  and intended command remain distinct concepts.
- `9398145ff331474eac368f2e0ec9a98e.npz`: natural visible rest; one LSTM seed
  predicts DOWN. These resting styles warrant inspection, not automatic relabeling.
- `d4bea05346814a7cbc5bf50e282ba514.npz`: intended hands-out-of-view NONE,
  no hand tracked in any frame; BiLSTM seed 11 predicts LEFT with score .3104.
  All other 24 models predict NONE. No hand geometry exists to compare, so the
  viewer shows no pose reference. This is a legitimate no-hand control, not
  automatically a failed recording. A raw classification error is not a robot
  action or an evaluation of the application's execution gate.

The analysis motivates human review of shape/intent/coverage and recurrence
behaviour. It does not determine a new algorithm, correction policy or causal fix.

## What the user can review

[Open the offline viewer](../experiments/runs/failure_review/20261014T053532241062Z/review.html).
It embeds all 48 validation landmark recordings and 53 selected training references.
Original timestamps, tracking flags and mirrored-preview orientation are retained.
The centered view uses existing aspect/palm normalization, without adding raw
frames through missing-hand intervals. References remain explicitly training.

Filter by original category/family/seed and failures/all/unreviewed. Slow, pause,
scrub or replay; compare the nearest same-label training example. Model guesses
start hidden. Reveal them to inspect all five family panels and a nearest example
of the most common incorrect class. The latter is selected using known labels
for review, not as a label-free learning feature.

Save an assessment/free-text observation and optionally propose a label. Notes
remain separate browser data; export `gesture_review_notes.json` to preserve them.
The importer versions observations under `human_reviews/` without changing raw
files, current labels or any training pool. Actual user observations/corrections
are still pending and must count as review effort in future budgets.

No RGB frames were saved; this cannot show palm surface texture, scene context,
the full arm or private intent. Landmark replay is an incomplete observation.

## Verification

Verified all experiment 008 weights, 586 pinned inputs and 1,200 saved vectors
before extraction. Re-extracted all 48 validation and 53 reference feature arrays
and required exact agreement with their archived rows; recording hashes checked
before/after. The other training rows are used only as existing verified arrays
for distance comparisons, not exposed as new validation data.

Five unit checks passed: pose/wrist distance isolation, descriptive no-hand
signals, foreign-role refusal before decoding, script-data escaping/offline
rendering, and versioned feedback with unchanged originals/duplicate refusal.
Fifteen real Chrome browser checks passed for filters, missing-hand/zero-failure
cases, mirrored mapping, reveal, scrub, frame timing, observation export data and
label preservation. Screenshot visually inspected; layout and skeleton visible.
Browser QA uses an isolated temporary profile and synthetic observations only.
The sandboxed Chrome launch aborted; the approved headless run succeeded, with
nonfatal macOS display-link warnings retained in its log. It did not open the
user's browser profile or submit human feedback.

Saved analysis JSON, failures CSV, replay JSON/HTML, source archive, fingerprints
and browser QA results. No training, model activation, dependence installation,
test-pool reuse, research novelty or annotation-saving claim.
