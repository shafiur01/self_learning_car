# Development failure investigation and human landmark review

Declared 2026-10-14 (America/Chicago), before the new coverage analysis.
User requests the next step and a way to inspect failing clips in each category.
This is descriptive diagnosis and human review of existing development data.
No new fitting, threshold selection, test access or automatic label changes.

## Fixed inputs and questions

Use only the complete experiment 008 study
`experiments/runs/temporal_comparison/20261014T011524632211Z`: the same 538
training clips, 48 inspected validation clips and 25 saved model predictions.
Verify the study first, re-extract features from pinned recordings and require
exact agreement with archived arrays. Preserve recordings/weights/source plans.

For each validation clip, record all per-family/per-seed predictions, failures,
human prompt, timestamps and tracking. A category's failing-clip count means
unique clips missed by at least one of the 25 models, not repeated guesses.
Also report each family's unique failure counts and all-seed mistakes.

Describe feature signals without turning them into intended-command labels:
tracking fraction/gaps; wrist endpoint and path length in initial palm-length
units; local-coordinate RMS variation; log apparent-size change/range; feature
clipping. Wrist motion cannot distinguish intention by itself. Apparent scale
is not calibrated physical depth, and tracker coordinates are not measured metres.

For STOP/NONE compare failures with correctly recognized clips in the same
session, and inspect model disagreement. Nearest-training distances are:

- pose profile RMS over all 32×63 local coordinate values, with chronological
  alignment and without wrist/size/visibility channels;
- separate wrist-profile RMS over 32×2 initial-reference displacement values.

Do not combine them into an optimized metric. Nearest neighbours are examples
for inspection, not proof of missing coverage, wrong labels or an effective
classifier. Selecting neighbours by known human label is a review aid and must
never become hidden supervision in a later unlabeled learner.

For clips with tracked hands, replay includes the nearest same-label training clip and the nearest clip from
the most frequent incorrect predicted class, selected by pose RMS with pinned
row-order tie-breaking. Preserve every distance and show match/role/prompt.
The viewer can show all 48 validation clips, including successful controls.
Replay both camera-position coordinates and centered normalized poses; preserve
actual recording times, visibility and the original mirrored-preview convention.
No invented camera video, smoothing of tracking gaps or inferred private intent.
No-hand clips have no pose-neighbour comparison, because their hand shape is
undefined; zero-filled features must not be displayed as a real hand pose.

## Human review and cost

Use an offline local HTML viewer with no external requests/dependencies. Filter
by original category, model family and seed; show either failures or all clips.
Model guesses start hidden so the user can first observe the replay. Scores are
model scores, not verified probabilities of correctness.

Capture observations in a separate export: agrees with intended gesture,
ambiguous, tracking problem, possible label mistake, free-text explanation and
optional proposed label. The original label is always displayed/preserved.
Feedback is not automatically applied to data/training. Record review timestamps
and foreground review intervals as approximate effort, not pure annotation time.
Keep exported reviews and subsequent decisions versioned. New user-provided
labels/confirmations/corrections count toward future labeling budgets.

The artifact stores skeletons, not RGB; it cannot show palm skin texture, scene
context, full arm movement or intention. Include this limitation in the viewer
and teaching notes. Flag suspicious tracking/labels as hypotheses until review.
All clips are already inspected development material, not a new final benchmark.
