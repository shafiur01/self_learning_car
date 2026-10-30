# Lesson 13 — inspect errors before choosing a fix

Our goal is a learning method that reduces human labeling work while recognizing
robot commands well. Before adding unlabeled training, we need to understand
which existing mistakes a model could confidently repeat as machine labels.
Today we built an error investigation and a viewer for your observations.

## What is a failing clip?

For this viewer, a failing clip is an existing validation recording whose human
label disagrees with at least one selected model. One recording may fail for one
seed and work for the other four; filter by family and seed to distinguish that.
The union over all 25 models contains 15 unique clips. It is not the error rate
of a single model, and 25 predictions of one clip are not 25 new observations.

There are six validation clips per class: F/B/L/R/STOP/NONE/UP/DOWN contain
2/3/0/1/2/5/2/0 unique failing clips respectively across all models.
Successful examples are useful controls, so the viewer can show all 48 clips.

## What the first investigation found

The two difficult STOP recordings have full hand visibility. Their normalized
coordinates are farther from the recorded STOP training examples than the four
successful STOP recordings. That tells us where to look: palm angle, finger
configuration, changing shape and how normalization represents the hand.
It does not prove their labels are wrong or that more data is the only fix.

A relaxed repositioning NONE clip is predicted LEFT by 18 of the 25 models;
one wrong score is .995. Several resting/down-facing NONE clips become DOWN.
Your intention matters to the correct label, but the model sees coordinates.
Identical visible observations with different private intentions cannot always
be separated without an observable gesture convention or context.

A hands-out-of-view NONE clip fails for one BiLSTM seed. Its replay is blank
because no hand was tracked. That is appropriate for this prompted recording.
Do not assume every blank clip is bad tracking or should be deleted.

Nearest examples help us inspect coverage, but their distances are not calibrated
probabilities or proof of label truth. Two clips can look close numerically and
have different meanings. Pose distance here excludes global wrist movement;
wrist-profile distance is shown separately. No-hand geometry is undefined, so
that clip has no pose-neighbour comparison.

## Open the viewer

From the project terminal, run:

```sh
open research/experiments/runs/failure_review/20261014T053532241062Z/review.html
```

If your browser restricts local-file storage/downloads, use the local server:

```sh
.venv/bin/python -m research.experiments.failure_review serve
```

Then open `http://127.0.0.1:8765/review.html`; Ctrl+C stops the server.
The self-contained viewer has no external requests or libraries to download.

## Review step by step

1. Start with the default STOP category. Watch the two clips slowly before
   revealing model guesses. Pause or move the frame slider when needed.
2. Switch between Camera position and Centered hand shape. The camera view
   preserves wrist movement. Centering helps compare fingers/palm geometry but
   intentionally hides global wrist movement in the display.
3. Compare the same-label training example. Record whether you held the palm
   differently, curled fingers, moved naturally or deliberately changed direction.
   We did not save camera RGB; the skeleton cannot show every relevant detail.
4. Choose an assessment, write an observation and click Save observation.
   A proposed label is only a suggestion; the original label stays unchanged.
5. Reveal model guesses when ready. Each family lists its five seeds and scores.
   Those scores are not independently verified probabilities of correctness.
6. Choose NO_COMMAND, then the other categories. Use All clips to compare correct
   examples. Categories with zero failures are still accessible in that mode.
7. Click Export observations. Keep `gesture_review_notes.json` and tell the
   assistant when you finish so it can inspect your observations separately.

Browser notes are not yet files in the research repository; exporting preserves
a portable copy. If storage is blocked, export before closing. Review time is
approximate foreground viewer time, including thought and pauses. Confirmations,
corrections and evaluation review all count toward future human-effort budgets.

## Why these files exist

- `experiments/failure_review.py`: verifies pinned data/models, calculates fixed
  descriptors and neighbours, builds the viewer and imports feedback separately.
- `experiments/review_assets/viewer.html` and `viewer.js`: filters, animation,
  model comparison and observations. They change no training files.
- `FAILURE_REVIEW_PROTOCOL.md`: documents what we measure and its limits.
- `findings/009_failure_review.md`: preserves the evidence and hypotheses.
- The dated failure-review run: original-source fingerprints, replay data,
  error CSV/analysis, browser checks and the viewer itself.

We have completed the automated investigation and viewer. Your human review is
the next part. A useful observation may motivate a controlled feature, coverage
or learning-policy experiment; it does not justify silently changing labels,
reusing validation as untouched testing or declaring a new algorithm successful.
