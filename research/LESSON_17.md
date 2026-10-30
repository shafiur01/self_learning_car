# Lesson17 — teaching from evidence, not confidence alone

You have set a clear constraint: use the existing recordings and request no more
personal validation/test gestures. We can continue meaningful algorithm research
within that constraint. We will keep describing the existing evaluation sets
as historically inspected, rather than pretending they are new tests again.

The candidate I recommend is **Evidence-Profile Self-Training**, a working name.
It is a proposed algorithm to test, not an invented breakthrough we can already
claim. The full design is [here](ideas/013_evidence_profile/README.md).

## The problem we are trying to fix

Suppose a model predicts FORWARD with99% confidence and we use that prediction
as a new training label. If it is wrong, the student learns the same mistake.
The next model can become more certain without becoming more correct.

Our project has already shown confidently wrong commands. It has also shown
that motion inputs help in some situations and hurt in others. Simply making
the network bigger, adding stages or trusting confident predictions does not
explain or resolve that evidence.

## Ask a second question

The usual question is “How confident are you?” The proposed additional question
is “When information is hidden, does your support for this command change in a
way that resembles known examples of the same command?”

Use three types of information already in our landmarks:

- **Pose:** finger arrangement and the hand's camera-relative orientation.
- **Local change:** how the normalized hand coordinates change over time.
- **Transport:** overall wrist displacement and apparent-size change.

Keep timing/tracking validity separate. Hiding an evidence block means it is
unobserved; it does not mean the real hand suddenly stopped or changed shape.
We train teachers with explicit missing-information indicators so these probes
are familiar to the network.

For each candidate label, record three signed score changes: hide pose, hide
local change, hide transport. This little vector is its **response profile**.
Compare it with profiles of known-label reference clips excluded from that
teacher's training.

For a held UP, pose might provide most support; little incremental motion support
could be appropriate. For a beckoning gesture, local changes may be important,
and a shape-only view may remain uncertain or look like STOP. We learn reference
patterns rather than manually declaring that every member of a class must work
the same way. Several reference examples can represent several observed styles.

The method does **not** require all partial views to predict the same command.
An incomplete observation can legitimately be ambiguous.

## How training would proceed

1. Select80 existing human labels,10/command, with session support on both sides.
2. Hide the other325 labels in the primary405-recording pool. No data is deleted.
3. Train two small teachers on separate session groups, from scratch. A teacher
   uses the other group's labeled clips as its reference bank.
4. A proposed label needs confidence plus a compatible response profile before
   teaching the student. Unsupported labels stay unused; they do not become NONE.
5. Train one student on the80 real labels and the admitted soft targets.
6. Compare against supervised and ordinary self-training controls using the
   existing development/evaluation recordings, with unchanged roles.

Only the student is needed for eventual inference. Extra offline teacher work
must still be counted, so we cannot yet say training is more efficient.

The405 original recordings exclude133 model-selected corrections from the primary
comparison. Those corrections remain preserved for a separate practical analysis.
Choosing only80 labels simulates a labeling budget; all405 inputs are still used.
It does not prove that80 total recordings are enough, or that we actually saved
the effort already spent collecting/labeling them. Development48 and endpoint96
labels are additional study costs, not free hidden resources.
The draft also separates a later total-input comparison of80/160/405 clips at
the same80 revealed labels. At80 total clips there are no extra unlabeled inputs,
so ordinary self-training has no new targets to supply.

We cannot start from our existing538-label weights for this claim: those weights
already learned from the supposedly hidden labels. Fresh teachers/students are
required; the common pretrained hand detector remains disclosed prior knowledge.

## What could be new, and what is already known?

Missing-modality gesture training, confidence-based self-training, expert fusion
and rationale-based self-training already exist. We checked primary literature,
including [ModDrop](https://arxiv.org/abs/1501.00102),
[FixMatch](https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html)
and [Few-shot Rationalization](https://aclanthology.org/2021.emnlp-main.836/).

Our narrower proposal is to compare **same-teacher responses with disjoint,
class-specific reference neighborhoods to decide which labels may teach the
student**. Whether that adds anything useful beyond confidence or ordinary
embedding similarity is the experiment. A working name or combination of
published ideas is not proof of originality.

## A mathematical condition, not magic

If the rule gives less weight to wrong labels than correct ones, the weighted
pseudo-label set becomes cleaner. The exact finite-pool identity is

`weighted error − original error = covariance(weight, error) / mean(weight)`.

Negative covariance means wrong labels tend to get smaller weights. We have
not proved that our profile rule achieves that on real recordings. It is a
precise property to measure, followed by the harder question of whether the
student actually improves. Cleaner labels from rejecting almost every useful
example would not be enough.

Eight synthetic mathematical checks passed. They include a deliberate failure:
a confidently wrong prediction with the same profile as trusted examples is
still accepted. The checks demonstrate logic and limitations; they do not
measure gesture accuracy. No real model was trained in this design step.

Our next step is to implement a small controlled pilot with fixed data access,
baselines, seeds and compute accounting. We do not need new recordings to run
that pilot. We will keep the result whether it helps or fails.
