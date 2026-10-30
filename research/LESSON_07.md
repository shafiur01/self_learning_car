# Lesson 7 — why we freeze a model before testing it

Our goal is a learning procedure that uses fewer human labels while recognizing
robot commands accurately. So far, these are supervised landmark experiments:
we have not yet implemented the proposed semi-supervised learning algorithm.

Training adjusts the neural network's weights. Validation helps us choose a
checkpoint and understand errors. Our validation examples have already influenced
what we collected next, so they cannot give an untouched final judgment.

A fresh test asks a different question: **how do the fixed choices perform on
recordings they have never influenced?** Before those recordings exist, we fix
the candidates, preprocessing, meanings, collection schedule and calculation.
That is a freeze. It saves a dated plan and hashes; it does not train or improve
anything. A hash is a fingerprint that changes when a saved file changes.

In experiment 004, five seeds initialized five versions of each condition.
Reference kept the original training data. Routine and targeted each added 40
examples under matched training budgets. All 15 checkpoints remain candidates;
we will not choose whichever seed happens to look best on the test.

The new test has two sittings, five clips per eight classes in each: 80 total.
These 80 labels are evaluation effort, not additional training labels. Record
without seeing model predictions, so a persuasive wrong prediction cannot steer
your label. Replay the actual saved two-second gesture before labeling it.

Accuracy counts correct clips divided by all clips. Recall asks how many actual
examples of a class were found; precision asks how many predictions of that class
were correct. F1 balances precision and recall. Macro F1 averages the eight class
F1 scores equally. We use it as the primary comparison and also inspect every
class and NO_COMMAND/STOP errors; an average can conceal a dangerous confusion.

For each seed, subtract routine macro F1 from targeted macro F1, then average
the five differences. Positive favors targeted on this sample; negative favors
routine. Separate sitting results reveal whether the pooled result hides a
change. Five seeds use the same recordings: they show training variation, not
five independent people or five independent tests of the world.

`experiments/boundary_test.py` exists because the earlier single-condition test
tool cannot jointly evaluate this three-condition study. It keeps the fresh data
separate, guides prompts, verifies saved files and scores every fixed candidate
together after both sittings. Its synthetic test file checks those safeguards
with fake landmarks; fake-landmark accuracy says nothing about your hand.

When evaluation first reads the landmark payloads it writes an access ledger.
Even a failed evaluation means the pool is no longer guaranteed untouched. This
avoids accidentally repeating or tuning a test until the result looks better.
After the results, we can learn from errors, but another final claim then needs
new data after the next set of choices freezes. There is no unlimited final test.

A good result would support fresh-session recognition for this user's specified
style sample. It would not establish fewer training labels, a novel algorithm,
unseen-user recognition or robot safety. Those need the next cost-matched SSL,
participant/benchmark and robot experiments. A weak result is also useful: it
shows what our current representation or training data fail to handle.
