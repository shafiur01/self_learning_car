# Lesson 9 — clear labels come before better learning algorithms

The new plan is now prepared: one-finger up, two-finger up and thumbs-up all mean
UP, with remaining fingers folded. The recorder prompts all three shapes across
50 training clips. This teaches one class with several appearances; it does not
create three output commands. Your STOP-history report was "almost all open palm,
no fists"; we preserve that qualification when reusing the existing examples.

You clarified that STOP means an open palm. Our earlier written definition said
closed fist. That was an incorrect assumption in the project setup. We must fix
the new contract and document the discrepancy, rather than silently rewrite the
earlier experiment as though it used the corrected instructions.

Our classifier receives hand landmarks and human labels, not the English text
of the protocol. If recordings labeled STOP show open palms, it can learn open
palms as STOP despite the wrong text. But inconsistent recording instructions
can create a dataset containing different shapes under the same label. This is
why we ask whether all old STOP clips were open palm or whether they were mixed.

Movement is already part of the input: the 67 features include wrist x/y motion,
hand shape, size change and visibility over time. A model predicting STOP on an
upward movement does not prove that movement is unusable. Possible explanations
include training coverage, overlapping gestures, what the two-second clip caught,
and the representation or learned weights. We have not established the cause.

A distinct UP hand shape is a reasonable interface experiment. For example,
one index finger extended upward differs visibly from a fully open STOP palm.
Keep the selected shape consistent, but vary camera distance and modest angle so
the network does not learn only one pose at one location. Fifty clips across
two sittings is a pilot target; it is not a guarantee of accuracy.

We use a new data version because UP's meaning is changing and STOP's documented
meaning was wrong. The new training set excludes the old UP examples. Previous
recordings and checkpoints stay available to explain prior outcomes, while the
new schema prevents accidental use with an old model. If STOP examples are mixed,
they need review/replacement too. The model cannot infer which label semantics
we intended merely because we changed a line of documentation.

After collecting new training examples, we still need fresh validation under
the new meanings. Training changes weights; validation chooses and diagnoses
them; a later untouched test checks fixed choices. Old inspected tests cannot
become untouched again. No new data collection automatically updates weights.

Finally, distinguish interface design from ML method improvement. A narrower,
more visually distinct gesture may be easier to classify. Even a large accuracy
gain would not prove that a new learning algorithm solved the original broader
gesture task or used fewer labels. Our robotics/label-efficient research still
needs controlled comparisons under the same command definitions and human cost
budgets, then participant and robot evidence.
