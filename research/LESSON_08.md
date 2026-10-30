# Lesson 8 — an improved average can hide an important failure

We tested all 15 fixed models on 80 new clips after freezing their weights and
rules. That tests generalization to this user's new recordings, rather than
whether the network can memorize training examples. No training happened here.

Mean accuracy was 79% for the original models, 84.5% for routine additions and
83% for targeted additions. The same test clips went to every model. Routine
beats targeted on the primary macro F1 comparison in four of five seed pairs;
the targeted advantage we hoped for did not appear. Reporting that is research.

Why does 84.5% not settle the problem? Accuracy combines every class into one
number. Routine models recognize 86% of the NO_COMMAND clips, compared with 34%
for the original models. That large improvement can outweigh another weakness:
UP recall is only 26%, compared with an already poor 32% for the originals.
Here recall asks: **of the actual UP examples, how many were called UP?**

Most missed UP clips were called STOP. Look at the confusion matrix: its rows
are the human labels and columns the predictions. The UP row spilling into the
STOP column identifies this failure. STOP's precision is also low, because many
predictions of STOP are really UP. Precision asks: **of the clips called STOP,
how many actually were STOP?** These two viewpoints explain the same confusion.

Macro F1 averages a precision/recall balance across each of the eight classes.
It helps expose class weaknesses, but even macro F1 needs its class-level table:
no scalar can fully describe the command interface.

The shared errors are especially instructive. All models call six UP clips STOP,
covering prompts for whole-hand motion and angled pointing. We know the prompt
and human label; we have not independently verified the actual performed style.
Possible causes include missing training styles, how motion is represented, or
what was captured. We must inspect development data before choosing an explanation.
Agreement by 15 models is not proof that the human label is wrong.

This connects directly to semi-supervised learning. Pseudo-labeling uses a model
guess as a label for an unlabeled clip. If the model confidently calls UP STOP,
learning from that guess can strengthen the mistake. A later algorithm must
compare ways to audit such mistakes under equal human/compute budgets, rather
than assume high confidence means truth. This experiment identifies a failure
mode to investigate; it does not validate the proposed audited algorithm.

Adding 40 labeled examples per condition improves some results, but that is
**not evidence of needing fewer labels**. To establish label efficiency, compare
methods at the same label budget or show a target performance reached with fewer
labels, while counting collection, validation, testing and human review effort.
These 80 test labels are evaluation cost and never entered weight updates.

Five seeds share ten actual clips per class. They measure variation due to
training randomness; they do not give us 50 independent people or gestures per
class. A larger claim needs multiple people/sessions and another benchmark.
There are still raw false movement guesses on NO_COMMAND and STOP, and no robot
was connected, so these results cannot establish robot safety or task success.

Our next diagnostic should inspect UP/STOP training coverage and saved landmark
motion. Keep old models and labels. This test pool is now inspected; if it guides
future improvements, another honest final comparison needs new data after those
future choices freeze. Improving on an inspected test is development, not a new
untouched test result.
