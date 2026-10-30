# Lesson 15 — what makes a comparison convincing?

The new motion inputs helped every model family on our current validation set.
The surprising part is that a model which averages motion features also scored
very well. We therefore have evidence for useful **inputs**, but not yet for a
special advantage from an LSTM's memory. Learning that distinction is part of
doing research correctly.

## Representation and model are different choices

A representation is the information we give the network. Our representation
contains hand shape, wrist movement, estimated orientation, velocity, tracking
validity and timing. A model is the calculation that turns these inputs into
eight class scores.

Think of describing a journey. Positions are a series of locations; velocities
also tell you how fast and in which direction the traveler moved. An LSTM can
try to infer movement from positions. Explicit velocities make some information
easier to use, including for a simpler model. They do not create new recordings
or turn pretrained hand landmarks into calibrated real-world measurements.

We tested four representations with five model families. The complete table is
in [finding011](findings/011_cross_model_motion.md). Camera-coordinate motion
mean accuracy: CNN99.17%, LSTM97.92%, BiLSTM99.58%, local set98.75%, feature
set99.17%. These are development scores on the same48 clips, not final accuracy.

## A control tests an alternative explanation

Our hypothesis was that remembering a sequence helps distinguish a changing
FORWARD gesture from a maintained STOP palm. A recurrent model can process order,
but a high score alone does not show that memory caused its success.

The set baseline processes each feature row separately and averages the resulting
embeddings. It has no recurrent state or temporal convolution. Its strong result
means we should investigate whether explicit motion information already makes
this small task easy. We also removed its elapsed-time channel; its mean accuracy
remained98.75%, while the matching elapsed-masked LSTM scored97.92%.

There is an important boundary to this conclusion. A velocity already compares
an earlier observation with a later one. Averaging velocity-containing rows does
not erase all chronology. Initial-reference displacement and sample intervals
also remain. We have not shown that raw gesture order is unnecessary; we have
shown that these data do not establish a benefit from recurrent memory over
these particular controls.

The causal CNN is another control. Each convolutional position reads only the
present and past. Its final head still combines the whole observed clip. A model
being causal does not by itself show how quickly it can issue a correct command.
For that we need gesture onset labels, ambiguous prefixes, a WAIT/defer decision,
continuous non-command time, and measured false actions and STOP delay.

## Why hold things constant?

Each model uses the same538 training recordings and48 validation recordings,
the same560 optimizer updates and five random seeds. A seed controls random
starting weights and sampled training batches. Matching batch streams reduces
one source of accidental differences; matching initial weights across input
conditions within a family makes that comparison easier to interpret.

Different architectures cannot have identical weights, so we approximately match
their parameter counts around49,000. They still require different compute and
may prefer different training recipes. This is a fixed-budget comparison of
these implementations, not proof that one architecture is best after every
possible tuning method.

We keep allfive runs. Reporting only a100% seed would hide the others. Five runs
on six NONE clips still contain only six unique NONE situations, not30 independent
scenarios. Seed standard deviation measures training variability on these clips;
it does not measure uncertainty across people or recording conditions.

## Validation can become too familiar

Training data changes weights through gradients. Validation data chooses a
checkpoint and informs development decisions. A genuinely held-out evaluation
tests frozen choices on new examples. Once we repeatedly inspect an evaluation
set and redesign the system from it, it becomes development evidence.

We audited every canonical raw cohort. Ninety clips in the legacy validation
folder have already been copied into current training. They cannot independently
validate that model even though their old metadata says validation. Old UP clips
also mean a different gesture from today's held finger/thumb UP. Those examples
must not be silently pooled into an eight-class accuracy claim for the new task.

No malformed recordings or conflicting canonical labels were found. That means
their arrays and lineage passed our checks. It does not prove every human label
or intended gesture is correct. The80 old boundary-test files match their original
hash manifest. We could not locate an equivalent complete original byte-hash
manifest for the52 older test clips; we state that limitation.

We kept historical test scores instead of re-running them as another apparently
fresh benchmark. The new48-clip cohort has selected weights repeatedly, so its
high scores now need a new-session check.

## Recognition accuracy and robot behavior are different

BiLSTM had the highest mean accuracy,99.58%, but one seed classified a natural
repositioning NONE clip as LEFT. The CNN and feature set had slightly lower
accuracy but no observed NONE-to-motion errors here. A robot cares about the
consequences of errors, not just their total count.

This is why we report STOP recall and false movement as well as accuracy and
macroF1. MacroF1 averages each class's precision/recall balance so we can see
performance across commands. Even these metrics are isolated-clip measurements;
we have not tested robot actions or established reliable real-world operation.

## Why the new files exist

| File | Purpose |
| --- | --- |
| `experiments/cross_model_motion.py` | Fit the new model families/controls, preserve references and verify archived results |
| `experiments/dataset_validation.py` | Audit numerical validity, duplicates, training overlap and gesture-definition compatibility |
| `experiments/prospective_motion.py` | Freeze all candidates, collect new prompted clips without predictions, then evaluate jointly once |
| `CROSS_MODEL_MOTION_PROTOCOL.md` | Record comparison choices before seeing the new results |
| `PROSPECTIVE_MOTION_PLAN.md` | Record the next cohort's rules before collecting or scoring it |
| `experiments/runs/cross_model_motion/.../` | Preserve weights, sources, settings, histories, probabilities, tables, failures and costs |
| `CONTEXT.md`, `PROTOCOL.md`, `LOG.md` | Preserve current state, scientific decisions and an honest experiment history |

The model test files check mathematical/software properties with synthetic data.
Passing them does not show recognition accuracy on people. The research runner
does not silently change the active camera model.

## Your next step: first fresh sitting

All115 candidates are now frozen. We will collect two new physical sittings,
48 clips each, six/class. This checks whether the development gains survive a
new recording occasion for you. It does not yet test unseen people or systematically
controlled speed/view changes. Both sittings finish before model scores are shown.

From the project terminal, run:

```bash
.venv/bin/python -m research.experiments.prospective_motion record --sitting 1
```

1. Read the command and style shown on screen. Perform it for about two seconds.
2. Press SPACE to freeze the preceding two seconds and inspect the replay.
3. Save with the displayed label key if it shows the prompted gesture. Press x
   to repeat a mistaken or poorly tracked command. No predictions are used.
4. Continue until48/48. STOP is an open palm held still; use the displayed
   one-finger/two-finger/thumb variant for UP, with other fingers folded.
5. Press q. Take a break before the second physical sitting. Its command uses
   `--sitting 2`. If interrupted within the same physical sitting, the original
   command resumes its saved position.

The96 new labels and all recording/review time count as effort. We will evaluate
all frozen candidates together and report both sittings, including regressions.
After seeing those results the cohort becomes inspected development data.
The later label-efficient learning experiment and final paper evaluation need
their own declared budgets and genuinely separate data. We have not established
label savings, SOTA or a novel algorithm from today's supervised comparison.
