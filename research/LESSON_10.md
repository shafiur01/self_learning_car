# Lesson 10 — from recordings to learned weights

We have now trained five models using538 training clips and48 validation clips.
The external pretrained hand tracker turns your video into21 hand landmarks;
our classifier learns commands from those landmarks. RGB video is not the
training input here. The32 time samples each have67 features containing hand
shape, wrist movement, size change and visibility.

The network has48,520 adjustable numbers called parameters or weights. Each
training update predicts labels for a batch, calculates cross-entropy loss and
uses gradients plus Adam to adjust these numbers. Labels do not become weights
directly: they indicate which output the model should favor. Class weights give
less frequent training classes more influence; they do not invent missing styles.

Each seed starts with different random weights. We used560 updates per seed,
32 clips per update, sampled with replacement. That is17,920 presentations,
not17,920 distinct recordings. A clip may be presented many times. None of the
48 validation examples enter those gradient calculations.

Every20 updates we measure unweighted validation loss with the model in
evaluation mode. This asks how well the current weights predict separate data.
We copy the weights at the lowest validation loss, then save that checkpoint.
The final update need not be the best saved model. The plot stars show selected
updates420/480/300/360/500 for seeds11/22/33/44/55. We keep every seed instead of
reporting only the best-looking run.

The files have different jobs:

- `commands_v3_train.py`: coordinates the run and loads only models with our
  corrected open-palm STOP and three-shape UP contract. It reuses existing ML code.
- `manifest.json`: records which clips train and which validate, with fingerprints.
- `development_features.npz`: saves the exact preprocessed numbers for reproduction.
- `config.json` and archived source/environment: record the budget and implementation.
- Each `model.pt`: learned weights plus meanings/settings/data provenance; it is
  not a file containing the original training videos.
- `history.json`: validation loss over training, explaining which state was saved.
- Predictions/metrics/summary: preserve every guess, mistake and class result.

Reloading saved weights reproduced all240 validation score vectors and checked
586 input fingerprints. This verifies our artifacts, not the real-world reliability
of the model. Training code never silently replaces the camera application's model.

Mean validation accuracy is85.83%, macro F1 .8611. New UP recall is90% on this
sample: index/two-finger examples are all recognized across seeds, while one
angled thumbs-up example is LEFT in three models. Because both gesture meanings
and validation recordings changed, this is not a controlled accuracy-improvement
comparison with v2's broad UP movement task.

STOP recall is only66.67%: every model recognizes four of six STOP clips and
misses the same two. Most wrong guesses are FORWARD, one with score99.4%.
Confidence is not proof of correctness. Repeated errors across seeds suggest a
shared problem to investigate, rather than something choosing another seed
automatically solves. They do not prove the human labels are wrong.

This also explains why we examine each class. Six actual examples/class viewed
by five models still gives six independent recorded clips per class, not30.
Mean accuracy can hide STOP errors. Precision and recall expose who receives
wrong commands and which intended commands are missed.

Finally, the48 clips helped choose the checkpoints. Their score therefore has
selection bias and is development validation, not an untouched final test.
We should inspect the shared errors using existing data, preserve every outcome,
then freeze choices before new testing. A changed interface and supervised
training result do not establish the proposed SSL algorithm, label savings,
novelty or robot performance. Those remain separate controlled experiments.
