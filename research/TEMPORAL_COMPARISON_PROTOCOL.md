# Experiment 008 — does recurrent temporal modelling help our v3 commands?

Declared 2026-10-13, before any real comparison fitting. This is a finite
single-user development experiment, not the final algorithm or publication test.
User authorized the next research task. Preserve all recordings and checkpoints.

## Question and five fixed conditions

Our fixed-weight probe did not establish strong reliance on frame order. Now
train different inductive biases using the same gesture definitions and data.

| Condition | Inputs and operation | Parameters |
| --- | --- | ---: |
| cnn | Existing 67-feature, two-layer temporal CNN | 48,520 |
| pose_set | Local 63 coordinates + visibility; frame MLP, mean over time | 48,792 |
| feature_set | All 67 features; same frame MLP, mean over time | 49,176 |
| lstm | All 67 features; one LSTM with 62 hidden units, four ordered pooled bins | 48,944 |
| bilstm | All 67 features; 36 hidden units per direction, four ordered pooled bins | 49,256 |

Set models use frame MLP input→128→176, then mean pooling, then 176→96→8.
ReLU follows hidden linear layers; dropout .2 precedes the output layer.
They are permutation invariant at inference, up to floating-point summation.
pose_set can observe the collection of changing hand shapes, but not their order
or global wrist/size channels. It is not a single-frame classifier. feature_set
also receives wrist displacement relative to the first frame, so preprocessing
can encode some trajectory information even though frame order is unavailable.
Neither baseline proves motion entirely absent. Recurrent heads pool to four
bins, then flatten→64→8 with ReLU/dropout .2 like the existing head.

Primary comparisons: lstm minus cnn and bilstm minus cnn in mean validation
macro F1 across all five seeds. Report paired seed differences without choosing
a best seed. Secondary comparisons with the set models help interpret order and
trajectory, but architecture changes prevent a pure causal attribution to order.
No architecture-specific tuning, extra augmented inputs or failed-seed replacement.
The common optimizer may favor some families; results compare this fixed recipe,
not each family's fully tuned potential. DD-Net/graph/explicit derivatives remain
a separate representation study, so this first comparison isolates a smaller
question rather than bundling an architecture and a new feature extractor.

## Data and training fixed before fitting

Source study: `experiments/runs/commands_v3/20261012T205727638067Z`.
Re-audit the v3 cohort and reproduce its existing weights/predictions before
allocating a new study. The exact original row order and extracted arrays must
match: 538 training clips, 48 development validation clips, one legitimate
no-hand NO_COMMAND validation clip retained. Whole recording sessions remain
in their original roles. No historical test coordinates/predictions accessed.

Seeds 11/22/33/44/55; 560 Adam updates at .001; replacement batches of 32;
class-weighted training cross entropy; no augmentation; two CPU threads.
Independent per-seed sampling generator seed+1,000,000 gives the same sampled
indices for every condition. Distinct architectures cannot share identical
initial weights/dropout masks; disclose this rather than claim all randomness
is paired. Validate every 20 updates (28 checks), choose the earliest strict
minimum of unweighted validation loss, and preserve every model/history.

Refit the CNN as well, and require it to reproduce the original initialization,
batch-index stream, saved update and validation vectors before interpreting the
new comparison. This is five extra reproduction fits, not five new independent
datasets. All 25 planned fits run sequentially in fresh worker processes.

## Outcomes, resources and integrity

Report all five-seed means and seed standard deviations: accuracy, macro F1,
NLL, class recall/precision, F/B confusion, STOP recall/STOP→motion and
NO_COMMAND→motion/DOWN. Six validation examples per class limit precision.
Seeds share the same clips; seed SD is not population uncertainty. This validation
already selected checkpoints and informed this comparison; no untouched-test or
statistical significance claim. Freeze future choices before new sessions/users.

Equal updates and similar parameter counts do not equal equal compute. Record
fit-loop seconds, full fresh-worker wall time, parameter counts, process peak RSS
and CPU forward-pass latency on one already-preprocessed 32-frame window.
Peak RSS includes Python/Torch/data/optimizer and is not isolated tensor memory.
Latency uses 40 warmup passes then 200 timed passes, median and p90, two CPU
threads, no camera/landmark extraction and no two-second acquisition delay.
Hardware/background load and fixed condition order can confound timing; these
are local engineering measurements, not universal deployment benchmarks.

Archive exact sources, protocol, environment, config, input manifests/features,
all weights/predictions/loss curves/metrics/costs and failed attempts. Reload all
25 saved weights and reproduce all 1,200 score vectors, summaries, source/input
hashes, batch-stream pairing and minimum-loss selection. Verify old input/model
hashes afterwards. Do not activate a checkpoint in the camera application.

## Meaning for the main research

An LSTM win would motivate further temporal evaluation, not establish a new
algorithm. A set-model win would motivate studying missing dynamic coverage,
redundant cues and recipe effects, not prove gesture recognition never needs time.
Our long-term question remains fewer human labels/review minutes with lower
consequential command errors under session/person shift. Representation selection
and audited semi-supervised learning must be evaluated separately.

Implementation references: the local PyTorch 2.14 LSTM API was checked against
[official documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html).
Framewise encoding followed by symmetric pooling is an established construction;
[Deep Sets](https://arxiv.org/html/1703.06114v3) is a primary reference. Neither
baseline is claimed as a new algorithm or a full reproduction of that paper.
