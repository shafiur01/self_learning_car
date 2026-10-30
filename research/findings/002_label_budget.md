# Finding 002 — supervised label-budget pilot

> Git-tracked records: [configuration, results and provenance](../results/002_label_budget/README.md). Original run/data links below require the local private archives.

Recorded 2026-10-07 (America/Chicago). Study ID, in UTC:
`20261008T005037258990Z`. All 20 trials completed. These are development
validation results, not untouched test results, observed annotation savings,
robot trials, or evidence of novelty.

Local artifacts: [report](../experiments/runs/20261008T005037258990Z/report.md),
[summary](../experiments/runs/20261008T005037258990Z/summary.json),
[exact selections](../experiments/runs/20261008T005037258990Z/selections.json),
[PNG plot](../experiments/runs/20261008T005037258990Z/label_budget_curve.png), and
[PDF plot](../experiments/runs/20261008T005037258990Z/label_budget_curve.pdf).
Those artifacts are Git-ignored; this durable finding is stored separately.

## Design fixed before execution

Frozen development inventory: 334 training clips, of which 201 are original
recordings and 133 are model-selected corrections. Original recordings come
from one training session, with counts FORWARD 61, BACKWARD 38, LEFT 31, RIGHT
26, STOP 29, NO_COMMAND 16. No exact raw duplicates in the original pool.
Validation: the existing 72 clips from one separate single-user session.

Primary training uses only originals. Five/ten/fifteen labels per class give
30/60/90 labeled training recordings, plus all 201 originals as the reference.
The proposed 20/class condition was unavailable because only 16 original
NO_COMMAND recordings exist. Corrections were retained and excluded; their
earlier selection and acquisition costs are not treated as free labels.

Five paired seeds: 11, 22, 33, 44, 55. Each seed has nested balanced subsets;
the full reference retains its natural class imbalance. Same initial model
weights across budgets within a seed. Seeds jointly vary subsets and weights.
Session-aware sampling does not provide cross-session training diversity here,
because the eligible pool has only one session.

Existing landmark features and 48,390-parameter GestureNet. Adam lr 0.001,
class-weighted cross entropy, CPU/two threads, no augmentation, 560 updates,
batch size 32 sampled uniformly with replacement. Every trial has 17,920 draws
and validation checks at updates 20, 40, ..., 560. Select lowest unweighted
validation loss. These repeated draws are not new independent recordings.

## Results

| Training condition | Distinct labels | Accuracy mean ± run SD | Macro F1 mean ± run SD | NLL mean | Raw NO_COMMAND → motion mean | STOP recall mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5/class | 30 | 78.06% ± 3.73 pp | 0.7620 ± 0.0423 | 0.6779 | 29.09% | 68% |
| 10/class | 60 | 85.28% ± 3.34 pp | 0.8442 ± 0.0376 | 0.4989 | 18.18% | 84% |
| 15/class | 90 | 90.56% ± 2.67 pp | 0.9012 ± 0.0278 | 0.3636 | 10.91% | 96% |
| All originals | 201 | 93.33% ± 1.16 pp | 0.9297 ± 0.0087 | 0.2551 | 10.91% | 100% |

Training wall time mean: 3.94 / 4.02 / 3.97 / 3.96 seconds respectively.
Fixed updates make the measured optimization costs similar in this small CPU
pilot; this does not account for historical annotation or detector pretraining.

| Condition | Seed 11 accuracy | Seed 22 | Seed 33 | Seed 44 | Seed 55 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5/class | 76.39% | 80.56% | 83.33% | 75.00% | 75.00% |
| 10/class | 80.56% | 83.33% | 87.50% | 88.89% | 86.11% |
| 15/class | 88.89% | 87.50% | 91.67% | 90.28% | 94.44% |
| All originals | 93.06% | 93.06% | 91.67% | 94.44% | 94.44% |

Mean paired accuracy differences from all-original: -15.28, -8.06, -2.78
percentage points for 5/10/15 per class. Macro F1 differences:
-0.1678 / -0.0855 / -0.0285. No statistical significance/noninferiority claim.

| Condition | NO_COMMAND → motion counts by seed | STOP → motion counts by seed |
| --- | --- | --- |
| 5/class | 5, 3, 1, 4, 3 | 0, 1, 4, 3, 1 |
| 10/class | 2, 3, 2, 1, 2 | 2, 3, 2, 0, 0 |
| 15/class | 1, 1, 1, 2, 1 | 1, 1, 0, 0, 0 |
| All originals | 1, 2, 0, 1, 2 | 0, 0, 0, 0, 0 |

These reuse the same 11 NO_COMMAND and ten STOP clips across models. They are
not 55/50 independent observations, gated command events, or physical stopping
tests. A STOP misclassification can be another non-motion class; the explicit
STOP-to-motion count is distinct from all missed STOP classifications.

## Interpretation and unfavorable evidence

Five labels per gesture are inadequate for strong supervised recognition in
this particular comparison. Increasing budgets improved mean accuracy/F1/NLL
and STOP recognition here. Fifteen/class came closer to the full reference but
remained 2.78 accuracy points lower on average. Some individual seeds tied;
those do not establish that the smaller budget is equivalent. NO_COMMAND-to-
motion mean did not improve from 90 to 201 labels on these eleven clips.

This provides a measurable gap for later semi-supervised methods. Retain
30/60/90-label supervised conditions and the all-original reference; later
methods must count their added audit/correction/calibration labels and compute.
Do not select only the best seed or call simulated withholding actual savings.

Exploratory inspection of frozen validation predictions found one STOP/
NO_COMMAND-to-motion prediction with score >=0.95 in the 5/class condition,
and none in the other conditions. That single case was STOP → FORWARD, score
0.950678, seed 33, file `gesture_commands/data/02087edfe4684ce1a66935cde8afa49a.npz`.
These counts sum predictions across models, not independent clips or an audit.
They motivate checking confident errors but do not establish their prevalence.

The highest score among the full-reference NO_COMMAND-to-motion errors was
0.593003, below the application's 0.8 score requirement. Thus its 10.91% raw
rate must not be presented as an executed false-motion rate. We did not replay
continuous streams, temporal repetition, or the command gate. An execution
layer may change errors and useful coverage; that comparison remains pending.

The reference recipe differs from experiment 001 in source eligibility,
replacement batches, fixed updates, and validation timing. Comparing 93.33%
here to 96.67% there is not a controlled finding about improvement or decline.
Validation selects checkpoints and compares budgets; single-user/session data
cannot establish unseen-user accuracy or a robot-safety benefit.

## Next concrete step

Collect genuinely new development validation sessions before judging these
frozen candidates further: first ten clips of each of the six labels (60 clips)
in one sitting; repeat in two separate sittings for 180 new validation clips.
Keep training and final evaluation separated. These remain development data;
the eventual unseen-user study needs participant metadata and separate people.

Then compare the frozen models by session and define a matched-budget
semi-supervised reference. Robot tasks/controller and consequence auditing need
their own subsequent protocol; this finding does not validate that mechanism.

## Reproduction identity

- Configuration SHA-256: `1fecff6e62a1945cfc6c5f1df665cd2479e7783908d04c7b8f3230b3d3d76977`
- Manifest SHA-256: `6983dd295504092ac898fa168adaeca4f7fe5a3e33349ebb09903b98df3959eb`
- Selection SHA-256: `981d7b1c232f1530f376dcd023927c8026374266216d70bf04229db74395f191`
- Frozen features SHA-256: `52e6ff4d5ab4273465c8dd9e9c742d1a793ab534b994cd4f157123c4c9ef8efe`
- Budget runner SHA-256: `01c035523d2a9f0b9f2ed2d8fa0b591f5f6636cbde059be9123b3d1bbddbb046`
- Pre-execution protocol SHA-256: `8e913ea8171e75a53f443ecbbdcc124decbe90e445e7e381440b40fb7785e839`
- Python 3.14.7, Torch 2.14.0, NumPy 2.5.3, Matplotlib 3.11.2, macOS arm64.

Git HEAD: `1aa67d04df9506d10c796568f4d450456d62ad54`. Source was uncommitted;
use archived source/hashes as well as configuration, snapshot, and selections.
All checkpoints and old recordings preserved; no active app model changed.
Sixteen research-integrity checks passed before training. The standalone plot
was visually inspected. No dependencies, commit, or publication added.
