# Literature ledger

## 2026-10-12 — expanded temporal and label-efficient review

New primary-source maps with explicit reading depth/access failures are in
[temporal architectures](notes/2026-10-12_temporal_prior_art.md) and
[learning/audit methods](notes/2026-10-12_label_efficiency_review.md).
The code audit is in [mechanism/protocol](notes/2026-10-12_mechanism_protocol.md).
This is a bounded review, not an exhaustive novelty determination or reproduction.

Important additions include [DD-Net](https://arxiv.org/pdf/1907.09658),
[DG-STA](https://arxiv.org/pdf/1907.08871),
[MMANet2025](https://link.springer.com/article/10.1007/s40747-025-01807-x),
[AS3L](https://arxiv.org/html/2203.04560),
[prediction-powered risk control](https://arxiv.org/html/2412.11174) and
[SemiVLA2026](https://arxiv.org/html/2606.21493v1). Do not treat combining
pose/motion, active queries, SSL or robotics-specific gating as automatically new.
Unavailable methods remain unknown, not omissions supporting originality.

Root additionally inspected primary TCN/C3D paper text:
[Bai2018](https://arxiv.org/html/1803.01271v2),
[C3D](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf).
Current CNN is not that specific causal/dilated TCN; xyz input does not imply
Conv3d. Skeleton-volume Conv3d is also published, so do not say all3D-CNNs require RGB.
The staged proposal and new fixed-weight diagnostic are in
[TEMPORAL_ALGORITHM_PLAN](TEMPORAL_ALGORITHM_PLAN.md) and
[finding007](findings/007_motion_sensitivity.md). No new trained architecture,
label-savings result, final test or novel algorithm is claimed from this review.

Initial focused review: 2026-10-06. Ledger created: 2026-10-07.
This is a starting reference map, not an exhaustive novelty search.

| Work | Existing idea / evidence | Proposed use or limitation |
| --- | --- | --- |
| [CReLU, Shang et al.](https://arxiv.org/pdf/1603.05201) | Separates positive and negative convolution responses; tested on image benchmarks. | Separate architecture ablation. Channel expansion needs a fair model-size comparison. Does not establish a label-efficient workflow. |
| [FixMatch, Sohn et al.](https://proceedings.neurips.cc/paper_files/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f733-Paper.pdf) | Weak-view confidence-filtered hard targets and strong-view consistency, alongside supervised loss. | Candidate semi-supervised baseline. Image flips/strong augmentations cannot be copied blindly to directional gestures. |
| [Activation survey, Dubey, Singh, and Chaudhuri](https://arxiv.org/pdf/2109.14545) | Survey and benchmark of activations across architectures/tasks; costs and outcomes vary. | Small controlled activation comparisons; no universal winner implied. |
| [Mean Teacher, Tarvainen and Valpola](https://arxiv.org/pdf/1703.01780) | EMA weight teacher supplies consistency targets. | Optional teacher stabilization. A moving average can still be confidently wrong. |
| [MixMatch, Berthelot et al.](https://arxiv.org/pdf/1905.02249) | Average/sharpen augmented predictions and MixUp labeled/unlabeled examples. | Related baseline; synthetic mixing of opposing motion commands needs scrutiny. |
| [UDA, Xie et al.](https://arxiv.org/pdf/1904.12848) | Useful augmentation for unlabeled consistency training. | Motivation to test task-specific transformations before semi-supervision. |
| [Realistic SSL evaluation, Oliver et al.](https://arxiv.org/pdf/1804.09170) | Strong baselines, tuning fairness, validation size, and distribution mismatch affect conclusions. | Design comparisons with explicit label/compute budgets and mismatched unlabeled motions. |
| [SimCLR, Chen et al.](https://arxiv.org/pdf/2002.05709) | Contrastive visual representation learning without command labels. | Possible representation baseline; large pretraining can be costly. |
| [SimCLRv2, Chen et al.](https://arxiv.org/pdf/2006.10029) | Self-supervised pretraining, labeled fine-tuning, unlabeled distillation. | Evidence for a staged workflow on ImageNet, not a universal cheapest/best method. |
| [MAE, He et al.](https://arxiv.org/pdf/2111.06377) | Reconstruct masked image patches for representation pretraining. | Temporal landmark masking is our proposed adaptation, not a proven MAE result for gestures. |
| [TS2Vec, Yue et al.](https://arxiv.org/pdf/2106.10466) | Self-supervised representations for multivariate time series; invariance assumptions can fail. | Related temporal baseline; not established as a drop-in solution for this dataset. |
| [Calibration, Guo et al.](https://arxiv.org/pdf/1706.04599) | Temperature scaling improves confidence calibration in evaluated settings. | Measure calibrated confidence on held-out data; scaling does not change argmax classification. |
| [BADGE, Ash et al.](https://arxiv.org/pdf/1906.03671) | Active batches combine predictive uncertainty and diversity. | Candidate label-selection comparator; our confident-prediction audits require their own evaluation. |
| [GELU, Hendrycks and Gimpel](https://arxiv.org/pdf/1606.08415) | Smooth activation tested against ReLU/ELU. | Optional small activation experiment, distinct from augmentation experiment 001. |

## Claims still unsupported

- Any of these stages guarantees substantially better gesture accuracy here.
- More unlabeled recordings always help, even under new/out-of-class motions.
- High softmax confidence makes pseudo-labels correct.
- Our combination has never appeared in a paper.
- A landmark classifier result demonstrates the best general image-training flow.
- Fewer labels automatically means less data, lower total training cost, or reliable
  continuous car commands.

## Next targeted review

Find nearest methods and benchmarks for semi-supervised skeleton-based dynamic
gesture recognition, self-supervised hand motion, active annotation of short
gesture sequences, open-set/non-command rejection, and continuous false-command
evaluation. Compare each closest work's representation, label budget, people and
session split, human-review mechanism, compute accounting, and metrics.

## 2026-10-07 — initial task-specific related-work screen

Primary abstracts were inspected for the following works; full methods and
benchmark protocols still require reading before choosing reproduction baselines.

| Work | Relevant existing territory | What remains to compare |
| --- | --- | --- |
| [Skeleton-Contrastive 3D Action Representation Learning, Thoker et al., 2021](https://arxiv.org/abs/2108.03656) | Self-supervised skeleton representations, spatial/temporal augmentations, and semi-supervised downstream evaluation. | Body-action versus hand-command setting; training costs and label budgets. |
| [Hi-TRS, Chen et al., ECCV 2022](https://arxiv.org/abs/2207.09644) | Hierarchical self-supervised skeleton pretraining, with supervised/semi-supervised evaluation. | Model size, temporal objectives, and whether its protocol covers our intended operating point. |
| [SKETCH, Han et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ccf308ecd36579d82532851419c283cd-Abstract-Conference.html) | Hand skeleton sequences mapped to visual representations and used with pretrained vision backbones; evaluates SHREC'19/'22. | Relevance of motion magnitude preservation and cross-condition hand-gesture evaluation; label-efficiency claims need full-paper inspection. |

These establish related published territory. They do not establish that our
specific proposed procedure already exists, or that it is new. A general claim
of being first to combine self-supervised skeleton features with limited labeled
classification would need to account for this existing work.

## 2026-10-07 — closer staged-learning references

- [TCL, Singh et al., CVPR 2021](https://rpand002.github.io/data/CVPR_2021_TCL.pdf):
  method Section 3.3 and benchmark results inspected. A close existing staged
  procedure for gesture/action videos, so stage ordering alone is not our novel
  contribution. Its RGB-video setting is distinct from our landmark inputs.
  The CVF HTML/PDF fetch returned 403; the author's paper copy was accessible.
- [FreeMatch](https://arxiv.org/abs/2205.07246): primary abstract inspected.
  Learning-state-adaptive thresholds and class fairness are established ideas;
  include a relevant comparator rather than labeling threshold adaptation new.
  Full implementation/protocol reading remains pending.
- [USB benchmark](https://proceedings.neurips.cc/paper_files/paper/2022/hash/190dd6a5735822f05646dc27decff19b-Abstract.html):
  primary abstract inspected. Relevant framework for fair SSL comparison and
  explicit pretrained resources; not a gesture-specific solution.
- [SHREC 2021](https://arxiv.org/abs/2106.10980): primary abstract inspected.
  Relevant continuous hand-skeleton evaluation; exact split, access, and sensor
  representation compatibility remain to verify before choosing a benchmark.

This extends the reference map, not an exhaustive determination of the current
best method or a completed novelty search. `RESEARCH_DIRECTION.md` records our
working hypothesis and primary evaluation setting.

## 2026-10-07 — robotics-focused review

Primary methods inspected in a focused parallel review: SafeDAgger, HG-DAgger,
ThriftyDAgger, Risk-Aware Active IRL, gesture semantics with incremental feedback,
SESAR, unified demonstrations/corrections/preferences, assistive latent-action
alignment, RCIP, KnowNo, CSST, robot semantic-segmentation active/SSL planning,
and AdaConG. Conformal Policy Control (ICML 2026) was screened for its stated
setting/assumptions. CVPR 2025 cost-effective active/SSL primary method text was
accessible through search indexing but the direct PDF request failed. Emergency
gesture HRI (2025) was available only as a publisher preview. Reading depth and
exact links are retained in `ROBOTICS_RESEARCH_IDEA.md`; unavailable methods must
not be assumed to omit our candidate idea.

These establish that risk-aware queries, robot gesture feedback, skeleton active
learning, cost-sensitive pseudo-label learning, and calibrated guidance inside
training already exist. A candidate joint audit/admission mechanism requires
comparison at the level of equations/code, not a claim based on combining names.
The standard finite-pool audit inequality is not itself a new contribution.

Benchmark update: SHREC 2021 paper and revised organizer annotations were
inspected, including separated train/test people and removal of POINTING.
FR-GESTURE (2026) was inspected as static RGBD UGV-command prior art. Neither
dataset was downloaded; compatibility, licenses, and reproduction remain
pending. See the robotics proposal for precise scope and source links.
