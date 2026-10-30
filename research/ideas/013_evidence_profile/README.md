# Candidate algorithm: Evidence-Profile Self-Training

**Subsequent pilot:** [experiment013](../../findings/013_evidence_profile.md) is
now complete. It did not beat its main confidence baseline; almost no pseudo-
labels passed admission. The design below is preserved as the original proposal.

Proposed 2026-10-15. Working name, not an established novelty or performance
claim. At proposal time no model had been trained; the subsequent negative
pilot linked above is now the current empirical result.

**Central idea:** before a teacher's automatic label can teach another model,
check whether the teacher's response to missing shape/motion information resembles
its response on known examples of that proposed command.

We already found that explicit motion does not help every model or recording
sitting. Some intended commands are held poses; others involve motion. A single
rule such as “every view must agree” or “motion must be large” is therefore an
unjustified universal constraint. Confidence alone has also been wrong here.

The proposed rule asks whether an automatic label has *compatible evidence use*,
without demanding that incomplete observations remain certain. It could reduce
the propagation of confidently wrong labels. It can also fail, especially when
wrong predictions have the same response pattern as correct ones.

Start with [the beginner explanation](../../LESSON_17.md). The precise candidate,
equations, proof limits and failure cases are in [ALGORITHM.md](ALGORITHM.md).
The fixed-data study is drafted in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

No more personal validation/test recordings will be requested unless the user
explicitly changes that instruction. Existing405 original training recordings
are the proposed primary pool:80 revealed labels,325 labels hidden from the
learner. That is a simulation of fewer labels, not fewer total inputs or actual
saved annotation time. Existing48/96 evaluation roles remain separate; all have
been historically inspected, so no untouched-test claim is possible.

This is a proposal ready for implementation planning. Eight synthetic mathematical
checks passed, including a constructed failure that the method cannot catch.
They are not measured gesture results or proof of improved accuracy.

Closest ingredients already exist: [ModDrop](https://arxiv.org/abs/1501.00102),
[FixMatch](https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html),
and [self-training with rationalization](https://aclanthology.org/2021.emnlp-main.836/).
The question is whether this particular cross-fitted, class-conditional response
admission rule adds measurable value. The bounded review does not establish it
as unpublished. No SOTA, efficiency, safety or label-saving claim yet.

Supporting reviews:

- [Prior art](../../notes/2026-10-15_algorithm_prior_art_agent.md).
- [Mathematical design critique](../../notes/2026-10-15_algorithm_design_agent.md).
- [Existing-data feasibility](../../notes/2026-10-15_fixed_data_protocol_agent.md).

`feasibility.json` records metadata counts/provenance only. No endpoint was
rescored or used for fitting during this design step. All earlier checkpoints,
recordings and one-time evaluation ledgers remain unchanged.
