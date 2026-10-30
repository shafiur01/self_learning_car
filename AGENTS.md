# Project continuity

When working on the research workflow, first read `research/CONTEXT.md`,
`research/PROTOCOL.md`, and the latest entries in `research/LOG.md`.
Keep those files current when decisions, experiments, or results change.

The user is learning ML: explain what an experiment tests, why a file exists,
and what a result does and does not establish. Carry out authorized work.

Record unfavorable results as well as favorable results. Do not claim novelty,
publication readiness, general reliability, or label savings without evidence.
Preserve prior checkpoints and recordings. Keep recording sessions together.
Current user constraint (2026-10-15): do not request any more personal validation
or test gesture recordings. Use existing recordings for development/evaluation;
describe already-inspected endpoints as retrospective, never untouched. This
constraint persists unless the user explicitly changes it.
Use validation during development; reserve genuinely new sessions for final
evaluation after choices are frozen. Historical test results have already been
examined and cannot be repeatedly reused as an untouched final benchmark.

Research code lives in `research/experiments/`. It reuses the existing landmark
preprocessing and temporal model. It does not silently activate a checkpoint in
the camera application. Use the existing `.venv`; do not introduce dependencies
without a concrete need.
