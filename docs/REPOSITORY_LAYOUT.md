# Repository layout and reproducibility

The code keeps its current module paths. Existing experiments pin their source
hashes and input paths, so moving/formatting those modules would invalidate old
audit records. New repository tooling lives separately from scientific code.
`.gitattributes` retains LF source lines and disables newline conversion for
hashed result exports, so normal checkout settings do not change their bytes.

```text
main.py                        Original webcam hand-landmark demonstration
gesture_commands/              Six-command application, data processing and trainer
models/README.md               Instructions for downloading the hand detector
research/
  CONTEXT.md                   Current state and constraints; read first
  PROTOCOL.md / LOG.md          Declared decisions and chronological outcomes
  findings/                    Interpretations, including negative results
  experiments/                 Versioned recording, fitting and audit modules
    configs/                   Early experiment configuration files
    review_assets/             Reusable viewer HTML/JavaScript
    data/                      Local recordings/features (excluded from Git)
    runs/                      Full local runs/checkpoints (excluded from Git)
  results/                     Portable configuration, metrics, plots and hashes
  notes/                       Personal/internal feedback (excluded from Git)
  private/                     Local inventory/check reports (excluded from Git)
tools/                         Exporter, read-only Git checker and synthetic-test runner
docs/                          This map and the push/backup guide
requirements-research.txt       Direct dependencies for research/tests/plots
```

The eight-command research workflow is versioned separately from the original
application. See [current research](../research/CONTEXT.md); the older
[v3 collection workflow](../research/COMMANDS_V3_WORKFLOW.md) documents completed
collection, not a request to record more personal validation/test gestures.
Research checkpoints never automatically replace the camera application's model.

## What Git preserves

Git should preserve source, tests, requirements, plans, interpretations, the
research log and the compact [results catalog](../research/results/README.md).
Historical lesson/protocol pages remain for continuity. Their references into
`experiments/data/`, `experiments/runs/` or `notes/` refer to local archives and
will not open on GitHub. Each finding links to a portable exported record instead.

Results retain all exported seed outcomes, including unsuccessful hypotheses and
the selected/final ranking reversal. Exporting aggregate tables does not erase
full private error lists; those stay with the original archives.

## What requires a separate backup

Recordings, feature arrays, weights, full prediction traces, viewer bundles,
source snapshots inside runs, and personal feedback are ignored. Ignored files
are not included by an ordinary Git push. The exporter creates
`research/private/artifact_inventory.json`, with relative filenames, sizes and
SHA-256 values. The public `research/results/archives.json` contains only group
totals/index hashes and explicitly records that no off-device backup was made.

A GitHub clone is therefore a code/results record, not a complete experiment
backup. Preserve the original directory trees in private storage. Model files
can later be placed in separately versioned artifact storage or Git LFS after a
deliberate release decision. No artifact service or remote upload is configured.

## Reproduction levels

1. **Inspect the evidence:** read findings and tracked result tables. Verify the
   original/export checksums in each `provenance.json`.
2. **Check software without recordings:** install research requirements and run
   synthetic tests with `python -m tools.run_software_checks`. Its temporary asset
   fixture supports three fully mocked legacy camera tests without downloading a
   detector or changing production guards. These include small synthetic
   optimization checks, not new
   research fits, camera capture or evaluation of personal recordings.
3. **Rerun historical studies:** obtain the complete authorized private archives,
   recorded environments and dependencies. Some historical manifests pin absolute
   local paths and parent-run hashes. They need the original layout or a separately
   audited portability/migration layer; this preparation does not implement that
   layer or claim a one-command full rerun from a clean clone.

New benchmark work should use a portable input/archive interface. It should not
silently rewrite old frozen inputs to make historical checks pass.
