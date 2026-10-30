# Git-tracked research records

These are compact exports of **20 historical experiment/evaluation records**.
Both favorable and unfavorable seed outcomes are retained. They support reading
the evidence on GitHub without uploading personal recordings or all model files.

- [Catalog](index.json): experiment IDs, archive locations, findings and export sizes.
- [018: latest online comparison](018_online_softmatch/README.md): includes both
  development-selected and final outcomes; the method ranking reverses.
- [014: response-profile comparison](014_budget160/README.md): no supported
  proposed-method advantage.
- [Private artifact inventory summary](archives.json): sizes/content-index hashes,
  **not** a claim that the original archives have been backed up off this Mac.
- [Preparation environment](preparation_environment.json): directly observed
  packages; each study also retains its recorded environment when available.

Every experiment folder has `provenance.json`, original-file SHA-256 values,
export-file SHA-256 values, configuration/summary/seed records when available,
and approved charts or validation-loss histories. Some early evaluation/diagnostic
records have no separate configuration or training histories; they are not invented.

Only local path strings are normalized in JSON/CSV exports. Numerical values are
preserved. Original freezes are referenced by hash, not silently rewritten. The
exports must not be passed to historical loaders as if they were original files.
Raw coordinates, embeddings, per-clip viewer bundles, model weights and personal
notes remain local. Full original failure lists and provenance remain in those
private archives; aggregate and per-seed failures remain visible here/findings.

The current48/96 recordings have been inspected and are development/retrospective
data. No untouched final benchmark, actual annotation savings, new algorithm or
physical robot reliability is established by this export.

Recreate/verify from the original local archives:

```bash
.venv/bin/python -m tools.export_research_records
.venv/bin/python -m tools.check_repository
```

The exporter is idempotent and refuses to overwrite a changed existing record.
A fresh clone contains these exports but not the private archives, so it can
verify exported checksums and run synthetic software tests; it cannot recreate
historical training from the exports alone. See
[reproduction/storage guidance](../../docs/REPOSITORY_LAYOUT.md) and
[push instructions](../../docs/GITHUB_PUSH_GUIDE.md).
