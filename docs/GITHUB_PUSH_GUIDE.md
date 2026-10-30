# Prepare, commit and push this research snapshot

The current remote is `https://github.com/shafiur01/self_learning_car.git` and
is public (checked 2026-10-30). The commands below publish the prepared source,
protocols and compact research results there. A new branch is also public.
Recordings, model weights, internal notes and full run archives are excluded.
No branch, index, commit or remote has been changed by this preparation.

Prepared 2026-10-30: 20 historical records exported; 198 software checks pass in
a data-free copy. Exported numeric records match their originals, and a scratch-
repository staging check passes. Frozen scientific code remains unchanged.
[Preparation check record](PREPARATION_CHECKS.json).

## 1. Open the project terminal and check the files

In VS Code, open this project folder, then **Terminal → New Terminal**. Run:

```bash
git status --short --branch
git remote -v
.venv/bin/python -m tools.check_repository
```

The checker should report `"passed": true`. It checks the prepared files,
exported-result hashes, common credential patterns, local home paths and excluded
artifacts. It is a basic check, not a substitute for reviewing the diff.

The results are already exported. You only need to run
`.venv/bin/python -m tools.export_research_records` again when verifying or
preparing from the unchanged original local archives. Changed existing exports
are rejected for review rather than silently replaced.

## 2. Optional: run the software checks yourself

The existing virtual environment is sufficient. On a fresh clone, first create
`.venv` and install `requirements-research.txt`, as described in the root README.

```bash
.venv/bin/python -m tools.run_software_checks
```

These are synthetic software checks, including small toy fits. Three older mocked-
camera tests assumed the downloaded detector file existed; this runner supplies a
temporary test-only placeholder and blocks real camera/detector access for those
tests. It does not modify production code or put a fake model in `models/`.
Ordinary direct discovery of those older tests still needs the downloaded asset;
use this runner for a data-free clone. It does not record personal gestures or
launch the historical research experiments. Use `--suite packaging`, `--suite
research` or `--suite application` to run only one suite.

## 3. Create a branch and stage the intended files

For this first snapshot:

```bash
git switch -c research/checkpoint-018
git add .gitignore .gitattributes README.md AGENTS.md requirements-research.txt docs tools research
.venv/bin/python -m tools.check_repository --staged
git diff --cached --stat
git diff --cached --name-only
git diff --cached --check
```

`git add` stages a snapshot for your next commit. Existing ignore rules keep the
private/binary directories out; do not use `git add -f` for them. Inspect the
listed paths. Expect code, documents, compact JSON/CSV tables, histories and
scientific figures. Do not expect `.venv`, `.npz`, `.pt`, recordings, `runs/`,
`research/notes/` or `research/private/`.

If the original local archives are present, `.venv/bin/python -m
tools.check_repository --with-private-sources` also verifies that exported values
and original-file hashes match. It needs no model loading or retraining.

The staged checker reads **the actual index contents**. If you edit files after
staging, stage those files again and rerun the check. To remove an unintended
file from the next commit while keeping it on your Mac:

```bash
git restore --staged -- path/to/unwanted_file
```

## 4. Commit and push

Check that Git will attribute the commit to your intended identity:

```bash
git var GIT_AUTHOR_IDENT
```

Check the displayed name/email **before committing**. If missing or incorrect,
set them for this repository with `git config user.name "Your Name"` and
`git config user.email "Your GitHub Email"`. Then run:

```bash
git commit -m "Preserve gesture research through experiment 018"
git push -u origin research/checkpoint-018
git status --short --branch
git log -1 --oneline
```

The commit records this snapshot locally. The push copies it to GitHub. Then
select `research/checkpoint-018` on GitHub and open a pull request to review/merge
the snapshot if desired. Historical experiment dates stay in the records; the
new commit has its actual date, rather than pretending those runs were committed
earlier. Git history documents changes; it does not by itself prove novelty.

If you prefer private research storage, create a **new private repository** in
your GitHub account first. Use a separate remote instead of the public push:

```bash
git remote add research-private https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git
git push -u research-private research/checkpoint-018
```

Replace the two placeholders with the repository you created. That command is
an alternative to `git push -u origin ...`, not an additional required push.

## 5. Back up what Git intentionally excludes

The full local recordings and experiment archives still need separate private
backup. From the project folder, this optional command creates an archive next
to the project, outside the files you are staging:

```bash
tar --exclude='__pycache__' --exclude='plot_cache' -czf ../self-learning-car-artifacts-2026-10-30.tar.gz \
  gesture_commands/data gesture_commands/runs \
  research/experiments/data research/experiments/runs \
  research/notes research/private models
```

Use a new archive filename if one already exists. Copy the archive to private
off-device storage and verify it there. Creating another file on the same Mac
does not establish an off-device backup. Do not add that archive to ordinary Git.
The original files remain intact; this preparation has not uploaded or deleted
them. See [the repository map](REPOSITORY_LAYOUT.md) for reproduction limits.
