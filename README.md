# Temporal gesture learning for robot commands

A hand-landmark demonstration and a research workspace for label-efficient
recognition of temporal hand gestures. The research currently covers FORWARD,
BACKWARD, LEFT, RIGHT, STOP, NO_COMMAND, UP and DOWN. It uses a pretrained
MediaPipe hand detector and small learned classifiers; physical robot control,
unseen-user reliability and a novel training algorithm remain unproved.

Start here:

- [Push/backup guide](docs/GITHUB_PUSH_GUIDE.md): exact review, commit and push commands.
- [Repository map](docs/REPOSITORY_LAYOUT.md): code, records and private artifacts.
- [Current research state](research/CONTEXT.md) and [latest finding018](research/findings/018_online_softmatch.md).
- [Portable results catalog](research/results/README.md): configurations, all available seed results and plots for20 historical records.
- [Research history](research/README.md): protocols, findings and earlier lessons.

## Current evidence

The latest15-fit comparison uses160 revealed training labels and245 inputs with
labels withheld. Soft weighting improves development-selected F1, while the
confidence baseline is stronger at the final checkpoint. Both outcomes are
recorded; no algorithm is promoted. Existing48/96 clips are inspected development/
retrospective data, not an untouched final benchmark. No research checkpoint is
automatically activated in the camera application.

## Setup on macOS

The recorded research environment is Python3.14.7 on Apple silicon. Direct
versions are recorded in the requirements files and exported environments.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt
```

For just the original camera demonstration, install `requirements.txt` instead.
MediaPipe is pinned to0.10.32 because1.0.1 crashed during initialization on the
original Mac. Use `opencv-contrib-python` as pinned; do not install a second
OpenCV package owning the same `cv2` module.

Download the detector using [models/README.md](models/README.md), then:

```bash
python main.py
```

Allow camera access for the application running Python. Show one hand to see21
landmarks; press lowercase **q** to exit. If the camera is blocked, enable the
application in System Settings → Privacy & Security → Camera and reopen it.

The original [six-command application](gesture_commands/README.md) and historical
[eight-command v3 workflow](research/COMMANDS_V3_WORKFLOW.md) are documented
separately. Historical collection instructions are retained as records, not a
request for more personal validation/test recordings.

## Data-free checks

```bash
.venv/bin/python -m tools.run_software_checks
.venv/bin/python -m tools.check_repository
```

Software tests use synthetic data, including small toy fits. A test-only temporary
asset supplies the missing file fixture for three fully mocked camera tests; real
camera/detector calls are blocked there. No detector download is needed for these
checks, and no production guard is changed. The tests do not launch historical
experiments or record personal gestures. Full historical reruns require private
recordings/run archives and their recorded environments; exported tables alone
are not those inputs. Some historical loaders pin original local paths. See
[reproduction levels](docs/REPOSITORY_LAYOUT.md#reproduction-levels).

## What belongs in Git

Code, tests, requirements, protocols, findings and compact results are prepared
for Git. `.gitignore` excludes environments, recordings, tensors, weights, local
run trees and personal notes. Original archives are preserved locally and need
separate private backup; an ordinary Git push does not back up ignored files.

This preparation does not stage, commit or push. Follow the
[push guide](docs/GITHUB_PUSH_GUIDE.md) after reviewing the prepared files. The
current `origin` repository is public; the guide includes a private-remote option.
