"""Shared recording format and preprocessing used by training AND prediction."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LABELS = ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", "NO_COMMAND")
KEYS = dict(zip("fblrsn", LABELS))
FRAMES = 32
SECONDS = 2.0
FEATURES = 67
VERSION = 1
CONVENTION = "mirrored_preview_v1"


def validate_clip(xyz, times, visible):
    xyz = np.asarray(xyz, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    if xyz.shape != (len(times), 21, 3) or visible.shape != times.shape:
        raise ValueError("Expected N frames of 21 xyz landmarks, N timestamps and N visibility flags.")
    if len(times) < 2 or not np.isfinite(xyz).all() or not np.isfinite(times).all():
        raise ValueError("A clip needs at least two finite frames.")
    if not np.all(np.diff(times) > 0):
        raise ValueError("Frame times must increase.")
    return xyz, times, visible


def quality(times, visible):
    """Reject short, sparse or badly tracked clips before issuing any command."""
    if len(times) < 12 or times[-1] - times[0] < SECONDS * 0.85:
        return False
    return bool(visible[-1] and np.mean(visible) >= 0.8 and np.max(np.diff(times)) <= 0.3)


def features(xyz, times, visible, aspect):
    """32 time samples: local hand shape + wrist motion + size change + visibility.

    We preserve horizontal direction and motion. Rotating every hand upright or
    discarding wrist movement would erase information needed for these commands.
    Missing frames are interpolated for shape, but retain a separate zero mask.
    """
    xyz, times, visible = validate_clip(xyz, times, visible)
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("Image aspect ratio must be positive.")
    target = np.linspace(times[0], times[-1], FRAMES)
    output = np.zeros((FRAMES, FEATURES), dtype=np.float32)
    if not visible.any():
        return output
    coords = xyz[visible].copy()
    # MediaPipe x/z are scaled by image width; put all axes in image-height units.
    coords[:, :, (0, 2)] *= aspect
    sample = np.empty((FRAMES, 21, 3), dtype=np.float32)
    for joint in range(21):
        for axis in range(3):
            sample[:, joint, axis] = np.interp(target, times[visible], coords[:, joint, axis])
    wrist = sample[:, 0:1, :]
    scale = np.maximum(np.linalg.norm(sample[:, 9, :2] - wrist[:, 0, :2], axis=1), 0.02)
    local = (sample - wrist) / scale[:, None, None]
    motion = (wrist[:, 0, :2] - wrist[0, 0, :2]) / scale[0]
    size_change = np.log(scale / scale[0])[:, None]
    nearest = np.abs(times[:, None] - target[None, :]).argmin(axis=0)
    mask = visible[nearest].astype(np.float32)[:, None]
    output = np.concatenate([local.reshape(FRAMES, 63), motion, size_change, mask], axis=1)
    return np.clip(output, -10, 10).astype(np.float32)


def save_clip(folder, xyz, times, visible, metadata):
    xyz, times, visible = validate_clip(xyz, times, visible)
    if metadata["label"] not in LABELS or metadata["split"] not in ("train", "validation", "test"):
        raise ValueError("Unknown label or dataset split.")
    metadata = {**metadata, "schema_version": VERSION, "convention": CONVENTION}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid4().hex}.npz"
    np.savez_compressed(path, xyz=xyz, times=times-times[0], visible=visible,
                        metadata=json.dumps(metadata))
    return path


def load_clip(path):
    with np.load(path, allow_pickle=False) as clip:
        metadata = json.loads(str(clip["metadata"]))
        xyz, times, visible = validate_clip(clip["xyz"], clip["times"], clip["visible"])
    if metadata["schema_version"] != VERSION or metadata["convention"] != CONVENTION:
        raise ValueError(f"Incompatible recording format: {path}")
    if metadata["label"] not in LABELS or metadata["split"] not in ("train", "validation", "test"):
        raise ValueError(f"Invalid label or split: {path}")
    return xyz, times, visible, metadata


def load_dataset(folder):
    samples = {name: [] for name in ("train", "validation", "test")}
    sessions = {}
    for path in sorted(Path(folder).glob("*.npz")):
        xyz, times, visible, meta = load_clip(path)
        session, split = meta["session"], meta["split"]
        if session in sessions and sessions[session] != split:
            raise ValueError(f"Session {session} appears in multiple splits. Keep recording sessions together.")
        sessions[session] = split
        samples[split].append((features(xyz, times, visible, meta["aspect"]),
                               LABELS.index(meta["label"]), str(path)))
    return samples


class CommandGate:
    """Screen-demo stability filter. Unknown/lost tracking immediately clears motion."""

    def __init__(self, threshold=0.8, margin=0.2, repeats=3):
        self.threshold, self.margin, self.repeats = threshold, margin, repeats
        self.reset()

    def reset(self):
        self.candidate, self.count = None, 0

    def update(self, probabilities, tracked):
        if not tracked:
            self.reset()
            return "NO_HAND", "STOP"
        probabilities = np.asarray(probabilities)
        order = np.argsort(probabilities)
        top = order[-1]
        if probabilities[top] < self.threshold or probabilities[top] - probabilities[order[-2]] < self.margin:
            self.reset()
            return "UNCERTAIN", "STOP"
        label = LABELS[top]
        if label in ("STOP", "NO_COMMAND"):
            self.reset()
            return label, "STOP"
        self.count = self.count + 1 if self.candidate == label else 1
        self.candidate = label
        return label, label if self.count >= self.repeats else "STOP"
