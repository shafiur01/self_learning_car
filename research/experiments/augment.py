"""Candidate transformations of normalized landmark sequences, applied in training.

These transformations are hypotheses about label preservation, not new human
examples. No reflection, rotation, reversal, label mixing, or mask invention.
"""

import torch
from torch.nn import functional as F

from gesture_commands.data import FEATURES, FRAMES

VARIANTS = ("baseline", "jitter", "time_warp", "combined")


def augment(batch, variant, generator, settings):
    """Return a new B x 32 x 67 tensor; never alter the recorded features."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown augmentation: {variant}")
    if batch.ndim != 3 or batch.shape[1:] != (FRAMES, FEATURES):
        raise ValueError("Expected a batch of 32 x 67 feature sequences.")
    result = batch.clone()
    if variant in ("time_warp", "combined"):
        segments = settings["time_segments"]
        low, high = settings["time_increment_range"]
        # Positive increments create an increasing time map with fixed endpoints.
        increments = low + (high - low) * torch.rand(
            (len(batch), segments), generator=generator, dtype=batch.dtype)
        knots = torch.cat((torch.zeros_like(increments[:, :1]), increments.cumsum(1)), dim=1)
        knots = knots / knots[:, -1:] * (FRAMES - 1)
        time_map = F.interpolate(knots[:, None, :], size=FRAMES,
                                 mode="linear", align_corners=True)[:, 0, :]
        left = time_map.floor().long().clamp(0, FRAMES - 1)
        right = (left + 1).clamp(max=FRAMES - 1)
        rows = torch.arange(len(batch))[:, None]
        fraction = (time_map - left).unsqueeze(-1)
        result = batch[rows, left] * (1 - fraction) + batch[rows, right] * fraction
        # Visibility is categorical: sample the nearest frame, never interpolate.
        nearest = time_map.round().long().clamp(0, FRAMES - 1)
        result[:, :, -1] = batch[rows, nearest, -1]
    if variant in ("jitter", "combined"):
        anchors = torch.randn((len(batch), 63, settings["jitter_anchors"]),
                              generator=generator, dtype=batch.dtype)
        noise = F.interpolate(anchors, size=FRAMES, mode="linear", align_corners=True)
        noise = noise.transpose(1, 2) * settings["jitter_std"]
        # Coordinates are wrist-relative: the wrist itself must remain zero.
        noise[:, :, :3] = 0
        # All-invisible examples stay all zero; visibility never becomes invented.
        result[:, :, :63] += noise * result[:, :, -1:]
    return result.clamp(-10, 10)
