"""Feature version 2: hand shape/orientation and masked physical-time motion.

Experimental alternative; never edits the frozen 67-feature application path.
"""

import numpy as np

from gesture_commands.data import validate_clip

FRAMES, FEATURES, VERSION = 32, 146, 2
MODES = ("camera_pose", "camera_motion", "hand_pose", "hand_motion")


def palm_basis(pose):
    """Right-handed orthonormal columns: lateral, longitudinal, normal."""
    longitudinal = pose[:, 9] - pose[:, 0]
    length = np.linalg.norm(longitudinal, axis=1)
    y = longitudinal / np.maximum(length[:, None], 1e-12)
    lateral = pose[:, 5] - pose[:, 17]
    lateral -= np.sum(lateral * y, axis=1)[:, None] * y
    lateral_length = np.linalg.norm(lateral, axis=1)
    x = lateral / np.maximum(lateral_length[:, None], 1e-12)
    z = np.cross(x, y)
    x = np.cross(y, z)
    valid = (length > 1e-4) & (lateral_length > 1e-4)
    basis = np.stack((x, y, z), axis=2)
    basis[~valid] = 0
    return basis, valid


def interval_mask(times, visible, target, geometry_valid):
    valid = np.zeros(len(target), dtype=bool)
    for i in range(1, len(target)):
        left = max(0, int(np.searchsorted(times, target[i - 1], side="right")) - 1)
        right = min(len(times) - 1, int(np.searchsorted(times, target[i], side="left")))
        valid[i] = bool(geometry_valid[i - 1] and geometry_valid[i]
            and visible[left:right + 1].all()
            and (right == left or np.diff(times[left:right + 1]).max() <= .3))
    return valid


def features(xyz, times, visible, aspect, mode):
    if mode not in MODES:
        raise ValueError("Unknown experimental representation.")
    xyz, times, visible = validate_clip(xyz, times, visible)
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("Invalid camera aspect.")
    target = np.linspace(times[0], times[-1], FRAMES)
    result = np.zeros((FRAMES, FEATURES), dtype=np.float32)
    result[:, 144] = target - target[0]
    result[:, 145] = target[1] - target[0]
    if not visible.any():
        return result, {"no_hand": True, "invalid_geometry_samples": FRAMES,
                        "valid_motion_intervals": 0, "position_clipped": 0, "velocity_clipped": 0}
    coords = xyz[visible].astype(np.float64)
    coords[:, :, (0, 2)] *= aspect
    sample = np.empty((FRAMES, 21, 3), dtype=np.float64)
    for joint in range(21):
        for axis in range(3):
            sample[:, joint, axis] = np.interp(target, times[visible], coords[:, joint, axis])
    nearest = np.abs(times[:, None] - target[None, :]).argmin(axis=0)
    tracked = visible[nearest]
    wrist = sample[:, 0:1].copy()
    scale = np.maximum(np.linalg.norm(sample[:, 9] - sample[:, 0], axis=1), .02)
    camera = (sample - wrist) / scale[:, None, None]
    basis, basis_valid = palm_basis(camera)
    geometry_valid = tracked & basis_valid
    shape = np.matmul(camera, basis) if mode.startswith("hand_") else camera.copy()
    shape[~geometry_valid] = 0
    global_motion = np.column_stack(((wrist[:, 0, :2] - wrist[0, 0, :2]) / scale[0], np.log(scale / scale[0])))
    global_motion[~geometry_valid] = 0
    positions = np.column_stack((shape.reshape(FRAMES, 63), global_motion))
    result[:, :66] = np.clip(positions, -10, 10)
    result[:, 66] = tracked
    orientation = basis.copy()
    orientation[~geometry_valid] = 0
    result[:, 134:143] = orientation.reshape(FRAMES, 9)
    result[:, 143] = geometry_valid
    valid = interval_mask(times, visible, target, geometry_valid)
    velocity = np.zeros_like(positions)
    velocity[1:] = np.diff(positions, axis=0) / np.diff(target)[:, None]
    velocity[~valid] = 0
    if mode.endswith("_motion"):
        result[:, 67:133] = np.clip(velocity, -50, 50)
        result[:, 133] = valid
    if not np.isfinite(result).all():
        raise ValueError("Non-finite experimental features.")
    return result, {"no_hand": False, "invalid_geometry_samples": int(np.sum(~geometry_valid)),
        "valid_motion_intervals": int(valid.sum()),
        "position_clipped": int(np.sum(np.abs(positions) > 10)),
        "velocity_clipped": int(np.sum(np.abs(velocity) > 50)) if mode.endswith("_motion") else 0}


def prefix(xyz, times, visible, seconds):
    """Return only captured samples at or before an elapsed deadline."""
    xyz, times, visible = validate_clip(xyz, times, visible)
    indices = np.flatnonzero(times - times[0] <= seconds + 1e-10)
    if len(indices) < 2:
        raise ValueError("A prefix requires at least two observed samples.")
    return xyz[indices].copy(), times[indices].copy(), visible[indices].copy()
