"""Mathematical feature properties and meaningful tiny temporal-fit checks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import landmark_representations as rep
from . import motion_representation as experiment


def raw(n=120):
    times = np.linspace(0, 2, n)
    pose = np.zeros((21, 3))
    pose[:, 0] = np.linspace(-.04, .04, 21)
    pose[:, 1] = np.linspace(0, -.16, 21)
    pose[0] = 0
    pose[9] = [0, -.1, 0]
    pose[5] = [.035, -.09, .005]
    pose[17] = [-.04, -.075, -.004]
    xyz = np.tile(pose, (n, 1, 1))
    xyz[:, 8, 1] += .025 * np.sin(times * 2)
    xyz[:, :, 0] += .4 + .03 * times[:, None]
    xyz[:, :, 1] += .7
    return xyz.astype(np.float32), times, np.ones(n, dtype=bool)


class FeatureChecks(unittest.TestCase):
    def test_translation_scale_and_orientation_preserving_hand_frame(self):
        xyz, t, v = raw()
        camera, _ = rep.features(xyz, t, v, 1, "camera_motion")
        hand, _ = rep.features(xyz, t, v, 1, "hand_motion")
        transformed, _ = rep.features(xyz * 1.6 + np.array([.2, -.1, .03]), t, v, 1, "hand_motion")
        np.testing.assert_allclose(hand, transformed, atol=3e-4, rtol=3e-4)
        basis = hand[:, 134:143].reshape(32, 3, 3)
        np.testing.assert_allclose(np.matmul(basis.transpose(0, 2, 1), basis), np.tile(np.eye(3), (32, 1, 1)), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(basis), 1, atol=1e-6)
        restored = np.matmul(hand[:, :63].reshape(32, 21, 3), basis.transpose(0, 2, 1))
        np.testing.assert_allclose(restored, camera[:, :63].reshape(32, 21, 3), atol=1e-6)
        rotated = xyz @ np.diag([-1., -1., 1.])
        flipped, _ = rep.features(rotated, t, v, 1, "hand_motion")
        np.testing.assert_allclose(flipped[:, :63], hand[:, :63], atol=1e-6)
        np.testing.assert_allclose(flipped[:, 63:65], -hand[:, 63:65], atol=1e-6)
        self.assertGreater(np.max(np.abs(flipped[:, 134:143] - hand[:, 134:143])), 1)

    def test_actual_time_stretch_halves_velocity_and_retains_shape(self):
        xyz, t, v = raw()
        normal, _ = rep.features(xyz, t, v, 1, "camera_motion")
        slow, _ = rep.features(xyz, t * 2, v, 1, "camera_motion")
        np.testing.assert_allclose(normal[:, :67], slow[:, :67], atol=1e-6)
        np.testing.assert_allclose(slow[:, 67:133], normal[:, 67:133] / 2, atol=1e-6)
        np.testing.assert_allclose(slow[:, 144:146], normal[:, 144:146] * 2)
        pose, _ = rep.features(xyz, t, v, 1, "camera_pose")
        self.assertFalse(np.any(pose[:, 67:134]))
        np.testing.assert_array_equal(pose[:, 134:], normal[:, 134:])

    def test_missing_tracking_and_large_gaps_cannot_create_velocity(self):
        xyz, t, v = raw(200)
        v[101] = False
        x, _ = rep.features(xyz, t, v, 1, "camera_motion")
        target = np.linspace(0, 2, 32)
        index = np.searchsorted(target, t[101])
        self.assertEqual(x[index, 133], 0)
        self.assertFalse(np.any(x[index, 67:133]))
        keep = (t < .5) | (t > 1.5)
        gap, _ = rep.features(xyz[keep], t[keep], np.ones(keep.sum(), dtype=bool), 1, "camera_motion")
        self.assertFalse(np.any(gap[(target > .6) & (target < 1.4), 67:134]))

    def test_no_hand_degenerate_basis_and_prefix_future_isolation(self):
        xyz, t, v = raw()
        no_hand, _ = rep.features(xyz, t, np.zeros_like(v), 1, "hand_motion")
        self.assertFalse(np.any(no_hand[:, :144]))
        self.assertGreater(no_hand[-1, 144], 0)
        degenerate, _ = rep.features(np.zeros_like(xyz), t, v, 1, "hand_motion")
        self.assertTrue(np.isfinite(degenerate).all())
        self.assertFalse(np.any(degenerate[:, 67:144]))
        px, pt, pv = rep.prefix(xyz, t, v, .5)
        original, _ = rep.features(px, pt, pv, 1, "hand_motion")
        xyz[t > .5] += 50
        changed, _ = rep.features(*rep.prefix(xyz, t, v, .5), 1, "hand_motion")
        np.testing.assert_array_equal(original, changed)
        self.assertLessEqual(pt[-1] - pt[0], .5)


class FittingChecks(unittest.TestCase):
    def test_tiny_fit_pairing_gradients_and_strict_checkpoint(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(123)
        arrays = {"train_x": rng.normal(size=(40, 32, 146)).astype(np.float32),
                  "train_y": np.tile(np.arange(8), 5),
                  "validation_x": rng.normal(size=(8, 32, 146)).astype(np.float32),
                  "validation_y": np.arange(8)}
        config = {"optimizer_steps": 2, "validation_every_steps": 1, "batch_size": 16, "learning_rate": .001}
        model, history, stats = experiment.fit(arrays, 11, config)
        model2, history2, stats2 = experiment.fit(arrays, 11, config)
        self.assertEqual(stats["initial_state_sha256"], stats2["initial_state_sha256"])
        self.assertEqual(stats["batch_indices_sha256"], stats2["batch_indices_sha256"])
        self.assertEqual(history, history2)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 49276)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            checkpoint = {"state_dict": model.state_dict(), "experiment_id": "010_motion_representation",
                "feature_version": 2, "features": 146, "labels": list(experiment.previous.v2.LABELS),
                "frames": 32, "architecture": "lstm146_hidden47_four_bins", "schema_version": 3,
                "mode": "hand_motion", "command_contract": experiment.previous.training.collection.contract("three_shapes")}
            torch.save(checkpoint, path)
            loaded, _ = experiment.load_model(path)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(torch.from_numpy(arrays["validation_x"])), model2(torch.from_numpy(arrays["validation_x"])))
            checkpoint["feature_version"] = 1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                experiment.load_model(path)


if __name__ == "__main__":
    unittest.main()
