"""Synthetic checks verify plumbing, never real-world recognition accuracy."""

import tempfile
import sys
from pathlib import Path
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from .app import snapshot
from .data import (CONVENTION, FEATURES, FRAMES, LABELS, SECONDS, VERSION,
                   CommandGate, features, load_clip, load_dataset, quality, save_clip)
from .model import GestureNet, load_model
from .train import require_classes, train_model


def sample(direction=1):
    times = np.linspace(0, 1.98, 60)
    xyz = np.zeros((60, 21, 3), dtype=np.float32)
    xyz[:, :, 0] = 0.4 + direction * times[:, None] * 0.05
    xyz[:, :, 1] = 0.4 + np.arange(21)[None, :] * 0.01
    visible = np.ones(60, dtype=bool)
    return xyz, times, visible


class FeaturesTests(unittest.TestCase):
    def test_translation_invariance_and_direction_preserved(self):
        xyz, times, visible = sample()
        original = features(xyz, times, visible, 16/9)
        moved = xyz.copy()
        moved[:, :, :2] += 0.1
        np.testing.assert_allclose(original, features(moved, times, visible, 16/9), atol=1e-5)
        self.assertEqual(original.shape, (FRAMES, FEATURES))
        self.assertGreater(original[-1, 63], 0)
        reverse, _, _ = sample(-1)
        self.assertLess(features(reverse, times, visible, 16/9)[-1, 63], 0)

    def test_missing_hand_and_visibility_mask(self):
        xyz, times, visible = sample()
        visible[:] = False
        self.assertFalse(features(xyz, times, visible, 1).any())
        visible[::2] = True
        processed = features(xyz, times, visible, 1)
        self.assertTrue(np.isfinite(processed).all())
        self.assertIn(0, processed[:, -1])
        self.assertIn(1, processed[:, -1])

    def test_actual_timestamps_are_used(self):
        xyz, times, visible = sample()
        indices = np.array([0, 1, 5, 9, 10, 20, 35, 50, 59])
        np.testing.assert_allclose(features(xyz, times, visible, 1),
                                   features(xyz[indices], times[indices], visible[indices], 1), atol=1e-5)

    def test_quality_rejects_short_clips_and_stalls(self):
        _, times, visible = sample()
        self.assertTrue(quality(times, visible))
        self.assertFalse(quality(times[:10], visible[:10]))
        visible[-1] = False
        self.assertFalse(quality(times, visible))
        visible[-1] = True
        times[30:] += 1
        self.assertFalse(quality(times, visible))

    def test_invalid_timestamps_fail(self):
        xyz, times, visible = sample()
        times[1] = times[0]
        with self.assertRaises(ValueError):
            features(xyz, times, visible, 1)


class RecordingTests(unittest.TestCase):
    def test_frozen_copy_roundtrip_and_correction_metadata(self):
        xyz, times, visible = sample()
        buffer = [(t, x.copy(), v, None) for x, t, v in zip(xyz, times, visible)]
        frozen = snapshot(buffer)
        buffer[-1][1][:] = 99
        self.assertFalse((frozen[0][-1] == 99).any())
        with tempfile.TemporaryDirectory() as directory:
            meta = dict(label="FORWARD", split="train", session="practice", aspect=1.5,
                        prediction="BACKWARD", source="correction")
            path = save_clip(directory, *frozen, meta)
            stored_xyz, stored_times, stored_visible, stored_meta = load_clip(path)
            np.testing.assert_equal(stored_xyz, frozen[0])
            np.testing.assert_allclose(stored_times, frozen[1] - frozen[1][0])
            np.testing.assert_equal(stored_visible, frozen[2])
            self.assertEqual(stored_meta["prediction"], "BACKWARD")
            self.assertEqual(stored_meta["label"], "FORWARD")

    def test_reused_session_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            for split in ("train", "test"):
                save_clip(directory, *sample(), dict(label="FORWARD", split=split, session="same", aspect=1))
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                load_dataset(directory)


class GateTests(unittest.TestCase):
    def test_stability_then_loss_revokes_movement(self):
        gate = CommandGate()
        forward = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.assertEqual(gate.update(forward, True)[1], "STOP")
        self.assertEqual(gate.update(forward, True)[1], "STOP")
        self.assertEqual(gate.update(forward, True)[1], "FORWARD")
        self.assertEqual(gate.update(forward, False), ("NO_HAND", "STOP"))
        self.assertEqual(gate.update(forward, True)[1], "STOP")

    def test_uncertainty_stop_and_command_change(self):
        gate = CommandGate(repeats=1)
        self.assertEqual(gate.update(np.ones(6) / 6, True), ("UNCERTAIN", "STOP"))
        for label in ("STOP", "NO_COMMAND"):
            scores = np.zeros(6)
            scores[LABELS.index(label)] = 1
            self.assertEqual(gate.update(scores, True), (label, "STOP"))
        gate = CommandGate(repeats=2)
        gate.update(np.eye(6)[0], True)
        gate.update(np.eye(6)[0], True)
        self.assertEqual(gate.update(np.eye(6)[2], True), ("LEFT", "STOP"))


class TrainingTests(unittest.TestCase):
    def test_requires_all_classes(self):
        with self.assertRaisesRegex(ValueError, "Record more"):
            require_classes([], "train", 5)

    def test_synthetic_training_checkpoint_roundtrip(self):
        rng = np.random.default_rng(1)
        dataset = {"train": [], "validation": [], "test": []}
        for split, count in (("train", 5), ("validation", 2)):
            for label in range(len(LABELS)):
                for _ in range(count):
                    x = rng.normal(0, 0.01, (FRAMES, FEATURES)).astype(np.float32)
                    x[:, label] += 1
                    dataset[split].append((x, label, "synthetic"))
        model, epoch, validation = train_model(dataset, epochs=3, seed=10)
        self.assertIn(epoch, (1, 2, 3))
        self.assertEqual(validation["clips"], 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(dict(state_dict=model.state_dict(), schema_version=VERSION,
                            convention=CONVENTION, labels=list(LABELS), frames=FRAMES, seconds=SECONDS), path)
            loaded, _ = load_model(path)
            x = torch.from_numpy(dataset["validation"][0][0])[None]
            with torch.inference_mode():
                torch.testing.assert_close(model(x), loaded(x))
            self.assertEqual(tuple(loaded(x).shape), (1, 6))

    def test_rejects_wrong_label_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"labels": ["WRONG"]}, path)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                load_model(path)


class AppWorkflowTests(unittest.TestCase):
    def exercise_capture(self, correction):
        # Replace physical camera/window/detector with controlled frames and keys.
        # Real save, replay, preprocessing, and (for corrections) inference paths run.
        import cv2
        import mediapipe as mp
        from . import app
        xyz, _, _ = sample()
        landmarks = [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in xyz[0]]
        result = SimpleNamespace(hand_landmarks=[landmarks],
                                 handedness=[[SimpleNamespace(category_name="Right")]])
        detector = MagicMock()
        detector.__enter__.return_value = detector
        detector.detect_for_video.return_value = result
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = lambda: (True, np.zeros((360, 640, 3), dtype=np.uint8))
        clock = iter(np.arange(100, 200, 0.04))
        argv = ["app"] + (["--model", "synthetic.pt"] if correction else [])
        fake_model = MagicMock(return_value=torch.tensor([[9., 0., 0., 0., 0., 0.]]))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(app, "DATA", Path(directory)))
            stack.enter_context(patch.object(app.time, "monotonic", side_effect=lambda: next(clock)))
            stack.enter_context(patch.object(cv2, "VideoCapture", return_value=camera))
            stack.enter_context(patch.object(cv2, "imshow"))
            stack.enter_context(patch.object(cv2, "destroyAllWindows"))
            stack.enter_context(patch.object(cv2, "waitKey", side_effect=[255] * 55 + [32, ord("b"), ord("q")]))
            stack.enter_context(patch.object(mp.tasks.vision.HandLandmarker, "create_from_options", return_value=detector))
            stack.enter_context(patch("gesture_commands.model.load_model", return_value=(fake_model, {})))
            self.assertEqual(app.main(), 0)
            saved = list(Path(directory).glob("*.npz"))
            self.assertEqual(len(saved), 1)
            _, times, visible, metadata = load_clip(saved[0])
            self.assertTrue(quality(times, visible))
            self.assertEqual(metadata["label"], "BACKWARD")
            self.assertEqual(metadata["split"], "train")
            self.assertEqual(metadata["prediction"], "FORWARD" if correction else None)
            camera.release.assert_called_once()

    def test_record_freeze_label_save_exit(self):
        self.exercise_capture(correction=False)

    def test_wrong_prediction_freeze_correct_exact_clip(self):
        self.exercise_capture(correction=True)


if __name__ == "__main__":
    unittest.main()
