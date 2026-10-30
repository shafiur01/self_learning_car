"""Eight-class, migration, recording and evaluation integrity tests.

Synthetic fixtures test software only; they establish no gesture accuracy.
"""

import contextlib
import io
import json
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from gesture_commands.data import save_clip as save_old
from gesture_commands.model import load_model as load_old_model
from .commands_v2 import (LABELS, TRAIN_CONFIG, CommandGate, build_model, content_sha,
                          contract, development_snapshot, evaluate_test, freeze,
                          load_clip, load_model, metrics, prepare, record,
                          save_clip, train)
from .run import sha256, write_json


def raw(seed=1):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(0.4, 0.02, (60, 21, 3)).astype(np.float32)
    xyz[:, 9, 1] = xyz[:, 0, 1] + 0.1
    return xyz, np.linspace(0, 1.98, 60), np.ones(60, dtype=bool)


def dataset(folder):
    folder.mkdir(parents=True)
    write_json(folder / "contract.json", contract())
    write_json(folder / "migration.json", {"imports": [], "historical_sessions": ["retired"],
                                          "historical_development_raw_hashes": []})
    seed = 0
    for split in ("train", "validation"):
        for label in LABELS:
            for _ in range(5):
                seed += 1
                save_clip(folder, *raw(seed), {"label": label, "split": split, "session": split,
                                              "source": "recording", "aspect": 1.5})


class EightCommandTests(unittest.TestCase):
    def test_vertical_landmark_motion_preserved_and_schema_roundtrip(self):
        from gesture_commands.data import features
        xyz, times, visible = raw()
        xyz[:, :, 1] += times[:, None]*0.05
        x = features(xyz, times, visible, 1.5)
        self.assertGreater(x[-1, 64], 0)  # Image y grows downward.
        with tempfile.TemporaryDirectory() as tmp:
            path = save_clip(tmp, xyz, times, visible,
                             {"label": "DOWN", "split": "train", "session": "s", "aspect": 1.5})
            _, _, _, meta = load_clip(path)
            self.assertEqual(meta["label"], "DOWN")
            self.assertEqual(meta["labels"], list(LABELS))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                save_clip(tmp, xyz, times, visible, meta, filename=path.name)

    def test_up_down_gate_and_false_motion_metrics_use_explicit_labels(self):
        gate = CommandGate(repeats=2)
        for label in ("UP", "DOWN"):
            gate.reset()
            p = np.eye(8)[LABELS.index(label)]
            self.assertEqual(gate.update(p, True), (label, "STOP"))
            self.assertEqual(gate.update(p, True), (label, label))
            self.assertEqual(gate.update(p, False), ("NO_HAND", "STOP"))
        for label in ("STOP", "NO_COMMAND"):
            self.assertEqual(gate.update(np.eye(8)[LABELS.index(label)], True), (label, "STOP"))
        r = metrics(np.eye(8)[[6, 7]], np.array([4, 5]))
        self.assertEqual(r["stop_to_motion_count"], 1)
        self.assertEqual(r["no_command_to_motion_count"], 1)
        with self.assertRaises(ValueError):
            gate.update(np.ones(6)/6, True)

    def test_migration_copies_whole_session_retains_origin_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, old_study, evaluation, target = [root / name for name in ("source", "study", "evaluation", "target")]
            old_study.mkdir(); evaluation.mkdir()
            rows = []
            for i, split in enumerate(("train", "validation", "validation")):
                p = save_old(source, *raw(i), {"label": "FORWARD", "split": split,
                             "session": ("old_train", "old_val", "new_val")[i], "source": "recording", "aspect": 1.5})
                from gesture_commands.data import load_clip as old_load
                xyz, times, visible, meta = old_load(p)
                rows.append({"file": str(p), "sha256": sha256(p), "label": meta["label"],
                             "split": split, "session": meta["session"], "source": "recording",
                             "raw_content_sha256": content_sha(xyz, times, visible, 1.5)})
            write_json(old_study / "manifest.json", {"records": rows[:2]})
            write_json(evaluation / "manifest.json", {"records": rows[2:]})
            hashes = [sha256(r["file"]) for r in rows]
            with contextlib.redirect_stdout(io.StringIO()):
                migration = prepare(target, old_study, evaluation)
                again = prepare(target, old_study, evaluation)
            self.assertEqual(migration, again)
            self.assertEqual(len(list(target.glob("*.npz"))), 2)
            self.assertEqual(hashes, [sha256(r["file"]) for r in rows])
            copied = [load_clip(p)[3] for p in target.glob("*.npz")]
            self.assertEqual({r["session"] for r in copied}, {"old_train", "new_val"})
            self.assertTrue(all(r["split"] == "train" for r in copied))
            migrated = next(r for r in copied if r["session"] == "new_val")
            self.assertEqual(migrated["origin_split"], "validation")
            self.assertEqual(migrated["acquisition"], "recycled_inspected_validation")

    def test_development_excludes_test_payload_and_rejects_retired_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "data"
            dataset(folder)
            np.savez_compressed(folder / "test_metadata_only.npz", metadata=json.dumps({"split": "test", "session": "t"}))
            data, manifest = development_snapshot(folder)
            self.assertEqual(len(data["train"]), 40)
            self.assertEqual(len(data["validation"]), 40)
            self.assertFalse(any(r["session"] == "t" for r in manifest["records"]))
            save_clip(folder, *raw(1000), {"label": "UP", "split": "validation", "session": "retired", "aspect": 1.5, "source": "recording"})
            with self.assertRaisesRegex(ValueError, "fresh original"):
                development_snapshot(folder)

    def test_duplicate_contents_and_cross_split_sessions_rejected(self):
        for change in ("duplicate", "session"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp) / "data"
                dataset(folder)
                if change == "duplicate":
                    p = next(p for p in folder.glob("*.npz") if load_clip(p)[3]["split"] == "train")
                    xyz, times, visible, meta = load_clip(p)
                    save_clip(folder, xyz, times, visible, {**meta, "split": "validation", "session": "v2"})
                else:
                    save_clip(folder, *raw(1000), {"label": "UP", "split": "validation", "session": "train", "aspect": 1.5, "source": "recording"})
                with self.assertRaises(ValueError):
                    development_snapshot(folder)

    def test_training_requires_fresh_validation_and_both_new_labels(self):
        for missing in ("UP", "validation"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp) / "data"
                dataset(folder)
                for p in folder.glob("*.npz"):
                    meta = load_clip(p)[3]
                    if meta["label"] == missing or meta["split"] == missing:
                        p.unlink()
                with self.assertRaisesRegex(ValueError, missing):
                    train(folder, Path(tmp) / "runs")
                self.assertFalse((Path(tmp) / "runs").exists())

    def test_training_checkpoint_freeze_and_single_use_fresh_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "data"
            dataset(folder)
            config = {**TRAIN_CONFIG, "seeds": [11], "optimizer_steps": 3, "validation_every_steps": 2}
            with contextlib.redirect_stdout(io.StringIO()):
                output = train(folder, Path(tmp) / "runs", config)
                freeze(output)
            model, checkpoint = load_model(output / "seed_11/model.pt")
            self.assertEqual(len(checkpoint["labels"]), 8)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                load_old_model(output / "seed_11/model.pt")
            with torch.inference_mode():
                self.assertEqual(tuple(model(torch.zeros(1, 32, 67)).shape), (1, 8))
                with np.load(output / "development_features.npz", allow_pickle=False) as a:
                    scores = model(torch.from_numpy(a["validation_x"].copy())).softmax(1).numpy()
            saved = json.loads((output / "seed_11/validation_predictions.json").read_text())
            np.testing.assert_allclose(scores, [r["scores"] for r in saved], rtol=1e-6, atol=1e-7)
            self.assertEqual(json.loads((output / "seed_11/history.json").read_text())[-1]["step"], 3)
            after = (datetime.now(timezone.utc)+timedelta(seconds=1)).isoformat()
            for i, label in enumerate(LABELS):
                for j in range(5):
                    save_clip(folder, *raw(1000+i*5+j), {"label": label, "split": "test", "session": "new_test",
                              "source": "recording", "aspect": 1.5, "created_utc": after})
            with contextlib.redirect_stdout(io.StringIO()):
                final = evaluate_test(output, folder)
            self.assertEqual(json.loads((final / "summary.json").read_text())["trials"][0]["clips"], 40)
            with self.assertRaisesRegex(ValueError, "already inspected"):
                evaluate_test(output, folder)
            with self.assertRaisesRegex(ValueError, "already frozen"):
                freeze(output)

    def test_test_collection_before_freeze_rejected_before_payload_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "data"
            dataset(folder)
            config = {**TRAIN_CONFIG, "seeds": [11], "optimizer_steps": 1, "validation_every_steps": 1}
            with contextlib.redirect_stdout(io.StringIO()):
                output = train(folder, Path(tmp) / "runs", config)
                freeze(output)
            # No coordinate arrays: rejection must happen at metadata preflight.
            np.savez_compressed(folder / "old_test.npz", metadata=json.dumps({"split": "test", "session": "test",
                                "source": "recording", "created_utc": "2026-01-01T00:00:00+00:00"}))
            with self.assertRaisesRegex(ValueError, "after the study freeze"):
                evaluate_test(output, folder)
            self.assertFalse((folder / "test_access.json").exists())

    def test_camera_adapter_records_up_without_changing_legacy_labels(self):
        import cv2
        import mediapipe as mp
        from gesture_commands import app
        from gesture_commands.data import LABELS as original_labels
        xyz, _, _ = raw()
        landmarks = [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in xyz[0]]
        result = SimpleNamespace(hand_landmarks=[landmarks], handedness=[[SimpleNamespace(category_name="Right")]])
        detector = MagicMock()
        detector.__enter__.return_value = detector
        detector.detect_for_video.return_value = result
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = lambda: (True, np.zeros((360, 640, 3), dtype=np.uint8))
        clock = iter(np.arange(100, 200, 0.04))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            folder = Path(tmp)
            write_json(folder / "contract.json", contract())
            stack.enter_context(patch.object(app.time, "monotonic", side_effect=lambda: next(clock)))
            stack.enter_context(patch.object(cv2, "VideoCapture", return_value=camera))
            stack.enter_context(patch.object(cv2, "imshow"))
            stack.enter_context(patch.object(cv2, "destroyAllWindows"))
            stack.enter_context(patch.object(cv2, "waitKey", side_effect=[255]*55+[32, ord("u"), ord("q")]))
            stack.enter_context(patch.object(mp.tasks.vision.HandLandmarker, "create_from_options", return_value=detector))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(record(folder, ["--split", "train"], "right", "point upward"), 0)
            p = next(folder.glob("*.npz"))
            self.assertEqual(load_clip(p)[3]["label"], "UP")
            self.assertEqual(load_clip(p)[3]["hand_reported"], "right")
            self.assertEqual(len(original_labels), 6)
            camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
