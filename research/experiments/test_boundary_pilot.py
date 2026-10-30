"""Synthetic collection/leakage checks, not evidence of gesture accuracy."""

import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import boundary_pilot as pilot
from . import commands_v2 as v2
from .run import sha256, write_json
from .test_commands_v2 import dataset, raw


class BoundaryPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        data = self.folder / "base_data"
        dataset(data)
        with contextlib.redirect_stdout(io.StringIO()):
            self.study = v2.train(data, self.folder / "runs",
                                  {**v2.TRAIN_CONFIG, "seeds": [11], "optimizer_steps": 1,
                                   "validation_every_steps": 1})
        write_json(self.study / "verification.json",
                   {"checkpoints_sha256": {"seed_11/model.pt": sha256(self.study / "seed_11/model.pt")}})
        self.evaluation = self.folder / "evaluation"
        self.evaluation.mkdir()
        new = []
        for i, label in enumerate(v2.LABELS):
            p = v2.save_clip(data, *raw(500+i), {"label": label, "split": "validation",
                                "session": "new-val", "source": "recording", "aspect": 1.5})
            xyz, times, visible, _ = v2.load_clip(p)
            new.append({"file": str(p), "sha256": sha256(p), "split": "validation", "label": label,
                        "session": "new-val", "raw_content_sha256": v2.content_sha(xyz, times, visible, 1.5)})
        write_json(self.evaluation / "manifest.json", {"contract": v2.contract(), "records": new})
        write_json(self.evaluation / "status.json", {"state": "complete"})
        write_json(self.evaluation / "environment.json",
                   {"artifact_sha256": {"manifest.json": sha256(self.evaluation / "manifest.json")}})
        self.root = self.folder / "collection"
        self.plan = pilot.prepare(self.root, self.study, self.evaluation)

    def add_task(self, sitting, index, seed=None, session=None, **overrides):
        task = pilot.schedule(sitting)[index]
        meta = {"experiment_id": pilot.EXPERIMENT, "collection_sitting": sitting,
                "task_index": index, "collection_group": task["group"], "label": task["label"],
                "task_hint": task["hint"], "collection_plan_sha256": sha256(self.root / "plan.json"),
                "split": "train", "source": "recording", "session": session or f"new-train-{sitting}",
                "model": None, "prediction": None, "scores": None, "aspect": 1.5}
        return v2.save_clip(self.root / "recordings", *raw(seed or 1000+sitting*100+index), {**meta, **overrides})

    def test_class_matched_counterbalanced_schedule(self):
        for sitting in (1, 2):
            tasks = pilot.schedule(sitting)
            self.assertEqual(len(tasks), 40)
            self.assertEqual(Counter((t["group"], t["label"]) for t in tasks),
                             Counter({(g, label): 5 for g in pilot.GROUPS for label in pilot.FOCUS}))
        self.assertEqual(pilot.schedule(1)[0]["group"], "routine")
        self.assertEqual(pilot.schedule(2)[0]["group"], "targeted")
        with self.assertRaises(ValueError):
            pilot.schedule(True)

    def test_prepare_is_idempotent_keeps_validation_and_detects_changes(self):
        plan_hash = sha256(self.root / "plan.json")
        self.assertEqual(pilot.prepare(self.root, self.study, self.evaluation), self.plan)
        self.assertEqual(plan_hash, sha256(self.root / "plan.json"))
        self.assertEqual(len(self.plan["selection_validation_files"]), 40)
        self.assertEqual(len(self.plan["monitor_validation_files"]), 8)
        self.assertFalse(list((self.root / "recordings").glob("*.npz")))
        record = self.plan["base_records"][0]
        Path(record["file"]).write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "Frozen development input changed"):
            pilot.read_plan(self.root)

    def test_plan_and_reference_weights_are_protected(self):
        (self.root / "plan.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Collection plan changed"):
            pilot.read_plan(self.root)
        write_json(self.root / "plan.json", self.plan)
        (self.root / "plan.sha256").write_text(sha256(self.root / "plan.json"))
        (self.study / "seed_11/model.pt").write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "verification hash"):
            pilot.read_plan(self.root)

    def test_audit_rejects_wrong_role_group_model_or_old_session(self):
        for overrides in ({"split": "validation"}, {"source": "correction"},
                          {"collection_group": "targeted"}, {"model": "model.pt"},
                          {"prediction": "FORWARD"}, {"session": "train"}):
            with self.subTest(overrides=overrides):
                p = self.add_task(1, 0, **overrides)
                with self.assertRaises(ValueError):
                    pilot.audit(self.root)
                p.unlink()

    def test_audit_rejects_skipped_duplicate_or_cross_sitting_slots(self):
        p = self.add_task(1, 1)
        with self.assertRaisesRegex(ValueError, "completed prefix"):
            pilot.audit(self.root)
        p.unlink()
        self.add_task(1, 0)
        p = self.add_task(1, 0, seed=2000)
        with self.assertRaisesRegex(ValueError, "same task slot"):
            pilot.audit(self.root)
        p.unlink()
        p = self.add_task(1, 1, seed=1100)
        with self.assertRaisesRegex(ValueError, "Duplicate raw"):
            pilot.audit(self.root)
        p.unlink()
        self.add_task(2, 0, session="new-train-1")
        with self.assertRaisesRegex(ValueError, "crosses collection sittings"):
            pilot.audit(self.root)

    def test_audit_retains_tracking_failure_and_does_not_read_test_payload(self):
        for i in range(10):
            self.add_task(1, i)
        task = pilot.schedule(1)[10]  # NO_COMMAND
        xyz, times, visible = raw(3000)
        visible[:] = False
        v2.save_clip(self.root / "recordings", xyz, times, visible,
                     {"experiment_id": pilot.EXPERIMENT, "collection_sitting": 1, "task_index": 10,
                      "collection_group": task["group"], "task_hint": task["hint"],
                      "collection_plan_sha256": sha256(self.root / "plan.json"),
                      "split": "train", "source": "recording", "session": "new-train-1",
                      "label": task["label"], "aspect": 1.5})
        # The collector audits only its own isolated directory and frozen references.
        np.savez_compressed(self.folder / "test.npz", metadata="must not be read")
        state = pilot.audit(self.root)
        self.assertEqual(state["sittings"]["1"]["saved"], 11)
        self.assertEqual(len(state["tracking_quality_failures"]), 1)
        self.assertFalse(state["collection_complete"])

    def test_guided_recording_updates_prompt_and_resumes_without_loading_model(self):
        from gesture_commands import app
        original_overlay = app.overlay

        def simulated_camera(profile, argv):
            self.assertNotIn("--model", argv)
            self.assertEqual(argv[:2], ["--split", "train"])
            for i in range(6):
                expected = "FORWARD" if i < 5 else "BACKWARD"
                self.assertEqual(list(profile.keys), [pilot.KEY[expected]])
                profile.save_clip(profile.data, *raw(4000+i), {**profile.metadata,
                    "label": expected, "split": "train", "source": "recording", "session": "capture-1",
                    "model": None, "scores": None, "prediction": None, "aspect": 1.5})
            return 0

        with patch.object(app, "main", side_effect=simulated_camera), patch.object(v2, "load_model", side_effect=AssertionError("No model")), contextlib.redirect_stdout(io.StringIO()):
            pilot.record(1, self.root)
        self.assertIs(app.overlay, original_overlay)
        self.assertEqual(pilot.audit(self.root)["sittings"]["1"]["saved"], 6)
        with patch.object(app, "main") as camera:
            with self.assertRaisesRegex(ValueError, "Complete sitting 1"):
                pilot.record(2, self.root)
            camera.assert_not_called()

        def resumed(profile, argv):
            self.assertEqual(list(profile.keys), ["b"])
            return 0

        with patch.object(app, "main", side_effect=resumed), contextlib.redirect_stdout(io.StringIO()):
            pilot.record(1, self.root)
        self.assertEqual(len(list((self.root / "recording_attempts").glob("*.json"))), 2)

    def test_complete_collection_balanced_and_no_camera_reopens(self):
        for sitting in (1, 2):
            for index in range(40):
                self.add_task(sitting, index)
        state = pilot.audit(self.root)
        self.assertTrue(state["collection_complete"])
        self.assertEqual(state["groups"], {g: {label: 10 for label in pilot.FOCUS} for g in pilot.GROUPS})
        with patch("gesture_commands.app.main") as camera, contextlib.redirect_stdout(io.StringIO()):
            pilot.record(1, self.root)
            pilot.record(2, self.root)
            camera.assert_not_called()

    def test_overlay_restored_and_attempt_retained_after_camera_failure(self):
        from gesture_commands import app
        original = app.overlay
        with patch.object(app, "main", side_effect=RuntimeError("camera failed")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "camera failed"):
                pilot.record(1, self.root)
        self.assertIs(app.overlay, original)
        attempt = json.loads(next((self.root / "recording_attempts").glob("*.json")).read_text())
        self.assertEqual(attempt["state"], "failed")
        self.assertEqual(attempt["saved_after"], 0)


if __name__ == "__main__":
    unittest.main()
