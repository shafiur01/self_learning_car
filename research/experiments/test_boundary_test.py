"""Synthetic checks for joint final-pilot access and frozen recording policy."""

import contextlib
import io
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np

from . import boundary_test as final
from . import boundary_train as training
from . import commands_v2 as v2
from .run import sha256, write_json
from .test_boundary_train import MatchedTrainingTests
from .test_commands_v2 import raw


class JointTestChecks(unittest.TestCase):
    def setUp(self):
        # Reuse the already-tested tiny training/collection fixture, not real data.
        MatchedTrainingTests.setUp(self)
        MatchedTrainingTests.complete(self)
        with contextlib.redirect_stdout(io.StringIO()):
            self.study = training.run(self.collection,self.root/"joint_runs")
            self.test_root = self.root/"test_pool"
            self.plan = final.freeze(self.test_root,self.study,self.data/"migration.json")

    def add(self, sitting, index, *, session=None, data=None, override=None):
        task = final.schedule(sitting)[index]
        meta = {"split": "test", "source": "recording", "label": task["label"],
                "session": session or f"fresh-{sitting}", "aspect": 1.5, "experiment_id": final.EXPERIMENT,
                "participant": "current_user", "collection_sitting": sitting, "task_index": index,
                "task_hint": task["hint"], "freeze_sha256": sha256(self.test_root/"freeze.json"),
                "created_utc": v2.utc_now(), "model": None, "prediction": None, "scores": None}
        return v2.save_clip(self.test_root/"recordings", *(raw(9000+sitting*100+index) if data is None else data),
                            {**meta, **(override or {})})

    def complete_test(self):
        for sitting in (1,2):
            for index in range(40):
                # Legitimate no-hand example stays in metrics despite quality failure.
                data = raw(9000+sitting*100+index)
                if final.schedule(sitting)[index]["label"] == "NO_COMMAND" and index >= 32:
                    data = (data[0],data[1],np.zeros(60,dtype=bool))
                self.add(sitting,index,data=data)

    def test_schedule_and_prompt_keys_cover_all_labels(self):
        for s in (1,2):
            self.assertEqual(len(final.schedule(s)),40)
            self.assertEqual({l: sum(t["label"]==l for t in final.schedule(s)) for l in v2.LABELS},
                             dict.fromkeys(v2.LABELS,5))
        self.assertEqual(list(final.PromptKeys(lambda: {"label":"UP"})),["u"])
        self.assertEqual(list(final.PromptKeys(lambda: None)),[])
        with self.assertRaisesRegex(ValueError,"already contains"):
            final.freeze(self.test_root,self.study,self.data/"migration.json")

    def test_partial_metadata_only_collection_never_accesses_landmarks(self):
        self.add(1,0)
        with patch.object(v2,"load_clip",side_effect=AssertionError("No test payload")):
            self.assertEqual(final.audit(self.test_root)["sittings"]["1"]["saved"],1)
            with self.assertRaisesRegex(ValueError,"all 80"):
                final.evaluate(self.test_root)
        self.assertFalse((self.test_root/"test_access.json").exists())

    def test_historical_pre_freeze_and_model_assisted_metadata_rejected(self):
        for override, session, pattern in [({},"retired","Historical"),
                ({"created_utc":(datetime.fromisoformat(self.plan["created_utc"])-timedelta(seconds=1)).isoformat()},None,"after freeze"),
                ({"scores":[.125]*8},None,"human-only")]:
            path = self.add(1,0,session=session,override=override)
            with patch.object(v2,"load_clip",side_effect=AssertionError("No test payload")):
                with self.assertRaisesRegex(ValueError,pattern):
                    final.audit(self.test_root)
            path.unlink()  # Only disposable synthetic fixtures.

    def test_frozen_candidate_tampering_detected_without_test_access(self):
        path = self.study/"routine/seed_11/model.pt"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError,"checkpoint changed"):
            final.audit(self.test_root)
        self.assertFalse((self.test_root/"test_access.json").exists())

    def test_duplicate_failure_retires_pool_before_payload_load(self):
        self.complete_test()
        # Make the first new file identical in raw landmarks to development.
        first = final.audit(self.test_root)["records"][0]
        xyz,times,visible,_ = v2.load_clip(self.data/next(iter(training.read_json(self.study/"manifest.json")["rows"]["base_train"])).split("/")[-1])
        path = first["file"]
        with np.load(path,allow_pickle=False) as a:
            meta = str(a["metadata"])
        np.savez_compressed(path,xyz=xyz,times=times,visible=visible,metadata=meta)
        real = v2.load_clip

        def guarded(path):
            self.assertTrue((self.test_root/"test_access.json").exists())
            return real(path)

        with patch.object(v2,"load_clip",side_effect=guarded):
            with self.assertRaisesRegex(ValueError,"duplicates"):
                final.evaluate(self.test_root)
        self.assertEqual(training.read_json(self.test_root/"test_access.json")["accesses"][0]["state"],"failed_inspected")
        with self.assertRaisesRegex(ValueError,"already been accessed"):
            final.evaluate(self.test_root)

    def test_joint_evaluation_all_candidates_quality_retained_and_no_repeat(self):
        self.complete_test()
        self.assertTrue(final.audit(self.test_root)["complete"])
        with contextlib.redirect_stdout(io.StringIO()):
            output = final.evaluate(self.test_root)
        trials = training.read_json(output/"trials.json")
        self.assertEqual(len(trials),3)
        self.assertEqual({r["condition"] for r in trials},set(training.CONDITIONS))
        self.assertTrue(all(r["pooled"]["clips"]==80 for r in trials))
        self.assertTrue(all(r["sittings"]["1"]["clips"]==40 and r["sittings"]["2"]["clips"]==40 for r in trials))
        self.assertEqual(training.read_json(output/"manifest.json")["quality_failures_retained"],2)
        by = {r["condition"]:r for r in trials}
        self.assertAlmostEqual(training.read_json(output/"summary.json")["primary_comparison"]["mean"],
            by["targeted"]["pooled"]["macro_f1"]-by["routine"]["pooled"]["macro_f1"])
        self.assertTrue(final.audit(self.test_root)["inspected"])
        self.assertEqual(training.read_json(self.test_root/"test_access.json")["accesses"][0]["state"],"complete")
        with patch.object(v2,"load_clip",side_effect=AssertionError("No repeated test payload")):
            with self.assertRaisesRegex(ValueError,"already been accessed"):
                final.evaluate(self.test_root)

    def test_guided_recorder_saves_test_without_model_and_restores_overlay(self):
        from gesture_commands import app
        original = app.overlay

        def camera(*,profile,argv):
            self.assertEqual(argv,["--split","test","--camera","0"])
            self.assertEqual(list(profile.keys),["f"])
            profile.save_clip(profile.data,*raw(77777),{
                "label":"FORWARD","split":"test","source":"recording","session":"fresh-camera",
                "created_utc":v2.utc_now(),"aspect":1.5,"model":None,"prediction":None,"scores":None})
            self.assertEqual(list(profile.keys),["b"])
            return 0

        with patch.object(app,"main",side_effect=camera), patch.object(v2,"load_model",side_effect=AssertionError("No model")), contextlib.redirect_stdout(io.StringIO()):
            final.record(1,self.test_root)
        self.assertIs(app.overlay,original)
        self.assertEqual(final.audit(self.test_root)["sittings"]["1"]["saved"],1)
        self.assertEqual(len(list((self.test_root/"recording_attempts").glob("*.json"))),1)
        with patch.object(app,"main",side_effect=RuntimeError("camera error")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,"camera error"):
                final.record(1,self.test_root)
        self.assertIs(app.overlay,original)


if __name__ == "__main__":
    unittest.main()
