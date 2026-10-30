"""Prospective collection checks use temporary synthetic records; no camera."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import prospective_motion as p
from .test_commands_v2 import raw


class ProspectiveChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/"freeze.json").write_text("synthetic freeze")
        contract=p.validation.v3.contract("three_shapes")
        self.plan={"contract":contract,"schedules":{str(s):p.schedule(contract,s) for s in (1,2)},
            "banned_sessions":["historical"],"banned_raw_contents":[]}
        self.mock=patch.object(p,"read_plan",return_value=self.plan);self.mock.start();self.addCleanup(self.mock.stop)

    def add(self,sitting,index,*,override=None,seed=None):
        xyz,times,visible=raw(50000+sitting*100+index if seed is None else seed)
        task=self.plan["schedules"][str(sitting)][index]
        identity=p.validation.v2.content_sha(xyz,times,visible,1.5)
        return p.validation.v3.save_clip(self.root/"recordings",xyz,times,visible,{
            "split":"validation","label":task["label"],"session":f"new-sitting-{sitting}","aspect":1.5,"source":"recording",
            "experiment_id":p.EXPERIMENT,"prospective_sitting":sitting,"task_index":index,"up_shape":task["up_shape"],
            "task_hint":task["hint"],"freeze_sha256":p.sha256(self.root/"freeze.json"),"raw_content_sha256":identity,
            "passes_tracking_quality":True,**(override or {})},self.plan["contract"])

    def test_schedules_are_balanced_and_stop_open_palm(self):
        for s in (1,2):
            tasks=self.plan["schedules"][str(s)]
            self.assertEqual(len(tasks),48)
            for label in p.validation.v2.LABELS:self.assertEqual(sum(t["label"]==label for t in tasks),6)
            self.assertTrue(all("open palm" in t["hint"] or "open-palm" in t["hint"] for t in tasks if t["label"]=="STOP"))
        self.assertEqual([t["label"] for t in self.plan["schedules"]["2"][:8]],list(reversed(p.validation.v2.LABELS)))

    def test_incomplete_pool_rejected_before_payload_and_access_ledger(self):
        self.add(1,0)
        with patch.object(p.experiment.motion.previous.training,"load_clip",side_effect=AssertionError("No held-out decoding")):
            with self.assertRaisesRegex(ValueError,"both48"):
                p.evaluate(self.root)
        self.assertFalse((self.root/"evaluation_access.json").exists())

    def test_metadata_refuses_foreign_roles_duplicates_and_sessions(self):
        (self.root/"recordings").mkdir()
        path=self.root/"recordings/foreign.npz"
        np.savez_compressed(path,metadata=json.dumps({"prospective_sitting":1,"task_index":0,"split":"test"}))
        with self.assertRaisesRegex(ValueError,"no payload accessed"):p.status(self.root)
        path.unlink()
        path=self.add(1,0,override={"session":"historical"})
        with self.assertRaisesRegex(ValueError,"reuse"):p.status(self.root)
        path.unlink()
        self.add(1,0);self.add(1,0,seed=60000)
        with self.assertRaisesRegex(ValueError,"Duplicate"):p.status(self.root)

    def test_saved_prefix_resume_and_overlay_restoration(self):
        self.add(1,0);self.add(1,1)
        from gesture_commands import app
        original=app.overlay
        def fake_main(profile,argv):
            self.assertEqual(list(profile.keys),[p.KEY[self.plan["schedules"]["1"][2]["label"]]])
            xyz,times,visible=raw(60200)
            profile.save_clip(profile.data,xyz,times,visible,{"label":self.plan["schedules"]["1"][2]["label"],
                "split":"validation","source":"recording","session":"restart-same-physical-sitting","aspect":1.5})
            return 0
        with patch.object(app,"main",side_effect=fake_main),contextlib.redirect_stdout(io.StringIO()):
            p.record(1,self.root)
        self.assertIs(app.overlay,original);self.assertEqual(p.status(self.root)["counts"]["1"],3)
        with patch.object(app,"main",side_effect=RuntimeError("camera stopped")),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,"camera stopped"):p.record(1,self.root)
        self.assertIs(app.overlay,original)

    def test_second_sitting_cannot_start_before_first_complete(self):
        with self.assertRaisesRegex(ValueError,"Complete sitting1"):p.record(2,self.root)

    def test_complete_joint_evaluation_retires_pool_and_detects_saved_score_tampering(self):
        for sitting in (1,2):
            for index in range(48):self.add(sitting,index)
        models=self.root/"models";folder=models/"synthetic_feature_set";folder.mkdir(parents=True)
        torch.manual_seed(3);model=p.experiment.build_model("feature_set")
        checkpoint={"state_dict":model.state_dict(),"experiment_id":"011_cross_model_motion","schema_version":3,
            "feature_version":2,"features":146,"frames":32,"labels":list(p.validation.v2.LABELS),
            "command_contract":self.plan["contract"],"family":"feature_set","mode":"camera_motion","variant":"standard"}
        torch.save(checkpoint,folder/"model.pt")
        self.plan["study"]=str(models)
        self.plan["checkpoints"]=[{"folder":folder.name,"family":"feature_set","mode":"camera_motion","variant":"standard",
            "seed":11,"stage":"core","sha256":p.sha256(folder/"model.pt")}]
        with contextlib.redirect_stdout(io.StringIO()):p.evaluate(self.root)
        self.assertEqual(p.verify_evaluation(self.root)["prediction_vectors_verified"],96)
        self.assertEqual(json.loads((self.root/"evaluation_access.json").read_text())["state"],"complete")
        with self.assertRaisesRegex(ValueError,"already accessed"):p.evaluate(self.root)
        prediction=self.root/"evaluation"/folder.name/"predictions.json"
        rows=json.loads(prediction.read_text());rows[0]["scores"]=[0.0]*8;p.write_json(prediction,rows)
        with self.assertRaises(AssertionError):p.verify_evaluation(self.root)


if __name__=="__main__":unittest.main()
