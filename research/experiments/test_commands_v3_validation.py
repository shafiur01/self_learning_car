"""Synthetic validation-role, contract, resume and no-hand retention checks."""

import contextlib
import io
import json
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from . import commands_v3_validation as validation
from . import commands_v3_collection as v3
from . import commands_v2 as v2
from .run import sha256
from .test_commands_v3_collection import ReplacementChecks
from .test_commands_v2 import raw


class ValidationChecks(unittest.TestCase):
    def setUp(self):
        ReplacementChecks.setUp(self)
        ReplacementChecks.prepare(self,style="three_shapes")
        for s in (1,2):
            for i in range(25):ReplacementChecks.add(self,s,i)
        with contextlib.redirect_stdout(io.StringIO()):
            self.validation_plan=validation.prepare(self.new,legacy_migration=self.data/"migration.json")

    def add(self,index,*,override=None,data=None):
        task=self.validation_plan["tasks"][index]
        return v3.save_clip(self.new/"validation",*(raw(27000+index) if data is None else data),{
            "label":task["label"],"split":"validation","source":"recording","session":"fresh-validation",
            "aspect":1.5,"created_utc":v2.utc_now(),"schema_version":3,"experiment_id":validation.EXPERIMENT,
            "collection_sitting":1,"task_index":index,"up_shape":task["up_shape"],"task_hint":task["hint"],
            "validation_plan_sha256":sha256(self.new/"validation_plan.json"),"model":None,"prediction":None,"scores":None,
            **(override or {})},self.validation_plan["contract"])

    def test_snapshot_schedule_contract_and_roles(self):
        self.assertEqual(len(self.validation_plan["training_records"]),125)
        self.assertEqual(Counter(t["label"] for t in self.validation_plan["tasks"]),dict.fromkeys(v2.LABELS,6))
        self.assertEqual(Counter(t["up_shape"] for t in self.validation_plan["tasks"] if t["up_shape"]),
                         {"index_up":2,"two_fingers_up":2,"thumbs_up":2})
        self.assertTrue(all("open palm" in t["hint"] or "open-palm" in t["hint"] for t in self.validation_plan["tasks"] if t["label"]=="STOP"))
        self.assertEqual(validation.audit(self.new)["saved"],0)
        self.assertEqual(v3.audit(self.new)["new_up_total"],50)
        with self.assertRaisesRegex(ValueError,"already exists"):
            validation.prepare(self.new,legacy_migration=None)

    def test_incomplete_up_collection_refuses_validation(self):
        # Only this temporary fixture is changed; no real recordings touched.
        p=next((self.new/"new_training").glob("*.npz"));p.unlink()
        with self.assertRaises(ValueError):
            validation.training_snapshot(self.new)

    def test_foreign_test_metadata_refused_before_coordinates(self):
        (self.new/"validation").mkdir()
        np.savez_compressed(self.new/"validation"/"foreign.npz",metadata=json.dumps({"task_index":0,"split":"test"}))
        with patch.object(v2,"validate_clip",side_effect=AssertionError("Foreign payload must not be read")):
            with self.assertRaisesRegex(ValueError,"no payload accessed"):
                validation.audit(self.new)

    def test_training_session_duplicate_and_model_assisted_data_refused(self):
        for override,pattern in [({"session":self.validation_plan["training_records"][0]["session"]},"overlaps"),
                                 ({"scores":[.125]*8},"human-only")]:
            p=self.add(0,override=override)
            with self.assertRaisesRegex(ValueError,pattern):validation.audit(self.new)
            p.unlink()
        r=self.validation_plan["training_records"][0]
        with np.load(r["file"],allow_pickle=False) as a:data=(a["xyz"].copy(),a["times"].copy(),a["visible"].copy())
        self.add(0,data=data)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            validation.audit(self.new)

    def test_complete_validation_keeps_no_hand_and_does_not_change_training(self):
        before={r["file"]:sha256(r["file"]) for r in self.validation_plan["training_records"]}
        for i in range(48):
            data=raw(27000+i)
            if self.validation_plan["tasks"][i]["label"]=="NO_COMMAND" and i//8==5:
                data=(data[0],data[1],np.zeros(60,dtype=bool))
            self.add(i,data=data)
        state=validation.audit(self.new)
        self.assertTrue(state["complete"])
        self.assertEqual(state["labels"],dict.fromkeys(v2.LABELS,6))
        self.assertEqual(state["quality_failures_retained"],1)
        self.assertEqual({r["file"]:sha256(r["file"]) for r in self.validation_plan["training_records"]},before)
        self.assertFalse(set(before)&{r["file"] for r in state["records"]})
        with patch("gesture_commands.app.main",side_effect=AssertionError("Complete collection must not reopen camera")),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(validation.record(self.new),0)

    def test_resume_prompt_keys_no_predictions_and_overlay_restoration(self):
        self.add(0)
        from gesture_commands import app
        original=app.overlay

        def camera(*,profile,argv):
            self.assertEqual(argv,["--split","validation","--camera","0"])
            self.assertEqual(list(profile.keys),["b"])
            with self.assertRaisesRegex(ValueError,"No compatible v3"):
                profile.load_model("old.pt")
            profile.save_clip(profile.data,*raw(29111),{"label":"BACKWARD","split":"validation",
                "session":"resumed-validation","source":"recording","created_utc":v2.utc_now(),"aspect":1.5,
                "model":None,"prediction":None,"scores":None})
            self.assertEqual(list(profile.keys),["l"])
            return 0

        with patch.object(app,"main",side_effect=camera),contextlib.redirect_stdout(io.StringIO()):validation.record(self.new)
        self.assertIs(app.overlay,original)
        self.assertEqual(validation.audit(self.new)["saved"],2)
        with patch.object(app,"main",side_effect=RuntimeError("camera failed")),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,"camera failed"):validation.record(self.new)
        self.assertIs(app.overlay,original)


if __name__=="__main__":unittest.main()
