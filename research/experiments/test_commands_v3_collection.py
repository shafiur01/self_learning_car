"""Synthetic checks of semantic replacement, cohort exclusions and resumption."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from . import commands_v3_collection as v3
from . import commands_v2 as v2
from . import boundary_train as training
from .run import sha256
from .test_boundary_train import MatchedTrainingTests
from .test_commands_v2 import raw


class ReplacementChecks(unittest.TestCase):
    def setUp(self):
        MatchedTrainingTests.setUp(self);MatchedTrainingTests.complete(self)
        with contextlib.redirect_stdout(io.StringIO()):
            self.study=training.run(self.collection,self.root/"source_runs")
        self.new=self.root/"v3"

    def prepare(self,stop="open_palm",style="index_up",notes=None):
        with contextlib.redirect_stdout(io.StringIO()):
            self.plan=v3.prepare(style,stop,self.new,self.study,history_ledger=None,stop_history_notes=notes)
        return self.plan

    def add(self,sitting,index,override=None,data=None):
        meta={"label":"UP","split":"train","source":"recording","session":f"replacement-{sitting}",
            "aspect":1.5,"created_utc":v2.utc_now(),"experiment_id":v3.EXPERIMENT,
            "collection_sitting":sitting,"task_index":index,"task_hint":self.plan["tasks"][str(sitting)][index]["hint"],
            "up_shape":self.plan["tasks"][str(sitting)][index]["up_shape"],
            "plan_sha256":sha256(self.new/"plan.json"),"model":None,"prediction":None,"scores":None}
        return v3.save_clip(self.new/"new_training",*(raw(17000+sitting*100+index) if data is None else data),
                            {**meta,**(override or {})},self.plan["contract"])

    def test_old_up_excluded_originals_preserved_and_schema_incompatible(self):
        manifest=training.read_json(self.study/"manifest.json")
        before={r["file"]:sha256(r["file"]) for r in manifest["records"]}
        plan=self.prepare()
        self.assertEqual(len(plan["imports"]),75)
        self.assertTrue(all(r["label"]!="UP" for r in plan["imports"]))
        self.assertEqual(len(plan["excluded_training"]),5)
        self.assertEqual({r["file"]:sha256(r["file"]) for r in manifest["records"]},before)
        imported=plan["imports"][0]
        with self.assertRaisesRegex(ValueError,"incompatible v2"):
            v2.load_clip(imported["file"])
        source=next(r for r in manifest["records"] if r["file"]==imported["origin_file"])
        original=v2.load_clip(source["file"])
        with np.load(imported["file"],allow_pickle=False) as a:
            np.testing.assert_array_equal(a["xyz"],original[0])
        self.assertEqual(plan["contract"]["meanings"]["STOP"],"Hold a still open palm facing the camera, fingers extended; no closed fist.")
        self.assertEqual(v2.MEANINGS["STOP"],"Hold a closed fist still, distinct from beckoning.")
        with self.assertRaisesRegex(ValueError,"already contains"):
            self.prepare()

    def test_uncertain_stop_excluded_not_reclassified(self):
        plan=self.prepare("mixed")
        self.assertEqual(len(plan["imports"]),70)
        self.assertFalse(any(r["label"] in ("UP","STOP") for r in plan["imports"]))
        self.assertFalse(plan["stop_ready"])
        self.assertEqual(len(plan["excluded_training"]),10)
        self.assertFalse(v3.audit(self.new)["training_ready"])

    def test_three_up_shapes_balanced_across_sittings_and_stop_report_kept(self):
        from collections import Counter
        quote="yes almost all were open palm, no fists"
        plan=self.prepare(style="three_shapes",notes=quote)
        self.assertEqual(plan["contract"]["allowed_up_shapes"],["index_up","two_fingers_up","thumbs_up"])
        self.assertEqual(Counter(t["up_shape"] for tasks in plan["tasks"].values() for t in tasks),
                         Counter({"index_up":17,"two_fingers_up":17,"thumbs_up":16}))
        self.assertEqual(plan["stop_history_notes"],quote)
        self.assertFalse(plan["stop_independently_reviewed"])
        row=next(r for r in plan["imports"] if r["label"]=="STOP")
        with np.load(row["file"],allow_pickle=False) as a:
            self.assertEqual(json.loads(str(a["metadata"]))["stop_semantics_basis"],quote)
        for sitting in (1,2):
            for i in range(25):self.add(sitting,i)
        state=v3.audit(self.new)
        self.assertTrue(state["up_collection_complete"])
        self.assertEqual(state["sittings"]["1"]["prompted_shape_counts"],{"index_up":9,"two_fingers_up":8,"thumbs_up":8})
        self.assertEqual(state["sittings"]["2"]["prompted_shape_counts"],{"two_fingers_up":9,"thumbs_up":8,"index_up":8})

    def test_no_source_validation_imported_and_new50_are_training_only(self):
        self.prepare()
        manifest=training.read_json(self.study/"manifest.json")
        evaluation=set(manifest["rows"]["selection"]+manifest["rows"]["monitor"])
        self.assertFalse(evaluation&{r["origin_file"] for r in self.plan["imports"]})
        for sitting in (1,2):
            for i in range(25):self.add(sitting,i)
        state=v3.audit(self.new)
        self.assertEqual(state["new_up_total"],50)
        self.assertEqual(state["old_up_in_active_training"],0)
        self.assertTrue(state["up_collection_complete"])
        self.assertFalse(state["training_ready"])
        with patch("gesture_commands.app.main",side_effect=AssertionError("Must not reopen complete sitting")),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(v3.record(1,self.new),0)

    def test_foreign_evaluation_metadata_refused_without_payload(self):
        self.prepare();(self.new/"new_training").mkdir()
        np.savez_compressed(self.new/"new_training"/"foreign.npz",metadata=json.dumps({
            "split":"test","collection_sitting":1,"task_index":0}))
        with self.assertRaisesRegex(ValueError,"no payload accessed"):
            v3.audit(self.new)

    def test_duplicate_or_incomplete_tracking_rejected(self):
        self.prepare();self.add(1,0)
        p=self.add(1,1,data=raw(17100))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            v3.audit(self.new)
        p.unlink()  # Disposable synthetic recording only.
        xyz,times,visible=raw(18111)
        self.add(1,1,data=(xyz,times,np.zeros_like(visible)))
        with self.assertRaisesRegex(ValueError,"insufficient tracking"):
            v3.audit(self.new)

    def test_resume_at_next_prompt_no_model_overlay_restored(self):
        self.prepare();self.add(1,0)
        from gesture_commands import app
        original=app.overlay

        def camera(*,profile,argv):
            self.assertEqual(argv,["--split","train","--camera","0"])
            self.assertEqual(list(profile.keys),["u"])
            self.assertEqual(profile.metadata["acquisition"],"v3_replacement_up")
            with self.assertRaisesRegex(ValueError,"No compatible v3"):
                profile.load_model("old.pt")
            profile.save_clip(profile.data,*raw(19099),{"label":"UP","split":"train","source":"recording",
                "session":"replacement-resume","created_utc":v2.utc_now(),"aspect":1.5,
                "model":None,"prediction":None,"scores":None})
            return 0

        with patch.object(app,"main",side_effect=camera),contextlib.redirect_stdout(io.StringIO()):
            v3.record(1,self.new)
        self.assertIs(app.overlay,original)
        self.assertEqual(v3.audit(self.new)["new_up_total"],2)
        rows=list((self.new/"recording_attempts").glob("*.json"))
        self.assertEqual(len(rows),1)
        self.assertEqual(training.read_json(rows[0])["saved_before"],1)
        self.assertEqual(training.read_json(rows[0])["saved_after"],2)
        with patch.object(app,"main",side_effect=RuntimeError("camera failed")),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,"camera failed"):
                v3.record(1,self.new)
        self.assertIs(app.overlay,original)


if __name__=="__main__":
    unittest.main()
