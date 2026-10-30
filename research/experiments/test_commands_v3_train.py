"""Tiny synthetic fits check split isolation, contracts and checkpoint artifacts."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import commands_v3_train as training
from . import commands_v2 as v2
from .run import sha256,write_json
from .test_commands_v3_validation import ValidationChecks
from .test_commands_v2 import raw


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        ValidationChecks.setUp(self)
        self.config={**v2.TRAIN_CONFIG,"seeds":[11,22],"optimizer_steps":2,"validation_every_steps":1}

    def complete(self):
        for i in range(48):
            data=raw(27000+i)
            if self.validation_plan["tasks"][i]["label"]=="NO_COMMAND" and i//8==5:
                data=(data[0],data[1],np.zeros(60,dtype=bool))
            ValidationChecks.add(self,i,data=data)

    def test_incomplete_validation_refuses_run_before_allocation(self):
        with self.assertRaisesRegex(ValueError,"all48"):
            training.run(self.new,self.root/"v3_runs",self.config)
        self.assertFalse((self.root/"v3_runs").exists())

    def test_foreign_test_refused_without_fitter_or_payload(self):
        (self.new/"validation").mkdir()
        np.savez_compressed(self.new/"validation"/"foreign.npz",metadata=json.dumps({"split":"test","task_index":0}))
        with patch.object(training,"load_clip",side_effect=AssertionError("Foreign payload")),patch.object(training.fitting,"fit_candidate",side_effect=AssertionError("No fitting")):
            with self.assertRaisesRegex(ValueError,"no payload accessed"):
                training.run(self.new,self.root/"v3_runs",self.config)
        self.assertFalse((self.root/"v3_runs").exists())

    def test_snapshot_roles_and_no_hand_retention(self):
        self.complete();_,manifest,batches,arrays=training.snapshot(self.new)
        self.assertEqual(len(batches["train"]),125);self.assertEqual(len(batches["validation"]),48)
        self.assertEqual(arrays["train_x"].shape,(125,32,67))
        self.assertFalse(set(manifest["rows"]["train"])&set(manifest["rows"]["validation"]))
        self.assertEqual(manifest["quality_failures_retained"]["validation"],1)
        self.assertEqual(manifest["counts"]["train"]["UP"],50)
        self.assertEqual(manifest["counts"]["validation"],dict.fromkeys(v2.LABELS,6))

    def test_complete_study_reloads_new_contract_and_reproduces_saved_results(self):
        self.complete()
        before={r["file"]:sha256(r["file"]) for r in self.validation_plan["training_records"]}
        with contextlib.redirect_stdout(io.StringIO()):output=training.run(self.new,self.root/"v3_runs",self.config)
        self.assertEqual(training.fitting.read_json(output/"status.json"),{"state":"complete","completed_trials":2})
        verified=training.verify(output);self.assertEqual(verified["saved_prediction_vectors_verified"],96)
        self.assertEqual(verified["input_files_verified"],173)
        self.assertEqual(before,{path:sha256(path) for path in before})
        for seed in (11,22):
            model,checkpoint=training.load_model(output/f"seed_{seed}/model.pt")
            self.assertEqual(checkpoint["command_contract"],self.validation_plan["contract"])
            self.assertEqual(len(checkpoint["training_files"]),125)
            self.assertEqual(len(checkpoint["validation_files"]),48)
            self.assertFalse(set(checkpoint["training_files"])&set(checkpoint["validation_files"]))
            with self.assertRaisesRegex(ValueError,"v2 model"):v2.load_model(output/f"seed_{seed}/model.pt")
            self.assertEqual(tuple(model(torch.zeros(1,32,67)).shape),(1,8))
        _,checkpoint=training.load_model(output/"seed_11/model.pt")
        checkpoint["command_contract"]["meanings"]["STOP"]="Closed fist"
        torch.save(checkpoint,self.root/"wrong_contract.pt")
        with self.assertRaisesRegex(ValueError,"v3 meanings"):training.load_model(self.root/"wrong_contract.pt")
        p=output/"seed_11/validation_predictions.json"
        rows=training.fitting.read_json(p);rows[0]["scores"]=[.125]*8;write_json(p,rows)
        with self.assertRaises(AssertionError):training.verify(output)


if __name__=="__main__":unittest.main()
