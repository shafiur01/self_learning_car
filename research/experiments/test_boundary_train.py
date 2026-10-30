"""Matched training, data-role and artifact checks with synthetic landmarks."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import boundary_pilot as pilot
from . import boundary_train as study
from . import commands_v2 as v2
from .evaluate_v2_sessions import run as monitor_run
from .run import sha256, write_json
from .test_commands_v2 import dataset, raw


class MatchedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "old_data"
        dataset(self.data)
        self.config = {**v2.TRAIN_CONFIG, "seeds": [11], "optimizer_steps": 2,
                       "validation_every_steps": 1}
        with contextlib.redirect_stdout(io.StringIO()):
            self.base = v2.train(self.data, self.root/"base_runs", self.config)
            write_json(self.base/"verification.json", {"checkpoints_sha256": {
                "seed_11/model.pt": sha256(self.base/"seed_11/model.pt")}})
            for j, label in enumerate(v2.LABELS):
                for i in range(5):
                    v2.save_clip(self.data, *raw(1000+j*5+i), {"label": label, "split": "validation",
                                "session": "monitor", "aspect": 1.5, "source": "recording"})
            self.monitor = monitor_run(self.base, self.data)
        self.collection = self.root / "collection"
        with patch.object(v2, "TRAIN_CONFIG", self.config):
            pilot.prepare(self.collection, self.base, self.monitor)

    def complete(self):
        for sitting in (1, 2):
            for task in pilot.schedule(sitting):
                v2.save_clip(self.collection/"recordings", *raw(5000+sitting*100+task["index"]),
                             {"label": task["label"], "split": "train", "source": "recording",
                              "session": f"new-{sitting}", "aspect": 1.5, "experiment_id": pilot.EXPERIMENT,
                              "collection_sitting": sitting, "task_index": task["index"],
                              "collection_group": task["group"], "task_hint": task["hint"],
                              "collection_plan_sha256": sha256(self.collection/"plan.json")})

    def test_incomplete_collection_refused_before_allocating_study(self):
        with self.assertRaisesRegex(ValueError, "all 80"):
            study.run(self.collection, self.root/"new_runs")
        self.assertFalse((self.root/"new_runs").exists())

    def test_class_and_row_matching_without_evaluation_in_training(self):
        self.complete()
        _, config, manifest, _, arrays = study.snapshot(self.collection)
        self.assertEqual(config, self.config)
        self.assertEqual(manifest["summary"], {"base_train": 40, "selection": 40, "monitor": 40,
                                               "routine_add": 40, "targeted_add": 40})
        np.testing.assert_array_equal(arrays["routine_add_y"], arrays["targeted_add_y"])
        for c in pilot.GROUPS:
            files = manifest["rows"]["base_train"] + manifest["rows"][c+"_add"]
            self.assertFalse(set(files)&set(manifest["rows"]["selection"]))
            self.assertFalse(set(files)&set(manifest["rows"]["monitor"]))
            self.assertFalse(set(manifest["rows"][c+"_add"])&set(manifest["rows"][("routine" if c=="targeted" else "targeted")+"_add"]))
        expected_slots = [(label, sitting, repeat) for label in pilot.FOCUS for sitting in (1,2) for repeat in range(5)]
        by_name = {r["file"]: r for r in manifest["records"]}
        for c in pilot.GROUPS:
            actual = [(by_name[f]["label"], by_name[f]["sitting"], by_name[f]["task_index"]%5)
                      for f in manifest["rows"][c+"_add"]]
            self.assertEqual(actual, expected_slots)

    def test_foreign_test_file_refused_without_coordinate_access(self):
        self.complete()
        path = self.collection/"recordings"/"foreign_test.npz"
        np.savez_compressed(path, metadata=json.dumps({"split": "test", "experiment_id": pilot.EXPERIMENT}))
        with patch.object(v2, "load_clip", side_effect=AssertionError("Must inspect metadata first")):
            with self.assertRaisesRegex(ValueError, "no payload accessed"):
                study.snapshot(self.collection)

    def test_reference_recipe_mismatch_rejected(self):
        self.complete()
        plan = json.loads((self.collection/"plan.json").read_text())
        plan["training_config"]["optimizer_steps"] = 3
        write_json(self.collection/"plan.json", plan)
        (self.collection/"plan.sha256").write_text(sha256(self.collection/"plan.json"))
        # Update task provenance in this synthetic fixture to reach the recipe guard.
        with patch.object(pilot, "audit", return_value={"collection_complete": True}):
            with self.assertRaisesRegex(ValueError, "reference training recipe"):
                study.snapshot(self.collection)

    def test_paired_initialization_sampling_and_class_weights(self):
        self.complete()
        _, config, _, _, arrays = study.snapshot(self.collection)
        stats = []
        for c in pilot.GROUPS:
            x = np.concatenate((arrays["base_train_x"], arrays[c+"_add_x"]))
            y = np.concatenate((arrays["base_train_y"], arrays[c+"_add_y"]))
            _, history, result = study.fit_candidate(x, y, arrays["selection_x"], arrays["selection_y"], 11, config)
            self.assertEqual(len(history), 2)
            self.assertEqual(result["training_clips"], 80)
            stats.append(result)
        for field in ("initial_state_sha256", "batch_indices_sha256", "training_class_counts"):
            self.assertEqual(stats[0][field], stats[1][field])

    def test_complete_synthetic_study_preserves_sources_models_and_scores(self):
        self.complete()
        old_hash = sha256(self.base/"seed_11/model.pt")
        # A metadata-only final-test entry elsewhere must remain unused.
        np.savez_compressed(self.data/"test_metadata_only.npz", metadata=json.dumps({"split": "test", "session": "final"}))
        with contextlib.redirect_stdout(io.StringIO()):
            output = study.run(self.collection, self.root/"new_runs")
        self.assertEqual(sha256(self.base/"seed_11/model.pt"), old_hash)
        self.assertEqual(json.loads((output/"status.json").read_text()),
                         {"state": "complete", "trained_trials": 2, "evaluated_candidates": 3})
        verified = study.verify(output)
        self.assertEqual(verified["saved_prediction_vectors_verified"], 240)
        trials = json.loads((output/"trials.json").read_text())
        self.assertEqual(set(r["condition"] for r in trials), set(study.CONDITIONS))
        self.assertEqual(len(trials), 3)
        summary = json.loads((output/"summary.json").read_text())
        diff = trials[2]["monitor"]["macro_f1"]-trials[1]["monitor"]["macro_f1"]
        self.assertAlmostEqual(summary["primary_comparison"]["mean"], diff)
        self.assertTrue((output/"report.md").is_file())
        self.assertTrue((output/"source/research/experiments/boundary_train.py").is_file())
        for c in pilot.GROUPS:
            _, checkpoint = v2.load_model(output/c/"seed_11/model.pt")
            self.assertEqual(len(checkpoint["training_files"]), 80)
            self.assertFalse(set(checkpoint["training_files"]) & set(checkpoint["monitor_files"]))
        # Integrity checks detect an altered archived guess, rather than trusting metrics.
        p = output/"routine/seed_11/monitor_predictions.json"
        rows = json.loads(p.read_text()); rows[0]["scores"] = [0.125]*8
        write_json(p, rows)
        with self.assertRaises(AssertionError):
            study.verify(output)


if __name__ == "__main__":
    unittest.main()
