"""New-session leakage, artifact and no-retraining checks; no accuracy claims."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .commands_v2 import LABELS, TRAIN_CONFIG, contract, save_clip, train
from .evaluate_v2_sessions import aggregate, check_study, new_cohort, run
from .run import sha256, write_json
from .test_commands_v2 import dataset as fixture_dataset, raw


def cohort_fixture():
    old = [{"file": "old_train", "sha256": "train_hash", "split": "train", "session": "train",
            "raw_content_sha256": "old_train_raw"},
           {"file": "old_val", "sha256": "val_hash", "split": "validation", "session": "old_val",
            "raw_content_sha256": "old_val_raw"}]
    new = [{"file": f"{label}_{i}", "sha256": f"{label}_{i}", "split": "validation", "session": "new_val",
            "source": "recording", "label": label, "raw_content_sha256": f"{label}_{i}",
            "passes_tracking_quality": label != "NO_COMMAND"} for label in LABELS for i in range(5)]
    samples = {"train": [(None, 0, "old_train")],
               "validation": [(None, 0, "old_val")] + [(None, LABELS.index(r["label"]), r["file"]) for r in new]}
    return samples, {"records": old+new, "contract": contract()}, {"records": old, "contract": contract()}


class NewV2SessionTests(unittest.TestCase):
    def test_new_only_includes_all_classes_and_keeps_tracking_failures(self):
        data, current, previous = cohort_fixture()
        samples, manifest = new_cohort(data, current, previous)
        self.assertEqual(len(samples), 40)
        self.assertEqual(manifest["summary"]["sessions"], ["new_val"])
        self.assertEqual(len(manifest["summary"]["tracking_quality_failures"]), 5)
        self.assertFalse(any(s[2].startswith("old") for s in samples))

    def test_missing_or_changed_old_inputs_rejected(self):
        for operation in ("change", "remove"):
            with self.subTest(operation=operation):
                data, current, previous = cohort_fixture()
                if operation == "change":
                    current["records"][0] = {**current["records"][0], "sha256": "changed"}
                else:
                    current["records"] = current["records"][1:]
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    new_cohort(data, current, previous)

    def test_session_source_split_and_raw_overlap_rejected(self):
        for key, value in (("session", "old_val"), ("session", "train"),
                           ("source", "correction"), ("split", "train"),
                           ("raw_content_sha256", "old_train_raw"),
                           ("raw_content_sha256", "old_val_raw"),
                           ("raw_content_sha256", "BACKWARD_0")):
            with self.subTest(key=key, value=value):
                data, current, previous = cohort_fixture()
                current["records"][2] = {**current["records"][2], key: value}
                with self.assertRaises(ValueError):
                    new_cohort(data, current, previous)

    def test_empty_insufficient_misaligned_or_contract_changed_cohort_rejected(self):
        for operation in ("empty", "missing_label", "missing_features", "contract"):
            with self.subTest(operation=operation):
                data, current, previous = cohort_fixture()
                if operation == "empty":
                    current["records"] = copy.deepcopy(previous["records"])
                elif operation == "missing_label":
                    current["records"] = current["records"][:-1]
                elif operation == "missing_features":
                    data["validation"] = data["validation"][:-1]
                else:
                    current["contract"] = {"labels": ["wrong"]}
                with self.assertRaises(ValueError):
                    new_cohort(data, current, previous)

    def test_changes_pair_seeds_without_pooling_sessions(self):
        original = []
        for seed in (11, 22):
            original.append({"seed": seed, "accuracy": 0.8, "macro_f1": 0.8, "nll": 0.2,
                             "no_command_to_motion_rate": 0.2, "stop_recall": 0.8,
                             "per_class": {label: {"support": 6, "recall": 0.8} for label in LABELS}})
        new = [{**r, "accuracy": 0.6, "macro_f1": 0.5,
                "per_class": {label: {"support": 5, "recall": 0.6} for label in LABELS}} for r in original]
        s = aggregate(new, original)
        self.assertEqual(s["metrics"]["accuracy"]["mean"], 0.6)
        self.assertEqual(s["old_selection_session_metrics"]["accuracy"]["mean"], 0.8)
        self.assertAlmostEqual(s["paired_changes_from_selection_session"]["macro_f1"]["mean"], -0.3)
        self.assertEqual(s["class_recall_comparison"]["UP"]["new_support_per_model"], 5)

    def test_saved_checkpoint_inference_without_optimizer_or_test_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "data"
            fixture_dataset(folder)
            # Incomplete test payload must never be needed for this validation evaluation.
            np.savez_compressed(folder / "test_metadata_only.npz",
                                metadata=json.dumps({"split": "test", "session": "test"}))
            config = {**TRAIN_CONFIG, "seeds": [11], "optimizer_steps": 2, "validation_every_steps": 1}
            with contextlib.redirect_stdout(io.StringIO()):
                study = train(folder, Path(tmp) / "runs", config)
            before_model = sha256(study / "seed_11/model.pt")
            write_json(study / "verification.json", {"checkpoints_sha256": {"seed_11/model.pt": before_model}})
            for label_index, label in enumerate(LABELS):
                for i in range(5):
                    save_clip(folder, *raw(1000+label_index*5+i),
                              {"label": label, "split": "validation", "session": "new_val",
                               "source": "recording", "aspect": 1.5})
            with patch("torch.optim.Adam", side_effect=AssertionError("Evaluation must not fit weights")), contextlib.redirect_stdout(io.StringIO()):
                output = run(study, folder)
            self.assertEqual(sha256(study / "seed_11/model.pt"), before_model)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["records"]), 40)
            self.assertEqual(manifest["summary"]["sessions"], ["new_val"])
            self.assertFalse(any("test" in r["session"] for r in manifest["records"]))
            self.assertEqual(json.loads((output / "status.json").read_text()), {"state": "complete", "completed_trials": 1})
            scores = json.loads((output / "seed_11/predictions.json").read_text())
            self.assertEqual(len(scores), 40)
            with np.load(output / "new_validation_features.npz", allow_pickle=False) as a:
                self.assertEqual(a["x"].shape, (40, 32, 67))
            self.assertTrue((output / "report.md").exists())
            (study / "seed_11/model.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "verification hash"):
                check_study(study)


if __name__ == "__main__":
    unittest.main()
