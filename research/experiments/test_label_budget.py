"""Checks for budget integrity and artifacts; synthetic accuracy is not evidence."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gesture_commands.data import LABELS, save_clip
from gesture_commands.model import load_model
from .label_budget import (DEFAULT_CONFIG, nested_subsets, original_pool, run,
                           train_steps, validate_config)
from .run import development_snapshot


def settings():
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(labels_per_class=[5, 6], seeds=[11], optimizer_steps=3,
                  validation_every_steps=2)
    return config


def write_data(folder):
    rng = np.random.default_rng(71)
    for split, source, count in (("train", "recording", 7), ("train", "correction", 5),
                                 ("validation", "recording", 2)):
        for label in LABELS:
            for i in range(count):
                xyz = rng.normal(0.4, 0.03, (60, 21, 3)).astype(np.float32)
                xyz[:, 9, 1] = xyz[:, 0, 1] + 0.08
                session = f"{split}-{source}-{i % 2}" if split == "train" else "validation"
                save_clip(folder, xyz, np.linspace(0, 1.98, 60), np.ones(60, dtype=bool),
                          {"label": label, "split": split, "source": source,
                           "session": session, "aspect": 1.5})


class LabelBudgetTests(unittest.TestCase):
    def test_nested_budgets_exclude_corrections_and_balance_sessions(self):
        with tempfile.TemporaryDirectory() as folder:
            write_data(folder)
            dataset, manifest = development_snapshot(folder)
            pool = original_pool(dataset, manifest)
            self.assertEqual(len(pool), 42)
            plans = nested_subsets(pool, manifest, settings(), 11)
            repeated = nested_subsets(pool, manifest, settings(), 11)
            records = {r["file"]: r for r in manifest["records"]}
            small = {s[2] for s in plans["labels_5_per_class"]}
            large = {s[2] for s in plans["labels_6_per_class"]}
            self.assertTrue(small < large)
            self.assertTrue(large < {s[2] for s in plans["all_original"]})
            for name, samples in plans.items():
                self.assertEqual([s[2] for s in samples], [s[2] for s in repeated[name]])
                self.assertTrue(all(records[s[2]]["source"] == "recording" for s in samples))
                self.assertTrue(all(records[s[2]]["split"] == "train" for s in samples))
            for label_index in range(len(LABELS)):
                selected = [s for s in plans["labels_5_per_class"] if s[1] == label_index]
                self.assertEqual(len(selected), 5)
                count = sum(records[s[2]]["session"].endswith("0") for s in selected)
                self.assertIn(count, (2, 3))

    def test_corrections_cannot_fill_an_insufficient_original_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            write_data(folder)
            dataset, manifest = development_snapshot(folder)
            config = settings()
            config["labels_per_class"] = [5, 10]
            with self.assertRaisesRegex(ValueError, "Not enough original"):
                nested_subsets(original_pool(dataset, manifest), manifest, config, 11)

    def test_uniform_update_budget_and_checkpoint_schedule_are_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            write_data(folder)
            dataset, manifest = development_snapshot(folder)
            plans = nested_subsets(original_pool(dataset, manifest), manifest, settings(), 11)
            initial_hashes = []
            for samples in plans.values():
                model, metrics, history, probabilities = train_steps(samples, dataset["validation"], settings(), 11)
                self.assertEqual([h["step"] for h in history], [2, 3])
                self.assertEqual(metrics["optimizer_steps"], 3)
                self.assertEqual(metrics["example_draws"], 96)
                self.assertEqual(metrics["training_clips"], len(samples))
                initial_hashes.append(metrics["initial_weights_sha256"])
                repeated, _, _, repeated_probabilities = train_steps(samples, dataset["validation"], settings(), 11)
                np.testing.assert_array_equal(probabilities, repeated_probabilities)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, repeated.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(len(set(initial_hashes)), 1)

    def test_configuration_does_not_silently_change_sampling_or_sources(self):
        for key, value in (("eligible_source", "correction"), ("sampling", "shuffle"),
                           ("labels_per_class", [10, 5]), ("seeds", [11, 11])):
            config = settings()
            config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_complete_study_excludes_test_payload_and_archives_exact_training_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            write_data(data)
            # No test coordinates/labels exist: the runner must never require them.
            np.savez_compressed(data / "test_metadata_only.npz",
                                metadata=json.dumps({"split": "test", "session": "test"}))
            config = settings()
            config["data"] = str(data)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                output = run(config_path, root / "runs")
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status, {"state": "complete", "completed_trials": 3})
            selections = json.loads((output / "selections.json").read_text())
            with np.load(output / "development_features.npz", allow_pickle=False) as archive:
                vx = torch.from_numpy(archive["validation_x"].copy())
                self.assertEqual(len(archive["train_x"]), 72)  # full pool retained, not all used
            for selection in selections:
                trial = output / f"{selection['condition']}_seed_11"
                model, checkpoint = load_model(trial / "model.pt")
                self.assertEqual(checkpoint["training_label_files"], selection["files"])
                self.assertFalse(any("test" in session for session in checkpoint["development_sessions"]))
                metrics = json.loads((trial / "metrics.json").read_text())
                self.assertEqual(metrics["training_clips"], len(selection["files"]))
                with torch.inference_mode():
                    scores = model(vx).softmax(1).numpy()
                saved = json.loads((trial / "validation_predictions.json").read_text())
                np.testing.assert_allclose(scores, [r["scores"] for r in saved], rtol=1e-6, atol=1e-7)
            self.assertTrue((output / "label_budget_curve.png").is_file())
            self.assertTrue((output / "label_budget_curve.pdf").is_file())
            self.assertTrue((output / "source" / "research" / "LESSON_02.md").is_file())
            environment = json.loads((output / "environment.json").read_text())
            self.assertIn("selections.json", environment["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
