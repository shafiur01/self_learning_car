"""Check invariance, feature isolation, paired fitting and research artifact integrity."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import temporal_comparison as experiment
from . import commands_v3_train as training
from .run import sha256, write_json
from .test_commands_v3_train import TrainingChecks


class ArchitectureChecks(unittest.TestCase):
    def test_all_shapes_counts_and_trainable_gradients(self):
        torch.set_num_threads(2)
        for condition in experiment.CONDITIONS:
            torch.manual_seed(7)
            model = experiment.build_model(condition)
            x = torch.randn(3, 32, 67)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (3, 8))
            self.assertEqual(sum(p.numel() for p in model.parameters()), experiment.PARAMETERS[condition])
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 4, 7])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)

    def test_set_invariance_and_pose_trajectory_isolation(self):
        torch.manual_seed(17)
        x = torch.randn(4, 32, 67)
        permutation = torch.randperm(32)
        for condition in ("pose_set", "feature_set"):
            model = experiment.build_model(condition).eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(x), model(x[:, permutation]), rtol=1e-5, atol=1e-6)
        model = experiment.build_model("pose_set").eval()
        modified = x.clone()
        modified[:, :, 63:66] = 100 * torch.randn(4, 32, 3)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), model(modified), rtol=0, atol=0)
        self.assertTrue(torch.equal(x[:, :, :63], modified[:, :, :63]))

    def test_unknown_architecture_and_incomplete_summary_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            experiment.build_model("invented")
        with self.assertRaisesRegex(ValueError, "every declared"):
            experiment.summarize([], {"conditions": list(experiment.CONDITIONS), "training": {"seeds": [11]}})


class StudyChecks(unittest.TestCase):
    def setUp(self):
        TrainingChecks.setUp(self)
        TrainingChecks.complete(self)

    def test_complete_tiny_comparison_pairs_batches_preserves_inputs_and_detects_tampering(self):
        before = {r["file"]: sha256(r["file"]) for r in self.validation_plan["training_records"]}
        original_snapshot = training.snapshot
        with contextlib.redirect_stdout(io.StringIO()):
            original = training.run(self.new, self.root / "old_runs", self.config)
            with patch.object(training, "snapshot", side_effect=lambda: original_snapshot(self.new)):
                output = experiment.run(original, self.root / "comparison_runs", isolated=False)
        verified = experiment.verify(output)
        self.assertEqual(verified["saved_prediction_vectors_verified"], 480)
        self.assertEqual(len(verified["checkpoints_sha256"]), 10)
        self.assertTrue(verified["prior_cnn_states_reproduced"])
        self.assertEqual(before, {path: sha256(path) for path in before})
        models = []
        for condition in experiment.CONDITIONS:
            model, checkpoint = experiment.load_model(output / f"{condition}_seed_11/model.pt")
            models.append(model)
            self.assertEqual(checkpoint["condition"], condition)
            self.assertEqual(checkpoint["command_contract"], self.validation_plan["contract"])
            self.assertEqual(len(checkpoint["training_files"]), 125)
        prediction_file = output / "bilstm_seed_11/validation_predictions.json"
        rows = experiment.fitting.read_json(prediction_file)
        rows[0]["scores"] = [.125] * 8
        write_json(prediction_file, rows)
        with self.assertRaises(AssertionError):
            experiment.verify(output)

    def test_changed_cohort_refused_before_output_allocation(self):
        original_snapshot = training.snapshot
        with contextlib.redirect_stdout(io.StringIO()):
            original = training.run(self.new, self.root / "old_runs", self.config)
        def changed():
            plan, manifest, batches, arrays = original_snapshot(self.new)
            manifest["counts"]["train"]["UP"] += 1
            return plan, manifest, batches, arrays
        with patch.object(training, "snapshot", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "cohort differs"):
                experiment.run(original, self.root / "refused_runs", isolated=False)
        self.assertFalse((self.root / "refused_runs").exists())


if __name__ == "__main__":
    unittest.main()
