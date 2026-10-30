"""Research-integrity checks; synthetic data do not establish gesture accuracy."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gesture_commands.data import FEATURES, FRAMES, LABELS, save_clip
from gesture_commands.model import load_model
from gesture_commands.train import train_model
from .augment import VARIANTS, augment
from .run import (DEFAULT_CONFIG, development_snapshot, run, train_trial,
                  validate_config, validation_metrics)


def config():
    return json.loads(DEFAULT_CONFIG.read_text())


def motion_batch():
    batch = torch.zeros((2, FRAMES, FEATURES))
    batch[:, :, 3:63] = 0.2
    batch[0, :, 63] = torch.linspace(0, 2, FRAMES)
    batch[1, :, 63] = torch.linspace(0, -2, FRAMES)
    batch[:, :, -1] = 1
    batch[:, 10:14, -1] = 0
    return batch


class AugmentationTests(unittest.TestCase):
    def test_baseline_does_not_mutate_or_alias_recorded_data(self):
        original = motion_batch()
        before = original.clone()
        changed = augment(original, "baseline", torch.Generator(), config()["augmentation"])
        torch.testing.assert_close(before, changed, rtol=0, atol=0)
        changed.fill_(99)
        torch.testing.assert_close(before, original, rtol=0, atol=0)

    def test_jitter_preserves_wrist_motion_size_and_visibility(self):
        original = motion_batch()
        changed = augment(original, "jitter", torch.Generator().manual_seed(12), config()["augmentation"])
        torch.testing.assert_close(changed[:, :, :3], original[:, :, :3], rtol=0, atol=0)
        torch.testing.assert_close(changed[:, :, 63:], original[:, :, 63:], rtol=0, atol=0)
        torch.testing.assert_close(changed[:, 10:14], original[:, 10:14], rtol=0, atol=0)
        self.assertFalse(torch.equal(changed[:, :10, 3:63], original[:, :10, 3:63]))

    def test_time_warp_keeps_endpoints_and_direction_for_monotone_motion(self):
        original = motion_batch()
        changed = augment(original, "time_warp", torch.Generator().manual_seed(8), config()["augmentation"])
        torch.testing.assert_close(changed[:, 0], original[:, 0])
        torch.testing.assert_close(changed[:, -1], original[:, -1])
        self.assertTrue(torch.all(changed[0, 1:, 63] - changed[0, :-1, 63] > 0))
        self.assertTrue(torch.all(changed[1, 1:, 63] - changed[1, :-1, 63] < 0))
        self.assertEqual(set(changed[:, :, -1].flatten().tolist()), {0.0, 1.0})
        self.assertFalse(torch.equal(changed[:, :, 63], original[:, :, 63]))

    def test_all_invisible_sequences_do_not_acquire_synthetic_hands(self):
        empty = torch.zeros((2, FRAMES, FEATURES))
        for variant in VARIANTS:
            changed = augment(empty, variant, torch.Generator().manual_seed(42), config()["augmentation"])
            self.assertEqual(torch.count_nonzero(changed).item(), 0)

    def test_augmentation_is_reproducible_and_isolates_global_randomness(self):
        batch = motion_batch()
        before = torch.random.get_rng_state().clone()
        first = augment(batch, "combined", torch.Generator().manual_seed(77), config()["augmentation"])
        second = augment(batch, "combined", torch.Generator().manual_seed(77), config()["augmentation"])
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def write_dataset(folder):
    rng = np.random.default_rng(10)
    for split, count in (("train", 5), ("validation", 2)):
        for label in LABELS:
            for _ in range(count):
                xyz = rng.normal(0.4, 0.02, (60, 21, 3)).astype(np.float32)
                xyz[:, 9, 1] = xyz[:, 0, 1] + 0.08
                save_clip(folder, xyz, np.linspace(0, 1.98, 60), np.ones(60, dtype=bool),
                          {"label": label, "split": split, "session": split, "aspect": 1.5})


class IntegrityTests(unittest.TestCase):
    def test_snapshot_excludes_test_payload_and_checks_split_separation(self):
        with tempfile.TemporaryDirectory() as folder:
            write_dataset(folder)
            # Deliberately no coordinates: the development loader must skip them.
            np.savez_compressed(Path(folder) / "test_only.npz",
                                metadata=json.dumps({"split": "test", "session": "test"}))
            dataset, manifest = development_snapshot(folder)
            self.assertEqual(set(dataset), {"train", "validation"})
            self.assertEqual(len(dataset["train"]), 30)
            self.assertEqual(len(dataset["validation"]), 12)
            self.assertEqual(len(manifest["records"]), 42)
            self.assertFalse(any(record["split"] == "test" for record in manifest["records"]))
            np.savez_compressed(Path(folder) / "test_only.npz",
                                metadata=json.dumps({"split": "test", "session": "train"}))
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                development_snapshot(folder)

    def test_identical_raw_clip_cannot_leak_across_development_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            write_dataset(folder)
            first = next(Path(folder).glob("*.npz"))
            with np.load(first, allow_pickle=False) as clip:
                metadata = json.loads(str(clip["metadata"]))
                other = "validation" if metadata["split"] == "train" else "train"
                save_clip(folder, clip["xyz"], clip["times"], clip["visible"],
                          {**metadata, "label": "RIGHT", "split": other, "session": other})
            with self.assertRaisesRegex(ValueError, "identical raw clip"):
                development_snapshot(folder)

    def test_rejects_duplicate_seeds_and_reversing_time_increment(self):
        settings = config()
        settings["seeds"] = [11, 11]
        with self.assertRaises(ValueError):
            validate_config(settings)
        settings = config()
        settings["augmentation"]["time_increment_range"] = [-1, 1]
        with self.assertRaises(ValueError):
            validate_config(settings)

    def test_metrics_distinguish_accuracy_macro_f1_and_false_motion(self):
        metrics = validation_metrics(np.eye(6)[[0, 1, 2, 3, 0, 1]], np.arange(6))
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["macro_f1"], 5 / 9)
        self.assertEqual(metrics["no_command_to_motion_count"], 1)
        self.assertEqual(metrics["stop_to_motion_count"], 1)
        self.assertEqual(metrics["stop_recall"], 0)

    def test_baseline_matches_existing_training_and_repeats_exactly(self):
        rng = np.random.default_rng(1)
        dataset = {"train": [], "validation": []}
        for split, count in (("train", 5), ("validation", 2)):
            for label in range(len(LABELS)):
                for _ in range(count):
                    x = rng.normal(0, 0.01, (FRAMES, FEATURES)).astype(np.float32)
                    x[:, label] += 1
                    dataset[split].append((x, label, "synthetic"))
        settings = config()
        settings["epochs"] = 2
        original, original_epoch, _ = train_model(dataset, epochs=2, seed=11)
        baseline, metrics, _, _ = train_trial(dataset, settings, "baseline", 11)
        self.assertEqual(original_epoch, metrics["best_epoch"])
        for name, values in original.state_dict().items():
            torch.testing.assert_close(values, baseline.state_dict()[name], rtol=0, atol=0)
        repeated, repeated_metrics, _, _ = train_trial(dataset, settings, "baseline", 11)
        self.assertEqual(metrics["initial_weights_sha256"], repeated_metrics["initial_weights_sha256"])
        for name, values in baseline.state_dict().items():
            torch.testing.assert_close(values, repeated.state_dict()[name], rtol=0, atol=0)

    def test_complete_study_archives_inputs_and_loads_compatible_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            write_dataset(data)
            settings = config()
            settings.update(data=str(data), epochs=1, seeds=[11], variants=["baseline", "combined"])
            config_path = root / "config.json"
            config_path.write_text(json.dumps(settings))
            with contextlib.redirect_stdout(io.StringIO()):
                output = run(config_path, root / "runs")
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
            self.assertEqual(json.loads((output / "status.json").read_text())["completed_trials"], 2)
            with np.load(output / "development_features.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["train_x"].shape, (30, FRAMES, FEATURES))
                self.assertEqual(archive["validation_x"].shape, (12, FRAMES, FEATURES))
                inputs = torch.from_numpy(archive["validation_x"].copy())
            for variant in ("baseline", "combined"):
                trial = output / f"{variant}_seed_11"
                model, checkpoint = load_model(trial / "model.pt")
                self.assertEqual(checkpoint["development_sessions"], ["train", "validation"])
                with torch.inference_mode():
                    scores = model(inputs).softmax(1).numpy()
                saved = json.loads((trial / "validation_predictions.json").read_text())
                np.testing.assert_allclose(scores, [item["scores"] for item in saved], rtol=1e-6, atol=1e-7)
            self.assertTrue((output / "source" / "research" / "PROTOCOL.md").is_file())
            self.assertTrue((output / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
