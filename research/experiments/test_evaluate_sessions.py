"""Leakage guards for frozen-model evaluation; no synthetic accuracy claims."""

import copy
import unittest

from gesture_commands.data import LABELS
from .evaluate_sessions import new_validation_cohort, summarize


def cohort_fixture():
    old = [{"file": "old_train", "sha256": "old_train_hash", "split": "train",
            "session": "train_session", "raw_content_sha256": "old_train_raw"},
           {"file": "old_validation", "sha256": "old_val_hash", "split": "validation",
            "session": "old_val_session", "raw_content_sha256": "old_val_raw"}]
    new = [{"file": label, "sha256": label, "split": "validation", "session": "new_session",
            "source": "recording", "label": label, "raw_content_sha256": label,
            "passes_tracking_quality": label != "NO_COMMAND"} for label in LABELS]
    data = {"train": [(None, 0, "old_train")],
            "validation": [(None, 0, "old_validation")] + [(None, i, label) for i, label in enumerate(LABELS)]}
    return data, {"records": old + new}, {"records": old}


class FrozenEvaluationTests(unittest.TestCase):
    def test_new_session_only_keeps_all_labels_and_tracking_failures(self):
        data, current, previous = cohort_fixture()
        samples, manifest = new_validation_cohort(data, current, previous)
        self.assertEqual([s[2] for s in samples], list(LABELS))
        self.assertEqual(manifest["summary"]["clips"], 6)
        self.assertEqual(manifest["summary"]["tracking_quality_failures"], 1)
        self.assertFalse(any(r["session"].startswith("old") for r in manifest["records"]))

    def test_changed_or_missing_old_recording_rejected(self):
        for change in ("modify", "delete"):
            data, current, previous = cohort_fixture()
            if change == "modify":
                current["records"][0] = {**current["records"][0], "sha256": "changed"}
            else:
                current["records"] = current["records"][1:]
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                new_validation_cohort(data, current, previous)

    def test_session_content_source_and_split_leakage_rejected(self):
        changes = [("session", "train_session"), ("session", "old_val_session"),
                   ("raw_content_sha256", "old_train_raw"), ("raw_content_sha256", "old_val_raw"),
                   ("raw_content_sha256", "BACKWARD"), ("source", "correction"), ("split", "train")]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                data, current, previous = cohort_fixture()
                current["records"][2] = {**current["records"][2], key: value}
                with self.assertRaises(ValueError):
                    new_validation_cohort(data, current, previous)

    def test_empty_incomplete_or_misaligned_cohorts_rejected(self):
        for change in ("empty", "missing_label", "missing_features"):
            data, current, previous = cohort_fixture()
            if change == "empty":
                current["records"] = copy.deepcopy(previous["records"])
            elif change == "missing_label":
                current["records"] = current["records"][:-1]
                data["validation"] = data["validation"][:-1]
            else:
                data["validation"] = data["validation"][:-1]
            with self.assertRaises(ValueError):
                new_validation_cohort(data, current, previous)

    def test_paired_comparison_and_old_new_sessions_stay_separate(self):
        rows = []
        for seed in (11, 22):
            for condition, accuracy in (("small", 0.4), ("all_original", 0.8)):
                rows.append({"seed": seed, "condition": condition, "training_clips": 30,
                             "accuracy": accuracy, "macro_f1": accuracy, "nll": 1.,
                             "no_command_to_motion_rate": 0.2, "stop_recall": 0.5,
                             "stop_to_motion_count": 2,
                             "per_class": {label: {"recall": 0.5} for label in LABELS}})
        old = [{**row, "accuracy": 0.9} for row in rows]
        result = summarize(rows, old)["conditions"]["small"]
        self.assertAlmostEqual(result["accuracy"]["mean"], 0.4)
        self.assertAlmostEqual(result["accuracy_change_from_old_session_pp"], -50.)
        self.assertAlmostEqual(result["paired_difference_from_all_original"]["macro_f1"]["mean"], -0.4)


if __name__ == "__main__":
    unittest.main()
