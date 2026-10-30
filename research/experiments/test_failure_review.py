"""Check descriptive signals, pinned replay isolation, rendering and feedback."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import failure_review as review
from .run import sha256, write_json


class ReviewChecks(unittest.TestCase):
    def test_profile_distance_excludes_wrist_and_preserves_inputs(self):
        x = np.zeros((32, 67), dtype=np.float32)
        train = np.zeros((3, 32, 67), dtype=np.float32)
        train[0, :, :63] = .5
        train[1, :, 63:65] = 8
        train[2, :, :63] = .25
        before = train.copy()
        values = review.nearest_profiles(x, train, np.array([0, 0, 1]), ["STOP", "FORWARD"])
        self.assertEqual(values["STOP"]["training_row"], 1)
        self.assertEqual(values["STOP"]["pose_profile_rms"], 0)
        self.assertEqual(values["STOP"]["wrist_profile_rms"], 8)
        self.assertEqual(values["FORWARD"]["pose_profile_rms"], .25)
        np.testing.assert_array_equal(before, train)

    def test_signals_separate_tracking_from_undefined_no_hand_motion(self):
        x = np.zeros((32, 67), dtype=np.float32)
        x[:, 63] = np.linspace(0, 2, 32)
        result = review.signals(x, np.linspace(0, 2, 30), np.ones(30, dtype=bool))
        self.assertAlmostEqual(result["wrist_path_palm_units"], 2)
        self.assertEqual(result["tracking_fraction"], 1)
        no_hand = review.signals(np.zeros_like(x), np.linspace(0, 2, 30), np.zeros(30, dtype=bool))
        self.assertFalse(no_hand["motion_or_shape_defined"])

    def test_foreign_role_refused_before_payload(self):
        with patch.object(review.training, "load_clip", side_effect=AssertionError("Payload opened")):
            with self.assertRaisesRegex(ValueError, "foreign payload"):
                review.load_replay({"split": "test"}, {}, None)

    def test_html_script_data_escaped_and_self_contained(self):
        html = review.render_html({"text": "</script><script>alert(1)</script>"})
        self.assertIn("\\u003c/script>", html)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertNotIn("__REVIEW_DATA__", html)
        self.assertNotIn("__REVIEW_JS__", html)
        self.assertIn("connect-src 'none'", html)

    def test_feedback_is_versioned_without_changing_original_labels(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            clip = root / "recording.npz"
            clip.write_bytes(b"original recording")
            digest = sha256(clip)
            payload = {"bundle_id": "fixed", "labels": ["STOP", "FORWARD"], "clips": [
                {"id": "one", "file": str(clip), "sha256": digest, "label": "STOP"}]}
            write_json(root / "review_data.json", payload)
            incoming = {"version": 1, "bundle_id": "fixed", "reviews": [
                {"clip_id": "one", "clip_sha256": digest, "original_label": "STOP",
                 "assessment": "possible_label_error", "suggested_label": "FORWARD", "notes": "Review this."}]}
            path = root / "notes.json"
            write_json(path, incoming)
            first = review.import_feedback(path, root)
            second = review.import_feedback(path, root)
            self.assertNotEqual(first, second)
            self.assertEqual(sha256(clip), digest)
            self.assertEqual(json.loads((root / "review_data.json").read_text()), payload)
            incoming["reviews"].append(incoming["reviews"][0].copy())
            write_json(path, incoming)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                review.import_feedback(path, root)


if __name__ == "__main__":
    unittest.main()
