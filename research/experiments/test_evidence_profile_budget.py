"""Check the budget intervention, session partition, hidden-label boundary and gate."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import evidence_profile as ep
from . import evidence_profile_budget_run as run


class BudgetChecks(unittest.TestCase):
    def test_nested_balanced_session_sampling(self):
        y = np.tile(np.repeat(np.arange(8), 16), 2)
        groups = np.repeat(np.arange(2), 128)
        for seed in (11, 22, 33, 44, 55):
            old, labeled80, _ = run.sample_rows(y, groups, seed, 5)
            new, labeled160, hidden = run.sample_rows(y, groups, seed, 10)
            self.assertTrue(set(labeled80).issubset(labeled160))
            np.testing.assert_array_equal(np.bincount(y[labeled160]), [20] * 8)
            self.assertFalse(set(labeled160) & set(hidden))
            for side in (0, 1):
                self.assertTrue(set(old[side]).issubset(new[side]))
                np.testing.assert_array_equal(groups[new[side]], side)
                np.testing.assert_array_equal(np.bincount(y[new[side]]), [10] * 8)

    def test_insufficient_support_is_rejected(self):
        y = np.tile(np.repeat(np.arange(8), 9), 2)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            run.sample_rows(y, np.repeat(np.arange(2), 72), 11, 10)

    def test_160_bundle_does_not_expose_hidden_labels(self):
        y = np.tile(np.repeat(np.arange(8), 16), 2)
        sides, labeled, hidden = run.sample_rows(y, np.repeat(np.arange(2), 128), 11, 10)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'learner.npz'
            np.savez_compressed(path, x=np.zeros((256, 32, 146), np.float32),
                                labeled=labeled, y=y[labeled], teacher_a=sides[0],
                                teacher_b=sides[1], unlabeled=hidden)
            pool = ep.LearnerPool.load(path)
            with self.assertRaisesRegex(ValueError, 'hidden'):
                pool.labels_for(hidden[:1])
            np.testing.assert_array_equal(pool.labels_for(sides[0]), y[sides[0]])

    def test_endpoint_remains_blocked_before_all_fits(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'status.json').write_text(json.dumps({'phase': 'prepared', 'completed_fits': 0}))
            with patch.object(run, 'frozen', return_value={}), patch.object(np, 'load', side_effect=AssertionError('Payload accessed')):
                with self.assertRaisesRegex(ValueError, 'Complete all65'):
                    run.evaluate(p)
            self.assertFalse((p / 'endpoint_access.json').exists())

    def test_new_loader_refuses_old_experiment_checkpoint(self):
        torch.manual_seed(11)
        model = ep.Model()
        checkpoint = {'experiment_id': '013_evidence_profile_pilot', 'features': 149,
                      'frames': 32, 'labels': list(run.LABELS),
                      'feature_schema': 'motion_v2_availability_v1',
                      'parameters': ep.PARAMETERS, 'state_dict': model.state_dict()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.pt'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'compatible'):
                run.load_model(path)
            checkpoint['experiment_id'] = run.ID
            torch.save(checkpoint, path)
            loaded, _ = run.load_model(path)
            self.assertEqual(ep.state_hash(loaded), ep.state_hash(model))


if __name__ == '__main__':
    unittest.main()
