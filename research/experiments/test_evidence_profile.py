"""Meaningful leakage, pairing, missingness and selector checks; synthetic only."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import evidence_profile as ep
from . import evidence_profile_run as run
from .evidence_profile_demo import synthetic_logits


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_masks_preserve_context_inputs_and_missingness(self):
        x = np.ones((4, 32, 146), dtype=np.float32)
        copy = x.copy(); observed = ep.observed(x, np.arange(4)).numpy()
        np.testing.assert_array_equal(x, copy)
        np.testing.assert_array_equal(observed[:, :, list(ep.BLOCKS['context'])], 1)
        for j, key in enumerate(('pose', 'articulation', 'transport')):
            np.testing.assert_array_equal(observed[j + 1, :, list(ep.BLOCKS[key])], 0)
        np.testing.assert_array_equal(observed[:, 0, 146:], ep.MASKS)
        # All visible zero values still have bits111, unlike missing pose011.
        zero = ep.observed(np.zeros((1, 32, 146), np.float32)).numpy()
        np.testing.assert_array_equal(zero[0, 0, 146:], 1)

    def test_response_signs_and_logit_shift_invariance(self):
        logits = np.stack([synthetic_logits(.99, [.2, -.3, 2.1]), synthetic_logits(.97, [3, .1, .2])], axis=1)
        actual = ep.responses(logits)
        np.testing.assert_allclose(actual[:, 0], [[.2, -.3, 2.1], [3, .1, .2]], atol=1e-12)
        np.testing.assert_allclose(actual, ep.responses(logits + 17), atol=1e-12)

    def test_hidden_label_api_and_foreign_bundle_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pool.npz'
            fields = dict(x=np.zeros((96, 32, 146), np.float32), labeled=np.arange(80),
                y=np.tile(np.repeat(np.arange(8), 5), 2), teacher_a=np.arange(40), teacher_b=np.arange(40, 80), unlabeled=np.arange(80, 96))
            np.savez_compressed(p, **fields)
            pool = ep.LearnerPool.load(p)
            with self.assertRaisesRegex(ValueError, 'hidden'):
                pool.labels_for([80])
            np.savez_compressed(p, **fields, hidden_y=np.zeros(16))
            with self.assertRaisesRegex(ValueError, 'Foreign/hidden'):
                ep.LearnerPool.load(p)

    def test_matched_controls_preserve_predicted_class_weight_multisets(self):
        q = np.full((32, 8), .001, np.float32)
        for i in range(32): q[i, i % 8] = .993
        w = np.zeros(32, np.float32); w[:8] = np.linspace(.4, .9, 8); w[8:16] = .8
        base = np.ones(32, dtype=bool)
        for random in (False, True):
            out = ep.matched_weights(w, base, q, 11, random=random)
            for c in range(8):
                indices = q.argmax(1) == c
                np.testing.assert_array_equal(np.sort(out[indices]), np.sort(w[indices]))
            np.testing.assert_array_equal(out, ep.matched_weights(w, base, q, 11, random=random))

    def test_unsupported_anchors_never_admit(self):
        rng = np.random.default_rng(4)
        query = rng.normal(size=(16, 8, 3)); anchors = rng.normal(size=(16, 8, 3))
        y = np.repeat(np.arange(8), 2); p = np.eye(8)[y] * .99 + .01/8
        config = {'scale_floor': .05, 'anchor_confidence': .5, 'minimum_anchors': 3, 'radius_floor_squared': .0025}
        w, details = ep.support(query, anchors, y, p, np.arange(16) % 8, config)
        self.assertEqual(w.sum(), 0)
        self.assertTrue(all(not r['supported'] for r in details))

    def test_tiny_fits_pairing_zero_coverage_equivalence_and_checkpoint(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(40, 32, 146)).astype(np.float32); y = np.repeat(np.arange(8), 5)
        config = {'steps': 4, 'validation_checks': 2, 'batch_size': 8, 'learning_rate': .001, 'ramp_steps': 1}
        m, h, s = ep.fit(x, y, x, y, 11, config)
        n, _, z = ep.fit(x, y, x, y, 11, config, ux=x, target=np.eye(8, dtype=np.float32)[y], weight=np.zeros(40, np.float32))
        self.assertEqual(ep.state_hash(m), ep.state_hash(n))
        self.assertFalse(z['active_unsupervised_loss'])
        native, _, t = ep.fit(x, y, x, y, 11, config, subset=True)
        self.assertEqual(s['initial_state_sha256'], t['initial_state_sha256'])
        self.assertEqual(s['batch_indices_sha256'], t['batch_indices_sha256'])
        self.assertNotEqual(s['mask_sha256'], t['mask_sha256'])
        self.assertNotEqual(s['initial_state_sha256'], ep.state_hash(m))
        self.assertEqual(sum(p.numel() for p in m.parameters()), ep.PARAMETERS)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.pt'
            checkpoint = {'experiment_id': run.ID, 'features': 149, 'frames': 32, 'labels': list(run.LABELS),
                'feature_schema': 'motion_v2_availability_v1', 'parameters': ep.PARAMETERS, 'state_dict': m.state_dict()}
            torch.save(checkpoint, path)
            loaded, _ = run.load_model(path)
            np.testing.assert_array_equal(ep.infer(m, x)[0], ep.infer(loaded, x)[0])
            checkpoint['features'] = 146; torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'compatible'):
                run.load_model(path)

    def test_endpoint_blocked_until_all_fits(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p / 'status.json').write_text(json.dumps({'phase': 'prepared', 'completed_fits': 0}))
            with patch.object(run, 'frozen', return_value={}), patch.object(np, 'load', side_effect=AssertionError('Payload accessed')):
                with self.assertRaisesRegex(ValueError, 'Complete all65'):
                    run.evaluate(p)
            self.assertFalse((p / 'endpoint_access.json').exists())


if __name__ == '__main__':
    unittest.main()
