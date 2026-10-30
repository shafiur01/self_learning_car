"""Meaningful guards for 017 formulas, target semantics and access boundary."""

import unittest
from pathlib import Path

import numpy as np

from . import published_selector_replay as replay


class PublishedReplayTests(unittest.TestCase):
    def test_equal_confidence_has_unit_gaussian_weight(self):
        # Balanced repeated poses: no confidence deviation to penalize.
        q = np.full((8, 8), .05)
        np.fill_diagonal(q, .65)
        _, weights, params = replay.components(q)
        np.testing.assert_allclose(params['thresholds'], [.65] * 8)
        for name in ('gaussian_snapshot', 'gaussian_ua_snapshot'):
            np.testing.assert_allclose(weights[name], 1, atol=1e-12)

    def test_gaussian_half_two_sigma_and_zero_variance(self):
        # The paper's variance / 4 means original sigma/2 is one kernel sigma.
        values = replay.gaussian(np.array([.4, .6, .8, .9]), .8, .16)
        np.testing.assert_allclose(values, [np.exp(-2), np.exp(-.5), 1, 1])
        np.testing.assert_array_equal(replay.gaussian(np.array([.4, .5, .6]), .5, 0), [0, 1, 1])

    def test_alignment_does_not_relabel(self):
        # First vector's aligned winner changes to rare class1; raw target stays0.
        q = np.tile([.80, .08, .02, .02, .02, .02, .02, .02], (20, 1))
        q[0] = [.45, .40, .025, .025, .025, .025, .025, .025]
        aligned = q / q.mean(0)
        self.assertEqual(aligned[0].argmax(), 1)
        targets, _, _ = replay.components(q)
        self.assertEqual(targets[0], 0)

    def test_permutation_equivariance_and_finite_weights(self):
        q = np.random.default_rng(17).dirichlet(np.ones(8), size=30)
        targets, weights, _ = replay.components(q)
        order = np.random.default_rng(18).permutation(30)
        target2, weights2, _ = replay.components(q[order])
        np.testing.assert_array_equal(target2, targets[order])
        for method in weights:
            np.testing.assert_allclose(weights2[method], weights[method][order], atol=1e-12)

    def test_bad_probability_inputs_rejected(self):
        for q in (np.zeros((2, 8)), np.full((2, 8), np.nan), np.full((2, 7), 1/7)):
            with self.assertRaises(ValueError):
                replay.components(q)

    def test_forbidden_payload_fails_before_open(self):
        with self.assertRaisesRegex(ValueError, 'Forbidden input'):
            replay.checked_load(Path('/not-an-allowed-input.npz'), set(), [], 'score')

    def test_weighted_denominators_and_background_proxy(self):
        s = replay.stats(np.array([.5, .25, 1]), np.array([5, 6, 4]),
                         np.array([7, 6, 4]), np.array([5, 0, 4]))
        self.assertEqual(s['wrong_mass'], .5)
        self.assertEqual(s['correct_student_error_mass'], .25)
        self.assertEqual(s['none_to_movement_fraction'], 1)
        empty = replay.stats(np.zeros(3), np.array([5, 6, 4]), np.array([7, 6, 4]), np.array([5, 0, 4]))
        self.assertIsNone(empty['wrong_fraction'])
        self.assertIsNone(empty['none_to_movement_fraction'])


if __name__ == '__main__':
    unittest.main()
