"""018 preflight guards: semantics, statistics, gradients and paired randomness."""

import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import online_softmatch as study


class OnlineSoftMatchTests(unittest.TestCase):
    def test_statistics_timescale_and_synthetic_shift(self):
        result=study.preflight()
        self.assertAlmostEqual(result['momentum']**(245/32),.5)
        self.assertLess(result['bootstrap_initial_contribution_after420_updates'],1e-12)

    def test_bootstrap_formula_and_raw_targets(self):
        q=np.tile([.80,.08,.02,.02,.02,.02,.02,.02],(20,1))
        q[0]=[.45,.40,.025,.025,.025,.025,.025,.025]
        stats=study.SoftWeights(.9)
        target,weight=stats.update(torch.tensor(q,requires_grad=True))
        aligned=q/q.mean(0);aligned/=aligned.sum(1,keepdims=True)
        self.assertEqual(aligned[0].argmax(),1)
        self.assertEqual(target[0],0)
        self.assertFalse(weight.requires_grad)
        expected=np.exp(-2*np.minimum(aligned.max(1)-q.max(1).mean(),0)**2/q.max(1).var(ddof=1))
        np.testing.assert_allclose(weight.numpy(),expected.astype(np.float32),rtol=1e-6)

    def test_stats_update_uses_raw_distribution(self):
        q=torch.tensor([[.6,.1,.05,.05,.05,.05,.05,.05],
                        [.3,.4,.05,.05,.05,.05,.05,.05]],dtype=torch.float64)
        state=study.SoftWeights(.75)
        state.update(q)
        self.assertAlmostEqual(state.mean,.5)
        self.assertAlmostEqual(state.var,.02)
        q2=q.roll(1,1)
        state.update(q2)
        self.assertAlmostEqual(state.mean,.5)
        np.testing.assert_allclose(state.marginal,.75*q.mean(0)+.25*q2.mean(0))

    def test_weighted_loss_is_over_full_batch(self):
        logits=torch.zeros((4,8),requires_grad=True)
        targets=torch.tensor([0,1,2,3])
        weights=torch.tensor([1.,0,0,0])
        loss=study.weighted_loss(logits,targets,weights)
        self.assertAlmostEqual(loss.item(),np.log(8)/4,places=6)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[1:]==0))
        self.assertGreater(logits.grad[0].abs().sum(),0)

    def test_zero_weights_give_zero_gradient(self):
        logits=torch.randn((4,8),requires_grad=True)
        loss=study.weighted_loss(logits,torch.zeros(4,dtype=torch.long),torch.zeros(4))
        loss.backward()
        self.assertEqual(loss.item(),0)
        self.assertTrue(torch.all(logits.grad==0))

    def test_input_bytes_and_private_dropout_streams(self):
        torch.manual_seed(18)
        model=nn.Sequential(nn.Linear(149,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,8))
        raw=np.random.default_rng(18).normal(size=(8,32,146)).astype(np.float32)
        observed=study.ep.observed(raw)
        before=observed.clone()
        first=torch.Generator().manual_seed(23)
        paired=torch.Generator().manual_seed(23)
        other=torch.Generator().manual_seed(29)
        global_before=torch.get_rng_state().clone()
        a=study.stochastic_forward(model,observed,first)
        study.stochastic_forward(model,observed,other)
        b=study.stochastic_forward(model,observed,paired)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(global_before,torch.get_rng_state()))
        self.assertTrue(torch.equal(before,observed))
        np.testing.assert_array_equal(observed[:,:,:146].numpy(),raw)
        self.assertTrue(torch.all(observed[:,:,146:]==1))
        model.eval()
        with torch.no_grad():
            weak1=model(observed);weak2=model(observed)
        self.assertTrue(torch.equal(weak1,weak2))
        self.assertFalse(torch.equal(a,weak1))

    def test_payload_allowlist_fails_before_open(self):
        ledger=[]
        with study.payload_guard([],ledger):
            with self.assertRaisesRegex(ValueError,'Forbidden training payload'):
                np.load(Path('/forbidden/oracle.npz'))
        self.assertEqual(ledger,[])

    def test_uniform_zero_variance_finite(self):
        target,weight=study.SoftWeights(.9).update(torch.full((32,8),.125))
        self.assertTrue(torch.isfinite(weight).all())
        self.assertTrue(torch.all(weight==1))
        self.assertTrue(torch.all(target==0))

    def test_bad_probability_inputs(self):
        for q in (torch.zeros(32,8),torch.full((32,8),float('nan')),torch.ones(32,7)):
            with self.assertRaises(ValueError):
                study.SoftWeights(.9).update(q)


if __name__=='__main__':unittest.main()
