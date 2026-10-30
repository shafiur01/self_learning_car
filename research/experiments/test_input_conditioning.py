"""Check training-only units, masks, direction, LSTM equations and access guards."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import input_conditioning as r


class ConditioningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_invalid_samples_and_empty_clips_do_not_set_scales(self):
        x=np.full((3,32,146),2,dtype=np.float32);x[:,:,133]=1;x[:,:,143]=1
        x[0,16:]=1e6;x[0,16:,133]=0;x[0,16:,143]=0
        x[2]=0
        stats=r.fit_scales(x)
        for group in stats['groups'].values():
            self.assertAlmostEqual(group['rms'],2)
            self.assertEqual(group['usable_clips'],2)
        zero=r.fit_scales(np.zeros((2,32,146),np.float32))
        self.assertTrue(all(g['divisor']==1 for g in zero['groups'].values()))

    def test_each_clip_has_equal_statistical_weight(self):
        x=np.ones((2,32,146),np.float32)
        x[1,:,:]=3;x[:,:,133]=1;x[:,:,143]=1
        x[0,1:,133]=0;x[0,1:,143]=0
        self.assertAlmostEqual(r.fit_scales(x)['groups']['pose_xyz']['rms'],np.sqrt(5))

    def test_zero_masks_directions_and_inputs_preserved(self):
        rng=np.random.default_rng(16);x=rng.normal(size=(4,32,146)).astype(np.float32)
        x[:,:,66]=1;x[:,:,133]=1;x[:,:,143]=1;x[0,:,:144]=0
        original=x.copy();stats=r.fit_scales(x);y=r.transform(x,stats,'rms_clock')
        np.testing.assert_array_equal(x,original)
        np.testing.assert_array_equal(y[:,:,[66,133,143,144,145]],x[:,:,[66,133,143,144,145]])
        np.testing.assert_array_equal(y[0,:,:144],0)
        np.testing.assert_allclose(y[:,:,63:65],x[:,:,63:65]/stats['groups']['wrist_xy']['divisor'])
        np.testing.assert_array_equal(r.transform(x,stats,'raw_clock'),x)
        t=r.transform(x,stats,'rms_no_clock')
        np.testing.assert_array_equal(t[:,:,:144],y[:,:,:144]);np.testing.assert_array_equal(t[:,:,144:],0)

    def test_hidden_and_other_teacher_rows_cannot_fit_teacher_scales(self):
        n=176;x=np.ones((n,32,146),np.float32)
        pool=r.ep.LearnerPool(x,np.arange(160),np.tile(np.repeat(np.arange(8),10),2),np.arange(80),np.arange(80,160),np.arange(160,n))
        before=r.role_scales(pool,'teacher_a')
        x[80:]=1000
        self.assertEqual(before,r.role_scales(pool,'teacher_a'))
        self.assertNotEqual(before['groups'],r.role_scales(pool,'teacher_b')['groups'])
        self.assertEqual(r.role_scales(pool,'student')['training_clips'],160)

    def test_reconstructed_gate_equations_match_lstm_sequence(self):
        rng=np.random.default_rng(16);torch.manual_seed(16)
        model=r.ep.Model();x=rng.normal(size=(3,32,146)).astype(np.float32)
        z,expected=r.gate_values(model,x);cell=np.zeros((3,47));actual=[]
        sigmoid=lambda v:1/(1+np.exp(-v))
        for t in range(32):
            cell=sigmoid(z[:,t,1])*cell+sigmoid(z[:,t,0])*np.tanh(z[:,t,2])
            actual.append(sigmoid(z[:,t,3])*np.tanh(cell))
        np.testing.assert_allclose(np.stack(actual,axis=1),expected,rtol=2e-5,atol=2e-7)
        rates=r.gate_stats(model,x)
        self.assertTrue(all(0<=v['saturation_fraction']<=1 for v in rates.values()))

    def test_raw_training_reproduces_existing_fit_exactly(self):
        rng=np.random.default_rng(16);x=rng.normal(size=(40,32,146)).astype(np.float32);x[:,:,133]=1;x[:,:,143]=1
        y=np.repeat(np.arange(8),5);c={'steps':4,'validation_checks':2,'batch_size':8,'learning_rate':.001,'ramp_steps':1}
        a,_,sa=r.ep.fit(x,y,x,y,11,c,subset=True)
        b,_,sb=r.ep.fit(r.transform(x,r.fit_scales(x),'raw_clock'),y,x,y,11,c,subset=True)
        self.assertEqual(r.ep.state_hash(a),r.ep.state_hash(b));self.assertEqual(sa['mask_sha256'],sb['mask_sha256'])

    def test_evaluation_cannot_access_payloads_before_all_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'status.json').write_text(json.dumps({'phase':'prepared','completed_fits':0}))
            with patch.object(r,'frozen',return_value={}),patch.object(np,'load',side_effect=AssertionError('Payload touched')) as load:
                with self.assertRaisesRegex(ValueError,'all60'):r.evaluate(p)
                load.assert_not_called()
            self.assertFalse((p/'evaluation_access.json').exists())


if __name__=='__main__':unittest.main()
