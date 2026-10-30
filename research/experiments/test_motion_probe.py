"""Checks diagnostic intervention channels, time reversal and source preservation."""
import unittest
import numpy as np
from .motion_probe import perturb,reverse_features,js_divergence
from . import commands_v2 as v2
from .test_commands_v2 import raw


class MotionChecks(unittest.TestCase):
    def test_channel_isolation_and_no_input_mutation(self):
        x=np.random.default_rng(11).normal(size=(3,32,67)).astype(np.float32);before=x.copy()
        wrist=perturb(x,"zero_wrist_xy");np.testing.assert_array_equal(wrist[:,:,:63],x[:,:,:63]);self.assertFalse(wrist[:,:,63:65].any())
        np.testing.assert_array_equal(wrist[:,:,65:],x[:,:,65:])
        size=perturb(x,"zero_size_change");self.assertFalse(size[:,:,65].any());np.testing.assert_array_equal(size[:,:,66],x[:,:,66])
        local=perturb(x,"static_local_shape");np.testing.assert_array_equal(local[:,:,63:],x[:,:,63:])
        for i in range(32):np.testing.assert_array_equal(local[:,i,:63],x[:,16,:63])
        static=perturb(x,"static_shape_zero_trajectory");self.assertFalse(static[:,:,63:66].any());np.testing.assert_array_equal(static[:,:,66],x[:,:,66])
        np.testing.assert_array_equal(x,before)

    def test_permutation_preserves_complete_rows(self):
        x=np.arange(2*32*67,dtype=np.float32).reshape(2,32,67);p=np.arange(31,-1,-1)
        shuffled=perturb(x,"shuffled_frames",p);np.testing.assert_array_equal(shuffled,x[:,p,:])
        with self.assertRaises(ValueError):perturb(x,"shuffled_frames",np.zeros(32,dtype=int))

    def test_raw_reversal_reanchors_wrist_and_reverses_motion(self):
        xyz,t,visible=raw(33);xyz[:,:,0]+=np.linspace(0,.3,len(t))[:,None]
        original=v2.features(xyz,t,visible,1.5);backward=reverse_features(xyz,t,visible,1.5)
        self.assertFalse(backward[0,63:66].any())
        self.assertGreater(original[-1,63],0);self.assertLess(backward[-1,63],0)
        # Equal distributions have zero JS; different ones remain finite.
        p=np.eye(8,dtype=np.float32);self.assertAlmostEqual(js_divergence(p,p),0)
        self.assertGreater(js_divergence(p,p[::-1]),0)


if __name__=="__main__":unittest.main()
