"""Architecture/input-control properties and reproducible tiny fits."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import cross_model_motion as e
from . import dataset_validation as audit


class Controls(unittest.TestCase):
    def test_models_counts_gradients_and_set_invariance(self):
        torch.set_num_threads(2)
        for family in e.COUNTS:
            torch.manual_seed(12);model=e.build_model(family)
            x=torch.randn(3,32,146);logits=model(x)
            self.assertEqual(tuple(logits.shape),(3,8))
            self.assertEqual(sum(p.numel() for p in model.parameters()),e.COUNTS[family])
            torch.nn.functional.cross_entropy(logits,torch.tensor([0,4,7])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            if family in ("feature_set","local_set"):
                model.eval()
                with torch.inference_mode():torch.testing.assert_close(model(x),model(x[:,torch.randperm(32)]),rtol=1e-5,atol=1e-6)

    def test_causal_stack_does_not_read_future(self):
        torch.manual_seed(2);model=e.build_model("causal_cnn").eval()
        x=torch.randn(2,146,32);altered=x.clone();altered[:,:,16:]+=100
        with torch.inference_mode():torch.testing.assert_close(model.features(x)[:,:,:16],model.features(altered)[:,:,:16],rtol=0,atol=0)

    def test_input_masks_do_not_mutate_shared_features(self):
        x=np.ones((2,32,146),dtype=np.float32);before=x.copy()
        local=e.inputs(x,"local_set");clock=e.inputs(x,"feature_set","no_elapsed")
        self.assertFalse(local[:,:,63:66].any());self.assertFalse(local[:,:,130:133].any())
        self.assertTrue(local[:,:,67:130].all());self.assertFalse(clock[:,:,144].any());self.assertTrue(clock[:,:,145].all())
        np.testing.assert_array_equal(x,before)

    def test_tiny_fit_pairing_and_strict_loading(self):
        torch.set_num_threads(2);rng=np.random.default_rng(12)
        arrays={"train_x":rng.normal(size=(40,32,146)).astype(np.float32),"train_y":np.tile(np.arange(8),5),
            "validation_x":rng.normal(size=(8,32,146)).astype(np.float32),"validation_y":np.arange(8)}
        config={"optimizer_steps":2,"validation_every_steps":1,"batch_size":16,"learning_rate":.001}
        first=[]
        for family in ("cnn","feature_set"):
            a,h,s=e.fit(family,arrays,11,config);b,h2,s2=e.fit(family,arrays,11,config)
            self.assertEqual(h,h2);self.assertEqual(s["initial_state_sha256"],s2["initial_state_sha256"]);first.append(s["batch_indices_sha256"])
        self.assertEqual(first[0],first[1])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"wrong.pt"
            checkpoint={"experiment_id":"011_cross_model_motion","schema_version":3,"feature_version":2,
                "features":146,"frames":32,"labels":list(e.motion.previous.v2.LABELS),
                "command_contract":e.motion.previous.training.collection.contract("three_shapes"),
                "family":"feature_set","mode":"camera_motion","variant":"standard","state_dict":b.state_dict()}
            torch.save(checkpoint,path);loaded,_=e.load_model(path)
            with torch.inference_mode():torch.testing.assert_close(loaded(torch.from_numpy(arrays["validation_x"])),b(torch.from_numpy(arrays["validation_x"])))
            checkpoint["feature_version"]=1;torch.save(checkpoint,path)
            with self.assertRaisesRegex(ValueError,"Incompatible"):e.load_model(path)

    def test_audit_excludes_training_sessions_old_up_and_no_false_fresh_test(self):
        base={"split":"validation","raw_content_sha256":"new","session":"new","label":"STOP","cohort":"legacy"}
        self.assertEqual(audit.classify(base,set(),{"new"}),"session_in_current_training")
        self.assertEqual(audit.classify({**base,"label":"UP"},set(),set()),"old_up_definition")
        self.assertEqual(audit.classify({**base,"split":"test"},set(),set()),"historical_common_label_only")


if __name__=="__main__":unittest.main()
