"""Check ungated distances, equal coverage, missing support and access ordering."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import evidence_profile as ep
from . import evidence_profile_diagnostic as d


class DiagnosticChecks(unittest.TestCase):
    def test_dense_distances_reproduce_original_support(self):
        rng=np.random.default_rng(15)
        y=np.repeat(np.arange(8),4)
        bank=rng.normal(size=(32,8,3))
        p=np.eye(8)[y]*.99+.01/8
        proposed=np.tile(np.arange(8),8)
        query=rng.normal(size=(64,8,3))*10
        for i in range(32): query[i,proposed[i]]=bank[4*proposed[i],proposed[i]]
        # Class7 loses support, despite exact-query anchor matches.
        p[y==7]=1/8
        config={'scale_floor':.05,'minimum_anchors':3,'anchor_confidence':.5,'radius_floor_squared':.0025}
        expected,_=ep.support(query,bank,y,p,proposed,config)
        distance,inside,supported,_=d.dense_support(query,bank,y,p,proposed,config)
        actual=np.where(inside,np.exp(-distance),0)
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
        self.assertGreater(np.sum(actual>0),0)
        self.assertTrue(np.isinf(distance[proposed==7]).all())
        self.assertFalse(supported[proposed==7].any())

    def test_first_failure_is_disjoint_and_preserves_order(self):
        agree=np.array([False,True,True,True,True])
        base=np.array([False,False,True,True,True])
        supported=np.array([True,True,False,True,True])
        inside=np.array([True,True,False,False,True])
        np.testing.assert_array_equal(d.first_failure(agree,base,supported,inside),np.arange(5))

    def test_class_and_confidence_quotas_with_stable_ties(self):
        pred=np.repeat(np.arange(8),20)
        bins=np.tile(np.repeat(np.arange(5),4),8)
        domain=np.ones(160,dtype=bool)
        chosen=d.matched_pick(np.zeros(160),pred,domain,.25,bins)
        np.testing.assert_array_equal(chosen,np.arange(0,160,4))
        reverse=d.matched_pick(-np.arange(160),pred,domain,.25,bins)
        np.testing.assert_array_equal(np.bincount((pred*5+bins)[chosen]),np.bincount((pred*5+bins)[reverse]))
        for fraction in d.FRACTIONS:
            one=d.matched_pick(np.arange(160),pred,domain,fraction)
            two=d.matched_pick(-np.arange(160),pred,domain,fraction)
            np.testing.assert_array_equal(np.bincount(pred[one]),np.bincount(pred[two]))
        np.testing.assert_array_equal(d.matched_pick(np.arange(160),pred,domain,1),np.arange(160))

    def test_unshared_missing_scores_cannot_silently_rank(self):
        with self.assertRaisesRegex(ValueError,'same supported domain'):
            d.matched_pick(np.array([np.inf,0]),np.array([0,0]),np.ones(2,dtype=bool),.25)
        chosen=d.matched_pick(np.array([np.inf,0]),np.array([0,0]),np.array([False,True]),.25)
        np.testing.assert_array_equal(chosen,[1])

    def test_all_rankers_keep_identical_counts_and_random_is_reproducible(self):
        rng=np.random.default_rng(11)
        n=96
        scores={'pred':np.arange(n)%8,'confidence':rng.uniform(.2,1,n),
                'profile_distance':rng.uniform(0,5,n),'embedding_distance':rng.uniform(0,5,n),
                'agree':np.ones(n,dtype=bool),'shared_support':np.arange(n)%7!=0,
                'base':np.arange(n)%3==0}
        a=d.selections(scores,11)
        self.assertEqual(a,d.selections(scores,11))
        self.assertEqual(len(a),36)
        for domain in ('class_quota','frozen_gate','confidence_matched'):
            for fraction in ({r['fraction'] for r in a if r['domain']==domain}):
                rows=[r for r in a if r['domain']==domain and r['fraction']==fraction]
                self.assertEqual(len({len(r['rows']) for r in rows}),1)

    def test_empty_pool_has_undefined_error(self):
        row=d.measure(np.array([],dtype=np.int64),np.array([0]),np.array([0]))
        self.assertEqual(row,{'retained':0,'errors':0,'error_rate':None})

    def test_oracle_is_blocked_until_scores_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'status.json').write_text(json.dumps({'phase':'prepared'}))
            with patch.object(d,'frozen',return_value={}),patch.object(np,'load',side_effect=AssertionError('Oracle touched')) as load:
                with self.assertRaisesRegex(ValueError,'Freeze all scores'):
                    d.analyze(path)
                load.assert_not_called()
            self.assertFalse((path/'oracle_access.json').exists())


if __name__=='__main__':
    unittest.main()
