"""Focused boundary and coverage checks for the production global-block policy."""
import unittest
import numpy as np
import split_policy as sp


class PolicyTests(unittest.TestCase):
    def test_closed_flow_end_crosses_boundary(self):
        start=np.array([0,100000,200000,300000])
        first=start+1
        end=np.array([199999,200000,220000,320000])
        kept,reasons,_=sp.exclude_boundaries(start,first,end,np.array([0,0,1,2],np.uint8),0)
        self.assertEqual(kept.tolist(),[0,3,1,2])
        self.assertEqual(reasons.tolist(),[0,2,0,0])
        self.assertEqual(int(sp.interval_overlap(first,end,start,kept).sum()),0)

    def test_single_block_type_stays_train_and_seed_is_repeatable(self):
        times=np.array([b*900000+p*100000 for b in range(20) for p in (2,3,4)])
        start=np.tile(times,3)
        source=np.repeat(np.arange(3),len(times))
        first=start+1
        end=start+150000
        positives=np.zeros(len(start),np.int64)
        cells={}
        for j in range(len(times)):
            index=2*len(times)+j
            name="Rare" if j//3==9 else "Common"
            cells[(index,name)]=1
            positives[index]=1
        a=sp.split(start,first,end,source,positives,cells)
        b=sp.split(start,first,end,source,positives,cells)
        np.testing.assert_array_equal(a[0],b[0])
        notes=a[-1]
        rare=next(r for r in notes["coverage"] if r["attack_type"]=="Rare")
        self.assertGreater(rare["train_sequences"],0)
        self.assertEqual((rare["validation_sequences"],rare["test_sequences"]),(0,0))
        self.assertEqual(notes["shared_global_100s"],0)
        for t in np.unique(start):
            self.assertLessEqual(len(set(a[0][start==t])-{3}),1)
        with self.assertRaises(ValueError):
            sp.split(np.array([0]),np.array([1]),np.array([2]),np.array([0]),np.array([1]),{(0,"A"):1})


if __name__=="__main__":
    unittest.main()
