"""Small fixtures only: no full CIC scan and no remote/paid compute."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import prepare as preparation
from .prepare import prepare,raw_bucket,selected_distribution,import_csv
from .data import Originals,Donors,read_json
from .models import BucketAE,TemporalAE,bucket_loss,rotate,superimpose,sequence_loss
from .train import train_bucket,encode,train_temporal
from CIC_new import bucket_features as bf
from CIC_new.test_build_buckets import event,fixture_database


class VectorTests(unittest.TestCase):
    def test_original_csv_adapter(self):
        import pandas as pd
        import sqlite3
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row=event()
            values={name.upper():int(row[i]) for i,name in enumerate(bf.EVENT_COLUMNS)
                    if name not in ("raw_rowid","destination_id")}
            values.update(IPV4_SRC_ADDR="10.0.0.1",IPV4_DST_ADDR="10.0.0.2",Attack="Benign")
            source=root/"raw.csv";pd.DataFrame([values]).to_csv(source,index=False)
            import_csv(source,root/"raw.sqlite")
            conn=sqlite3.connect(root/"raw.sqlite")
            try:
                self.assertEqual(conn.execute("SELECT ipv4_src_addr,label,l7_proto FROM network_flows").fetchone(),("10.0.0.1",0,5))
            finally:conn.close()

    def test_histograms_before_vocabulary_and_unknown(self):
        events=np.stack([event(1000,l7=5),event(2000,l7=999,label=1),event(12000,protocol=17,l7=5)])
        values,hist=raw_bucket(events)
        self.assertEqual(values["numeric"].shape,(10,82))
        self.assertEqual(values["flow_counts"].sum(),3)
        self.assertEqual(values["y_sequence"],1)
        h=selected_distribution(hist["l7"],np.array([0]),[5],1)
        np.testing.assert_allclose(h[0,0],[.5,.5])
        np.testing.assert_allclose(h.sum(axis=-1),values["mask"][None,:])

    def test_circular_shift_and_vector_addition(self):
        a=torch.zeros(1,10,2);b=torch.zeros_like(a)
        a[0,0]=torch.tensor([1.,0.]);b[0,9]=torch.tensor([0.,1.])
        ma=torch.zeros(1,10);mb=ma.clone();ma[0,0]=1;mb[0,9]=1
        ya=torch.zeros(1,10,dtype=torch.long);yb=ya.clone();yb[0,9]=1
        ca=ya.clone();cb=ya.clone();ca[0,0]=2;cb[0,9]=3
        z,m,y,c=superimpose(a,b,ma,mb,ya,yb,ca,cb,torch.tensor([1]))
        np.testing.assert_allclose(z[0,0].numpy(),np.array([1.,1.])/np.sqrt(2),rtol=1e-6)
        self.assertEqual(int(c.sum()),5)
        self.assertEqual(int(y.sum()),1)
        self.assertEqual(int(y[0,0]),1)  # Last position wraps to first.
        self.assertEqual(float(m.sum()),1)
        self.assertTrue(torch.all(z[:,1:]==0))
        self.assertTrue(torch.isfinite(z).all())
        np.testing.assert_equal(rotate(c,torch.tensor([3])).numpy(),np.roll(c.numpy(),3,axis=1))

    def test_mixed_batch_multiple_shifts_and_cancellation(self):
        rng=np.random.default_rng(2)
        a=torch.tensor(rng.normal(size=(3,10,4)),dtype=torch.float32);b=-a.clone()
        m=torch.ones(3,10);labels=torch.zeros(3,10,dtype=torch.long);counts=torch.ones_like(labels)
        z,mask,y,c=superimpose(a,b,m,m,labels,labels,counts,counts,torch.tensor([0,1,9]))
        self.assertTrue(torch.all(z[0]==0))
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.all(mask==1))  # Zero vector from cancellation is not an empty bucket.
        self.assertEqual(int(c.sum()),60)

    def test_embedding_gradients_and_unknown_zero(self):
        torch.manual_seed(1)
        model=BucketAE(dict(protocol=4,l7=5,service=3),category_dim=4,bucket_dim=8)
        x=torch.randn(12,82)
        hist={f:torch.nn.functional.one_hot(torch.ones(12,dtype=torch.long),n).float()
              for f,n in dict(protocol=4,l7=5,service=3).items()}
        pred,logits=model(x,hist);loss=bucket_loss(x,hist,pred,logits);loss.backward()
        for f in hist:
            self.assertTrue(torch.all(model.categories[f].weight[0]==0))
            self.assertGreater(float(model.categories[f].weight.grad[1].abs().sum()),0)
        temporal=TemporalAE(features=9,width=4,latent=3)
        inp=torch.randn(2,10,9);mask=torch.ones(2,10)
        error=sequence_loss(inp,temporal(inp,mask),mask).mean();error.backward()
        self.assertEqual(temporal.encode(inp,mask).shape,(2,3))

    def test_full_small_pipeline_and_frozen_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);db=root/"fixture.sqlite";fixture_database(db)
            original_hash=hashlib.sha256(db.read_bytes()).hexdigest()
            data=root/"prepared";calls=[]
            aggregate_originals=preparation.aggregate_originals
            split_originals=preparation.legacy.split_originals
            def aggregate_first(*args,**kwargs):
                calls.append("bucket");return aggregate_originals(*args,**kwargs)
            def split_after(*args,**kwargs):
                calls.append("split");return split_originals(*args,**kwargs)
            with patch.object(preparation,"aggregate_originals",side_effect=aggregate_first),patch.object(preparation.legacy,"split_originals",side_effect=split_after):
                prepare(data,db=db,shard_size=17)
            self.assertEqual(calls,["bucket","split"])
            manifest=read_json(data/"manifest.json")
            self.assertEqual(manifest["order"][:3],["sequence_construction","bucket_aggregation","split"])
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(),original_hash)
            self.assertFalse(manifest["augmentation_materialized"])
            self.assertNotIn(999,read_json(data/"schema.json")["vocabularies"]["l7"])
            train=Originals(data,"train");val=Originals(data,"validation")
            active=train.arrays["mask"].astype(bool)
            np.testing.assert_allclose(train.arrays["numeric"][active].mean(axis=0),0,atol=2e-5)
            self.assertFalse(set(train.meta["id"])&set(val.meta["id"]))
            sampler=Donors(data,train.meta)
            left,right,shifts,provenance=sampler.batch(32)
            self.assertEqual(sum(p["synthetic"] for p in provenance),16)
            self.assertEqual(sum(p["label"] for p in provenance),16)
            for a,b,p in zip(left,right,provenance):
                if p["synthetic"]:
                    self.assertNotEqual(train.meta[a]["ip"],train.meta[b]["ip"])
                    self.assertEqual(train.meta[a]["label"],0)
            sampler=Donors(data,train.meta)
            _,_,_,cycle=sampler.batch(4*len(sampler.positive))
            self.assertEqual(sum(p["synthetic"] and p["label"] for p in cycle),len(sampler.positive))
            self.assertEqual(sum(p["synthetic"] and not p["label"] for p in cycle),len(sampler.positive))
            self.assertEqual({p["generation_cycle"] for p in cycle if p["synthetic"]},{0})
            shifted=Donors(data,train.meta,seed=12).batch(32,circular=True)
            unshifted=Donors(data,train.meta,seed=12).batch(32,circular=False)
            np.testing.assert_array_equal(shifted[0],unshifted[0])
            np.testing.assert_array_equal(shifted[1],unshifted[1])
            self.assertTrue(np.all(unshifted[2]==0))
            bucket=root/"bucket"
            base=dict(data=data,device="cpu",seed=1,cpu_threads=1,batch_size=8)
            train_bucket(SimpleNamespace(**base,out=bucket,steps=2,validate_every=1,val_limit=16,
                         category_dim=4,bucket_dim=8,lr=.001,stage="bucket"))
            saved=bucket/"best.pt";before=hashlib.sha256(saved.read_bytes()).hexdigest()
            encoded=root/"encoded"
            encode(SimpleNamespace(**base,out=encoded,checkpoint=saved,splits=["train","validation"],stage="encode"))
            temporal=root/"temporal"
            train_temporal(SimpleNamespace(**base,out=temporal,encoded=encoded,epochs=1,steps_per_epoch=2,
                           width=4,latent=3,log_every=1,val_limit=16,lr=.001,real_only=False,stage="temporal"))
            self.assertEqual(hashlib.sha256(saved.read_bytes()).hexdigest(),before)
            self.assertFalse(read_json(temporal/"summary.json")["test_used"])
            vectors=np.loadtxt(temporal/"projector_validation/vectors.tsv",delimiter="\t")
            self.assertEqual(vectors.shape,(16,3))
            self.assertEqual(read_json(temporal/"config.json")["model"]["features"],9)
            self.assertEqual(len((temporal/"projector_validation/metadata.tsv").read_text().splitlines()),17)
            # Both stages should have touched training/validation only.
            self.assertFalse((encoded/"test").exists())
            real_only=root/"real_only"
            train_temporal(SimpleNamespace(**base,out=real_only,encoded=encoded,epochs=1,steps_per_epoch=2,
                           width=4,latent=3,log_every=1,val_limit=16,lr=.001,real_only=True,stage="temporal"))
            np.testing.assert_array_equal(np.load(temporal/"validation_ids.npy"),np.load(real_only/"validation_ids.npy"))
            self.assertEqual(read_json(temporal/"config.json")["embedding_checkpoint_sha256"],
                             read_json(real_only/"config.json")["embedding_checkpoint_sha256"])


if __name__=="__main__":unittest.main()
