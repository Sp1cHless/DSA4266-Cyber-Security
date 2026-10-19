"""Export fixed train/validation/test arrays from frozen sequence embeddings.

This is NumPy-only data processing. It does not define or train an AE, classifier,
CNN or LSTM. Only the training split receives synthetic samples.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from .augmentation import Encoded,mixed_batch
from .dataset import Donors,fingerprint,read_json


class Writer:
    def __init__(self,folder,size):
        self.folder=folder;folder.mkdir(parents=True);self.size=size
        self.rows=[];self.part=0;self.counts=Counter()

    def append(self,x,mask,label,bucket_label,counts,sid,kind,left,right=-1,shift=0,left_ip="",right_ip="",attack="Benign"):
        if not np.isfinite(x).all():raise ValueError("Nonfinite model input")
        if np.any(x[mask==0]!=0):raise AssertionError("Nonzero empty bucket")
        if int(np.max(bucket_label))!=label:raise AssertionError("Bucket/sequence label mismatch")
        self.rows.append((x,mask,label,bucket_label,counts,sid,kind,left,right,shift,left_ip,right_ip,attack))
        self.counts[kind]+=1
        if len(self.rows)>=self.size:self.flush()

    def flush(self):
        if not self.rows:return
        rows=list(zip(*self.rows))
        np.savez_compressed(self.folder/f"part_{self.part:06d}.npz",
            X=np.stack(rows[0]).astype(np.float32),mask=np.stack(rows[1]).astype(np.uint8),
            y_sequence=np.array(rows[2],dtype=np.uint8),y_bucket=np.stack(rows[3]).astype(np.uint8),
            flow_counts=np.stack(rows[4]).astype(np.int64),sample_id=np.array(rows[5],dtype=np.int64),
            sample_type=np.array(rows[6]),left_id=np.array(rows[7],dtype=np.int64),
            right_id=np.array(rows[8],dtype=np.int64),right_shift_buckets=np.array(rows[9],dtype=np.int8),
            left_ip=np.array(rows[10]),right_ip=np.array(rows[11]),primary_attack=np.array(rows[12]))
        self.part+=1;self.rows.clear()


def export_dataset(data,encoded,out,seed=4266,shard_size=2048,shift_mode="circular",originals_only=False):
    data=Path(data).resolve();encoded=Path(encoded).resolve();out=Path(out).resolve()
    if shift_mode not in ("circular","none"):raise ValueError("Unknown shift mode")
    if shard_size<1:raise ValueError("Positive shard size required")
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    stores={s:Encoded(encoded,s) for s in ("train","validation","test")}
    identity=fingerprint(data)
    if any(s.manifest["data_fingerprint"]!=identity for s in stores.values()):raise ValueError("Encoding/data mismatch")
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(complete=False,format="model-independent-sequence-shards",data_fingerprint=identity,
                  embedding_checkpoint_sha256=stores["train"].manifest["checkpoint_sha256"],seed=seed,
                  feature_width=stores["train"].manifest["bucket_dim"]+1,timesteps=10,
                  shift_mode=shift_mode,originals_only=originals_only)
    path=out/"manifest.json";path.write_text(json.dumps(manifest,indent=2))
    writers={s:Writer(out/s,shard_size) for s in stores}
    for split,store in stores.items():
        for start in range(0,len(store.meta),128):
            indices=np.arange(start,min(start+128,len(store.meta)))
            x,mask,y,y_bucket,counts=mixed_batch(store,indices,np.full(len(indices),-1),np.zeros(len(indices),dtype=np.int64),return_details=True)
            for j,i in enumerate(indices):
                sid=int(store.meta[i]["id"]);label=int(y[j]);kind="real_malicious" if label else "real_benign"
                writers[split].append(x[j],mask[j],label,y_bucket[j],counts[j],sid,kind,sid,
                    left_ip=str(store.meta[i]["ip"]),attack=str(store.meta[i]["attack"]))
        print(f"[export {split}] {len(store.meta):,} real sequences",flush=True)
    train=stores["train"];P=int(train.meta["label"].sum())
    if not originals_only:
        donors=Donors(data,train.meta,seed)
        conn=sqlite3.connect(f"file:{data/'manifest.sqlite'}?mode=ro",uri=True)
        try:next_id=int(conn.execute("SELECT MAX(sequence_id) FROM original_sequences").fetchone()[0])+1
        finally:conn.close()
        remaining=P
        while remaining:
            per_class=min(8,remaining)
            left,right,shifts,provenance=donors.batch(4*per_class,circular=shift_mode=="circular")
            keep=np.flatnonzero(right>=0)
            x,mask,y,y_bucket,counts=mixed_batch(train,left[keep],right[keep],shifts[keep],return_details=True)
            for j,i in enumerate(keep):
                p=provenance[i];label=int(y[j])
                if label!=p["label"]:raise AssertionError("Synthetic label mismatch")
                kind="synthetic_malicious" if label else "synthetic_benign"
                writers["train"].append(x[j],mask[j],label,y_bucket[j],counts[j],next_id,kind,
                    p["left_id"],p["right_id"],int(shifts[i]),p["left_ip"],p["right_ip"],
                    str(train.meta[right[i]]["attack"]) if label else "Benign")
                next_id+=1
            remaining-=per_class
    for w in writers.values():w.flush()
    expected=0 if originals_only else P
    if writers["train"].counts["synthetic_malicious"]!=expected or writers["train"].counts["synthetic_benign"]!=expected:
        raise AssertionError("Augmentation budget mismatch")
    for split in ("validation","test"):
        if any(k.startswith("synthetic") for k in writers[split].counts):raise AssertionError("Synthetic held-out example")
    for name in ("schema.json","scaler.json","frozen_bucket_encoder.pt"):
        if (encoded/name).exists():shutil.copy2(encoded/name,out/name)
    manifest.update(complete=True,P=P,counts={s:dict(w.counts) for s,w in writers.items()},
                    augmentation_per_class=expected,total_samples={s:sum(w.counts.values()) for s,w in writers.items()})
    path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


def iter_model_shards(folder,split):
    """Yield model-independent X/mask/labels/provenance dictionaries."""
    folder=Path(folder);manifest=read_json(folder/"manifest.json")
    if not manifest.get("complete") or manifest.get("format")!="model-independent-sequence-shards":raise ValueError("Completed model-input dataset required")
    if split not in ("train","validation","test"):raise ValueError(split)
    for path in sorted((folder/split).glob("part_*.npz")):
        with np.load(path,allow_pickle=False) as a:yield {k:a[k] for k in a.files}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--encoded",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--seed",type=int,default=4266)
    p.add_argument("--shard-size",type=int,default=2048);p.add_argument("--originals-only",action="store_true")
    p.add_argument("--shift-mode",choices=["circular","none"],default="circular")
    args=p.parse_args();export_dataset(args.data,args.encoded,args.out,args.seed,args.shard_size,args.shift_mode,args.originals_only)
