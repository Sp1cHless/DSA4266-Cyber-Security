"""Export fixed train/validation/test arrays from frozen sequence embeddings.

This is NumPy-only data processing. It does not define or train an AE, classifier,
CNN or LSTM. Only the training split receives synthetic samples.
"""
import argparse
from collections import Counter
import hashlib
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


def export_fixed_plan(data,train,writer,path,shift_mode):
    """Use exact existing donor IDs without changing vector-space composition."""
    with np.load(path,allow_pickle=False) as archive:
        plan={k:archive[k] for k in archive.files}
    required=("sample_id","sample_type","left_id","right_id","right_shift_buckets")
    P=int(train.meta["label"].sum());n=2*P
    if any(k not in plan or plan[k].shape!=(n,) for k in required):
        raise ValueError("A fixed donor plan must contain exactly 2P rows")
    for kind in ("synthetic_benign","synthetic_malicious"):
        if np.count_nonzero(plan["sample_type"]==kind)!=P:raise ValueError("Plan budget is not P per synthetic class")
    connection=sqlite3.connect(f"file:{data/'manifest.sqlite'}?mode=ro",uri=True)
    try:maximum=connection.execute("SELECT MAX(sequence_id) FROM original_sequences").fetchone()[0]
    finally:connection.close()
    if not np.array_equal(plan["sample_id"],np.arange(maximum+1,maximum+1+n)):
        raise ValueError("Plan synthetic IDs do not match original sequence ownership")
    left=np.searchsorted(train.meta["id"],plan["left_id"])
    right=np.searchsorted(train.meta["id"],plan["right_id"])
    if np.any(left>=len(train.meta)) or np.any(right>=len(train.meta)):
        raise ValueError("Plan donor outside real train")
    if not np.array_equal(train.meta["id"][left],plan["left_id"]) or not np.array_equal(train.meta["id"][right],plan["right_id"]):
        raise ValueError("Plan donor outside real train")
    if np.any(train.meta["label"][left]) or np.any(train.meta["ip"][left]==train.meta["ip"][right]):
        raise ValueError("Plan needs a benign first donor and different source IPs")
    malicious=plan["sample_type"]=="synthetic_malicious"
    if not np.array_equal(train.meta["label"][right].astype(bool),malicious):
        raise ValueError("Plan class does not match second donor")
    shifts=plan["right_shift_buckets"]
    if np.any(shifts<0) or np.any(shifts>=10) or (shift_mode=="none" and np.any(shifts)):
        raise ValueError("Plan shifts do not match requested mode")
    for start in range(0,n,128):
        end=min(start+128,n)
        x,mask,y,y_bucket,counts=mixed_batch(train,left[start:end],right[start:end],shifts[start:end],return_details=True)
        for j,i in enumerate(range(start,end)):
            writer.append(x[j],mask[j],int(y[j]),y_bucket[j],counts[j],int(plan["sample_id"][i]),
                str(plan["sample_type"][i]),int(plan["left_id"][i]),int(plan["right_id"][i]),int(shifts[i]),
                str(train.meta[left[i]]["ip"]),str(train.meta[right[i]]["ip"]),
                str(train.meta[right[i]]["attack"]) if malicious[i] else "Benign")


def export_dataset(data,encoded,out,seed=4266,shard_size=2048,shift_mode="circular",originals_only=False,donor_plan=None):
    data=Path(data).resolve();encoded=Path(encoded).resolve();out=Path(out).resolve()
    if shift_mode not in ("circular","none"):raise ValueError("Unknown shift mode")
    if shard_size<1:raise ValueError("Positive shard size required")
    if originals_only and donor_plan is not None:raise ValueError("An originals-only export cannot use a donor plan")
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
    if donor_plan is not None:
        donor_plan=Path(donor_plan).resolve()
        export_fixed_plan(data,train,writers["train"],donor_plan,shift_mode)
        shutil.copy2(donor_plan,out/"donor_plan.npz")
        manifest.update(donor_selection="external frozen plan; seed does not resample donors",
                        donor_plan_source=str(donor_plan),
                        donor_plan_sha256=hashlib.sha256(donor_plan.read_bytes()).hexdigest())
    elif not originals_only:
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
    p.add_argument("--donor-plan",type=Path,help="Reuse exact synthetic IDs and training donor pairs from another exported dataset")
    args=p.parse_args();export_dataset(args.data,args.encoded,args.out,args.seed,args.shard_size,args.shift_mode,args.originals_only,args.donor_plan)
