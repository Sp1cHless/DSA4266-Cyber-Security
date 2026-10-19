"""Bucket embedding pretraining -> frozen encoding -> temporal AE training.

The real-only baseline and augmented temporal model share the SAME frozen bucket
encoder and scaler. No CUDA job is launched merely by importing this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import FIELDS,Originals,Donors,fingerprint,metadata,read_json
from .models import BucketAE,TemporalAE,bucket_loss,sequence_loss,superimpose


def device_for(args):
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no CPU fallback")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    return torch.device(args.device)


def fresh(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise FileExistsError(path)
    path.mkdir(parents=True,exist_ok=True)
    return path


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def config(args,extra=None):
    return {**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},**(extra or {})}


def bucket_batch(store,positions,device):
    x=torch.from_numpy(np.array(store.arrays["numeric"].reshape(-1,82)[positions],copy=True)).to(device)
    hist={f:torch.from_numpy(np.array(store.arrays["hist_"+f].reshape(-1,store.schema["category_slots"][f])[positions],copy=True)).to(device)
          for f in FIELDS}
    return x,hist


def train_bucket(args):
    device=device_for(args);out=fresh(args.out)
    train=Originals(args.data,"train");val=Originals(args.data,"validation")
    rng=np.random.default_rng(args.seed)
    train_positions=np.flatnonzero(train.arrays["mask"].reshape(-1))
    val_positions=np.flatnonzero(val.arrays["mask"].reshape(-1))
    if not len(train_positions) or not len(val_positions):raise ValueError("Empty bucket pool")
    if len(val_positions)>args.val_limit:val_positions=rng.choice(val_positions,args.val_limit,replace=False)
    model_args=dict(slots=train.schema["category_slots"],category_dim=args.category_dim,bucket_dim=args.bucket_dim)
    model=BucketAE(**model_args).to(device)
    shutil.copy2(Path(args.data)/"schema.json",out/"schema.json")
    shutil.copy2(Path(args.data)/"scaler.json",out/"scaler.json")
    optim=torch.optim.Adam(model.parameters(),lr=args.lr)
    dump(out/"config.json",config(args,dict(model=model_args,data_fingerprint=fingerprint(args.data),
             population="real training active buckets only; no augmentation",test_used=False)))
    np.save(out/"validation_bucket_positions.npy",val_positions)
    best=float("inf");history=[]
    @torch.no_grad()
    def validate():
        model.eval();total=0.;count=0
        for start in range(0,len(val_positions),args.batch_size):
            x,h=bucket_batch(val,val_positions[start:start+args.batch_size],device)
            pred,logits=model(x,h);total+=float(bucket_loss(x,h,pred,logits))*len(x);count+=len(x)
        return total/count
    initial=validate();running=[];begun=time.monotonic()
    for step in range(1,args.steps+1):
        model.train();positions=rng.choice(train_positions,args.batch_size,replace=True)
        x,h=bucket_batch(train,positions,device);pred,logits=model(x,h)
        loss=bucket_loss(x,h,pred,logits)
        if not torch.isfinite(loss):raise ValueError("Nonfinite bucket loss")
        optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step()
        running.append(float(loss.detach().cpu()))
        if step%args.validate_every==0 or step==args.steps:
            value=validate();history.append(dict(step=step,train_loss=float(np.mean(running)),validation_loss=value))
            running=[]
            checkpoint=dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            model_args=model_args,data_fingerprint=fingerprint(args.data),step=step)
            torch.save(checkpoint,out/"last.pt")
            if value<best:best=value;torch.save(checkpoint,out/"best.pt")
            print(f"[bucket {step}/{args.steps}] validation={value:.6f}",flush=True)
    dump(out/"history.json",history)
    dump(out/"summary.json",dict(complete=True,initial_validation_loss=initial,best_validation_loss=best,
                                  seconds=time.monotonic()-begun,test_used=False))


@torch.no_grad()
def encode(args):
    device=device_for(args);out=fresh(args.out)
    saved=torch.load(args.checkpoint,map_location=device,weights_only=True)
    if saved["data_fingerprint"]!=fingerprint(args.data):raise ValueError("Bucket checkpoint belongs to another preparation")
    model=BucketAE(**saved["model_args"]).to(device)
    model.load_state_dict(saved["model_state"]);model.eval();model.requires_grad_(False)
    checkpoint_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    # Default excludes test. Encoding the test set later is an explicit command.
    dimension=saved["model_args"]["bucket_dim"]
    for split in args.splits:
        original=Originals(args.data,split);n=len(original.meta)
        folder=out/split;folder.mkdir()
        embeddings=np.lib.format.open_memmap(folder/"embeddings.npy",mode="w+",dtype=np.float32,shape=(n,10,dimension))
        mask=np.lib.format.open_memmap(folder/"mask.npy",mode="w+",dtype=np.float32,shape=(n,10))
        labels=np.lib.format.open_memmap(folder/"y_bucket.npy",mode="w+",dtype=np.int64,shape=(n,10))
        counts=np.lib.format.open_memmap(folder/"flow_counts.npy",mode="w+",dtype=np.int64,shape=(n,10))
        np.save(folder/"metadata.npy",original.meta)
        for start in range(0,n,args.batch_size):
            end=min(start+args.batch_size,n);a=original.take(slice(start,end))
            numeric=torch.from_numpy(a["numeric"]).to(device)
            hist={f:torch.from_numpy(a["hist_"+f]).to(device) for f in FIELDS}
            m=torch.from_numpy(a["mask"]).to(device)
            z=F.normalize(model.encode(numeric,hist),p=2,dim=-1,eps=1e-8)*m.unsqueeze(-1)
            embeddings[start:end]=z.cpu().numpy();mask[start:end]=a["mask"]
            labels[start:end]=a["y_bucket"];counts[start:end]=a["flow_counts"]
            if start%10000<args.batch_size:print(f"[encode {split}] {end:,}/{n:,}",flush=True)
        for a in (embeddings,mask,labels,counts):a.flush()
    shutil.copy2(args.data/"scaler.json",out/"scaler.json")
    shutil.copy2(args.data/"schema.json",out/"schema.json")
    shutil.copy2(args.checkpoint,out/"frozen_bucket_encoder.pt")
    dump(out/"manifest.json",dict(complete=True,version=3,splits=args.splits,bucket_dim=dimension,
             data_fingerprint=fingerprint(args.data),checkpoint_sha256=checkpoint_hash,
             frozen=True,real_originals_only=True))


class Encoded:
    def __init__(self,folder,split):
        folder=Path(folder);self.manifest=read_json(folder/"manifest.json")
        if not self.manifest.get("complete") or not self.manifest.get("frozen"):raise ValueError("Frozen complete encoding required")
        self.meta=np.load(folder/split/"metadata.npy",mmap_mode="r")
        self.arrays={k:np.load(folder/split/(k+".npy"),mmap_mode="r") for k in ("embeddings","mask","y_bucket","flow_counts")}
        self.scaler=read_json(folder/"scaler.json")

    def take(self,indices,device):
        return {k:torch.from_numpy(np.array(a[indices],copy=True)).to(device) for k,a in self.arrays.items()}

    def volume(self,counts,mask):
        mean=self.scaler["mean"][0];scale=self.scaler["scale"][0]
        return ((torch.log1p(counts.float())-mean)/scale)*mask


def mixed_batch(store,left,right,shifts,device):
    a=store.take(left,device);b=store.take(np.maximum(right,0),device)
    is_synthetic=torch.from_numpy((right>=0).astype(np.float32)).to(device)[:,None]
    # Real examples have an absent second component, not a duplicated donor.
    b["mask"]*=is_synthetic;b["y_bucket"]*=is_synthetic.long();b["flow_counts"]*=is_synthetic.long()
    z,mask,y,counts=superimpose(a["embeddings"],b["embeddings"],a["mask"],b["mask"],
                               a["y_bucket"],b["y_bucket"],a["flow_counts"],b["flow_counts"],
                               torch.from_numpy(shifts).to(device))
    x=torch.cat([z,store.volume(counts,mask).unsqueeze(-1)],dim=-1)
    return x,mask,y.max(dim=1).values


@torch.no_grad()
def evaluate_temporal(model,store,ids,batch_size,device,return_vectors=False):
    model.eval();errors=[];vectors=[]
    for start in range(0,len(ids),batch_size):
        indices=ids[start:start+batch_size];a=store.take(indices,device)
        x=torch.cat([a["embeddings"],store.volume(a["flow_counts"],a["mask"]).unsqueeze(-1)],dim=-1)
        errors.extend(sequence_loss(x,model(x,a["mask"]),a["mask"]).cpu().tolist())
        if return_vectors:vectors.append(model.encode(x,a["mask"]).cpu().numpy())
    if not np.isfinite(errors).all():raise ValueError("Nonfinite validation error")
    return np.array(errors),np.concatenate(vectors) if vectors else None


def train_temporal(args):
    device=device_for(args);out=fresh(args.out)
    train=Encoded(args.encoded,"train");val=Encoded(args.encoded,"validation")
    if train.manifest["data_fingerprint"]!=fingerprint(args.data):raise ValueError("Encoding/data mismatch")
    # Frozen bucket encoder is never part of this optimiser or these targets.
    donors=Donors(args.data,train.meta,args.seed)
    val_ids=np.arange(len(val.meta))
    if len(val_ids)>args.val_limit:val_ids=np.sort(np.random.default_rng(args.seed+17).choice(val_ids,args.val_limit,replace=False))
    model_args=dict(features=train.manifest["bucket_dim"]+1,width=args.width,latent=args.latent)
    model=TemporalAE(**model_args).to(device);optim=torch.optim.Adam(model.parameters(),lr=args.lr)
    dump(out/"config.json",config(args,dict(model=model_args,test_used=False,
             embedding_checkpoint_sha256=train.manifest["checkpoint_sha256"],
             objective="mixed-class AE on frozen vector-space mixtures",
             circular_shift=("disabled for real-only control" if args.real_only else
                             "disabled for zero-shift control" if getattr(args,"shift_mode","circular")=="none" else
                             "one donor within 10 buckets before addition"),l2_axis="embedding dimension of each bucket",
             P=len(donors.positive),generation_cycle_per_class=len(donors.positive))))
    shutil.copy2(Path(args.encoded)/"frozen_bucket_encoder.pt",out/"frozen_bucket_encoder.pt")
    shutil.copy2(Path(args.encoded)/"scaler.json",out/"scaler.json")
    shutil.copy2(Path(args.encoded)/"schema.json",out/"schema.json")
    np.save(out/"validation_ids.npy",val.meta["id"][val_ids])
    initial,_=evaluate_temporal(model,val,val_ids,args.batch_size,device)
    best=float("inf");history=[];global_step=0;begun=time.monotonic()
    with (out/"provenance.jsonl").open("w") as audit:
        for epoch in range(1,args.epochs+1):
            model.train();losses=[];group_sums={};group_counts={};started=time.monotonic()
            for step in range(1,args.steps_per_epoch+1):
                left,right,shifts,provenance=donors.batch(args.batch_size,args.real_only,getattr(args,"shift_mode","circular")=="circular")
                x,mask,y=mixed_batch(train,left,right,shifts,device)
                expected=np.array([p["label"] for p in provenance])
                if not np.array_equal(y.cpu().numpy(),expected):raise AssertionError("Donor-label mismatch")
                optim.zero_grad(set_to_none=True);errors=sequence_loss(x,model(x,mask),mask);loss=errors.mean()
                if not torch.isfinite(loss):raise ValueError("Nonfinite temporal loss")
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step()
                losses.append(float(loss.detach().cpu()));global_step+=1
                for p,error in zip(provenance,errors.detach().cpu().tolist()):
                    key=f"{'synthetic' if p['synthetic'] else 'real'}_{'malicious' if p['label'] else 'benign'}"
                    group_sums[key]=group_sums.get(key,0)+error;group_counts[key]=group_counts.get(key,0)+1
                    audit.write(json.dumps(dict(step=global_step,**p))+"\n")
                if step%args.log_every==0 or step==args.steps_per_epoch:
                    print(f"[temporal epoch {epoch} step {step}] loss={losses[-1]:.6f}",flush=True)
            audit.flush();validation,_=evaluate_temporal(model,val,val_ids,args.batch_size,device)
            row=dict(epoch=epoch,step=global_step,train_mse=float(np.mean(losses)),validation_mse=float(validation.mean()),seconds=time.monotonic()-started)
            labels=val.meta["label"][val_ids]
            for label,name in ((0,"benign"),(1,"malicious")):
                row["validation_"+name+"_mse"]=float(validation[labels==label].mean()) if np.any(labels==label) else None
            row.update({"train_"+k+"_mse":group_sums[k]/group_counts[k] for k in group_sums});history.append(row)
            with (out/"history.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            checkpoint=dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},model_args=model_args,
                            embedding_checkpoint_sha256=train.manifest["checkpoint_sha256"],epoch=epoch)
            torch.save(checkpoint,out/"last.pt")
            if row["validation_mse"]<best:best=row["validation_mse"];torch.save(checkpoint,out/"best.pt")
            print(f"[epoch {epoch}] train={row['train_mse']:.6f} val={row['validation_mse']:.6f}",flush=True)
    saved=torch.load(out/"best.pt",map_location=device,weights_only=True);model.load_state_dict(saved["model_state"])
    _,vectors=evaluate_temporal(model,val,val_ids,args.batch_size,device,return_vectors=True)
    folder=out/"projector_validation";folder.mkdir()
    np.savetxt(folder/"vectors.tsv",vectors,delimiter="\t",fmt="%.8g")
    with (folder/"metadata.tsv").open("w",newline="") as handle:
        writer=csv.writer(handle,delimiter="\t");writer.writerow(["id","label","primary_attack","source_ip","start_ms","active_buckets","split"])
        for r in val.meta[val_ids]:writer.writerow([int(r["id"]),int(r["label"]),str(r["attack"]),str(r["ip"]),int(r["start"]),int(r["active"]),"validation"])
    dump(out/"summary.json",dict(complete=True,initial_validation_mse=float(initial.mean()),best_validation_mse=best,
                                 steps=global_step,seconds=time.monotonic()-begun,test_used=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="stage",required=True)
    for name in ("bucket","encode","temporal"):
        p=sub.add_parser(name);p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
        p.add_argument("--device",choices=["cuda","cpu","mps"],default="cuda");p.add_argument("--seed",type=int,default=4266)
        p.add_argument("--cpu-threads",type=int,default=2);p.add_argument("--batch-size",type=int,default=512 if name=="bucket" else 256 if name=="encode" else 32)
        if name=="bucket":
            p.add_argument("--steps",type=int,default=5000);p.add_argument("--validate-every",type=int,default=200)
            p.add_argument("--category-dim",type=int,default=16);p.add_argument("--bucket-dim",type=int,default=64)
        if name=="encode":
            p.add_argument("--checkpoint",type=Path,required=True)
            p.add_argument("--splits",nargs="+",choices=["train","validation","test"],default=["train","validation"])
        if name=="temporal":
            p.add_argument("--encoded",type=Path,required=True);p.add_argument("--epochs",type=int,default=10)
            p.add_argument("--steps-per-epoch",type=int,default=200);p.add_argument("--width",type=int,default=32)
            p.add_argument("--latent",type=int,default=32);p.add_argument("--log-every",type=int,default=10)
            p.add_argument("--real-only",action="store_true")
            p.add_argument("--shift-mode",choices=["circular","none"],default="circular")
        if name!="encode":
            p.add_argument("--lr",type=float,default=.001);p.add_argument("--val-limit",type=int,default=4096)
    args=parser.parse_args()
    for key in ("batch_size","cpu_threads","steps","validate_every","category_dim","bucket_dim","epochs","steps_per_epoch","width","latent","log_every","val_limit","lr"):
        if hasattr(args,key) and getattr(args,key)<=0:parser.error(f"{key} must be positive")
    if args.stage=="temporal" and args.batch_size%4:parser.error("Temporal batch size must divide by four")
    {"bucket":train_bucket,"encode":encode,"temporal":train_temporal}[args.stage](args)


if __name__=="__main__":main()
