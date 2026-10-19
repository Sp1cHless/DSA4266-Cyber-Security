"""Train ONLY the optional temporal AE on an already exported sequence dataset.

No raw-data processing, embedding fitting or synthetic sample construction occurs
in this script. CNN/LSTM/classifier experiments can consume the same dataset.
"""
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from CIC_new.preprocessing.dataset import ModelInputs
from CIC_new.training_utils import device_for,fresh,dump,config,validate_arguments
from .autoencoder import TemporalAE,sequence_loss


@torch.no_grad()
def evaluate_temporal(model,store,ids,batch_size,device,return_vectors=False):
    model.eval();errors=[];vectors=[]
    for start in range(0,len(ids),batch_size):
        a=store.take(ids[start:start+batch_size])
        x=torch.from_numpy(a["X"]).to(device);mask=torch.from_numpy(a["mask"]).to(device)
        errors.extend(sequence_loss(x,model(x,mask),mask).cpu().tolist())
        if return_vectors:vectors.append(model.encode(x,mask).cpu().numpy())
    if not np.isfinite(errors).all():raise ValueError("Nonfinite validation error")
    return np.array(errors),np.concatenate(vectors) if vectors else None


def train_temporal(args):
    device=device_for(args);out=fresh(args.out)
    train=ModelInputs(args.dataset,"train");val=ModelInputs(args.dataset,"validation")
    val_ids=np.arange(len(val))
    if len(val_ids)>args.val_limit:
        val_ids=np.sort(np.random.default_rng(args.seed+17).choice(val_ids,args.val_limit,replace=False))
    model_args=dict(features=train.manifest["feature_width"],width=args.width,latent=args.latent,
                    timesteps=train.manifest["timesteps"])
    model=TemporalAE(**model_args).to(device);optim=torch.optim.Adam(model.parameters(),lr=args.lr)
    dump(out/"config.json",config(args,dict(model=model_args,test_used=False,
         objective="mixed-class reconstruction of already prepared sequence inputs",
         embedding_checkpoint_sha256=train.manifest["embedding_checkpoint_sha256"],
         dataset_augmentation=train.manifest["shift_mode"])))
    shutil.copy2(Path(args.dataset)/"manifest.json",out/"dataset_manifest.json")
    for name in ("frozen_bucket_encoder.pt","scaler.json","schema.json"):
        source=Path(args.dataset)/name
        if source.exists():shutil.copy2(source,out/name)
    np.save(out/"validation_ids.npy",val.arrays["sample_id"][val_ids])
    initial,_=evaluate_temporal(model,val,val_ids,args.batch_size,device)
    best=float("inf");history=[];global_step=0;begun=time.monotonic()
    generator=train.batches(args.batch_size,args.epochs*args.steps_per_epoch,args.seed,args.real_only)
    with (out/"sample_draws.jsonl").open("w") as audit:
        for epoch in range(1,args.epochs+1):
            model.train();losses=[];group_sums={};group_counts={};started=time.monotonic()
            for step in range(1,args.steps_per_epoch+1):
                batch=next(generator)
                x=torch.from_numpy(batch["X"]).to(device);mask=torch.from_numpy(batch["mask"]).to(device)
                optim.zero_grad(set_to_none=True)
                errors=sequence_loss(x,model(x,mask),mask);loss=errors.mean()
                if not torch.isfinite(loss):raise ValueError("Nonfinite temporal loss")
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step()
                losses.append(float(loss.detach().cpu()));global_step+=1
                for sid,kind,error in zip(batch["sample_id"],batch["sample_type"],errors.detach().cpu().tolist()):
                    key=str(kind);group_sums[key]=group_sums.get(key,0)+error;group_counts[key]=group_counts.get(key,0)+1
                    audit.write(json.dumps(dict(step=global_step,sample_id=int(sid),sample_type=key))+"\n")
                if step%args.log_every==0 or step==args.steps_per_epoch:
                    print(f"[AE epoch {epoch} step {step}] loss={losses[-1]:.6f}",flush=True)
            audit.flush();validation,_=evaluate_temporal(model,val,val_ids,args.batch_size,device)
            row=dict(epoch=epoch,step=global_step,train_mse=float(np.mean(losses)),
                     validation_mse=float(validation.mean()),seconds=time.monotonic()-started)
            labels=val.arrays["y_sequence"][val_ids]
            for label,name in ((0,"benign"),(1,"malicious")):
                row["validation_"+name+"_mse"]=float(validation[labels==label].mean()) if np.any(labels==label) else None
            row.update({"train_"+k+"_mse":group_sums[k]/group_counts[k] for k in group_sums});history.append(row)
            with (out/"history.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            checkpoint=dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            model_args=model_args,epoch=epoch,
                            embedding_checkpoint_sha256=train.manifest["embedding_checkpoint_sha256"])
            torch.save(checkpoint,out/"last.pt")
            if row["validation_mse"]<best:best=row["validation_mse"];torch.save(checkpoint,out/"best.pt")
            print(f"[epoch {epoch}] train={row['train_mse']:.6f} val={row['validation_mse']:.6f}",flush=True)
    saved=torch.load(out/"best.pt",map_location=device,weights_only=True);model.load_state_dict(saved["model_state"])
    _,vectors=evaluate_temporal(model,val,val_ids,args.batch_size,device,return_vectors=True)
    folder=out/"projector_validation";folder.mkdir()
    np.savetxt(folder/"vectors.tsv",vectors,delimiter="\t",fmt="%.8g")
    with (folder/"metadata.tsv").open("w",newline="") as handle:
        writer=csv.writer(handle,delimiter="\t")
        writer.writerow(["id","label","primary_attack","source_ip","active_buckets","split"])
        a=val.take(val_ids)
        for i in range(len(val_ids)):
            writer.writerow([int(a["sample_id"][i]),int(a["y_sequence"][i]),str(a["primary_attack"][i]),
                             str(a["left_ip"][i]),int(a["mask"][i].sum()),"validation"])
    dump(out/"summary.json",dict(complete=True,initial_validation_mse=float(initial.mean()),
          best_validation_mse=best,steps=global_step,seconds=time.monotonic()-begun,test_used=False))


def main():
    # --dataset is the final exported directory, not raw data or an embedding model.
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",choices=["cuda","cpu","mps"],default="cuda")
    p.add_argument("--seed",type=int,default=4266);p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--steps-per-epoch",type=int,default=200);p.add_argument("--width",type=int,default=32)
    p.add_argument("--latent",type=int,default=32);p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--real-only",action="store_true");p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--val-limit",type=int,default=4096)
    args=p.parse_args();validate_arguments(p,args)
    if args.batch_size%4:p.error("Batch size must be a multiple of four")
    train_temporal(args)


if __name__=="__main__":main()

