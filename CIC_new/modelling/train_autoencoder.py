"""Train ONLY the optional temporal AE on an already exported sequence dataset.

No raw-data processing, embedding fitting or synthetic sample construction occurs
in this script. CNN/LSTM/classifier experiments can consume the same dataset.
"""
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from CIC_new.preprocessing.dataset import ModelInputs
from CIC_new.training_utils import device_for,fresh,dump,config,validate_arguments
from .autoencoder import TemporalAE,sequence_loss


class EarlyStopping:
    """Stop after patience epochs without a cumulative relative improvement.

    The reference updates only after a significant improvement. Small decreases
    can therefore accumulate rather than being discarded at every epoch.
    Checkpoint selection separately tracks the absolute minimum validation MSE.
    """
    def __init__(self,patience=20,min_delta=.001):
        if patience<0:raise ValueError("patience must be nonnegative; 0 disables early stopping")
        if not math.isfinite(min_delta) or not 0<=min_delta<1:
            raise ValueError("min_delta must be a finite relative fraction in [0, 1)")
        self.patience=patience;self.min_delta=min_delta
        self.reference=math.inf;self.wait=0

    def update(self,value):
        if not math.isfinite(value) or value<0:raise ValueError("Invalid validation MSE")
        if value<self.reference*(1-self.min_delta):
            self.reference=value;self.wait=0
        else:self.wait+=1
        return self.patience>0 and self.wait>=self.patience


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
    stopping=EarlyStopping(getattr(args,"patience",20),getattr(args,"min_delta",.001))
    device=device_for(args);out=fresh(args.out)
    status=dict(status="running",complete=False,run_id=out.name,
                output_directory=str(out.resolve()),started_at=datetime.now().astimezone().isoformat())
    dump(out/"run_status.json",status)
    print(f"Run directory: {out.resolve()}",flush=True)
    try:
        print(f"Loading train/validation on {device}; first use builds disk mmap caches...",flush=True)
        transform=getattr(args,"input_transform","none")
        with ModelInputs(args.dataset,"train",transform) as train,ModelInputs(args.dataset,"validation",transform) as val:
            _train_temporal(args,device,out,train,val,stopping)
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
                      finished_at=datetime.now().astimezone().isoformat(),error=type(error).__name__,detail=str(error))
        dump(out/"run_status.json",status)
        raise
    else:
        status.update(status="complete",complete=True,finished_at=datetime.now().astimezone().isoformat())
        dump(out/"run_status.json",status)


def _train_temporal(args,device,out,train,val,stopping):
    val_ids=np.arange(len(val))
    if len(val_ids)>args.val_limit:
        val_ids=np.sort(np.random.default_rng(args.seed+17).choice(val_ids,args.val_limit,replace=False))
    model_args=dict(features=train.manifest["feature_width"],width=args.width,latent=args.latent,
                    timesteps=train.manifest["timesteps"])
    model=TemporalAE(**model_args).to(device);optim=torch.optim.Adam(model.parameters(),lr=args.lr)
    dump(out/"config.json",config(args,dict(model=model_args,test_used=False,
         objective="mixed-class reconstruction of already prepared sequence inputs",
         architecture="temporal convolutional autoencoder",
         dataset_representation=train.manifest.get("representation","legacy"),
         input_transform=train.input_transform,
         input_transform_spec=dict(mode=train.input_transform,axis="features within each bucket",
             stage="after final scaling and synthetic composition",epsilon=1e-8,
             includes_flow_count=True,padding="zero") if train.input_transform=="bucket_l2" else None,
         sampling="balanced real benign/malicious" if args.real_only else "balanced four sample types, each 25%; with replacement",
         validation_examples=len(val_ids),
         early_stopping=dict(monitor="validation_mse",mode="min",patience=stopping.patience,
                             min_delta_relative=stopping.min_delta,restore_best=True),
         embedding_checkpoint_sha256=train.manifest["embedding_checkpoint_sha256"],
         dataset_augmentation=train.manifest["shift_mode"])))
    shutil.copy2(Path(args.dataset)/"manifest.json",out/"dataset_manifest.json")
    for name in ("frozen_bucket_encoder.pt","embeddings.npz","embedding_manifest.json","validation.json","scaler.json","schema.json"):
        source=Path(args.dataset)/name
        if source.exists():shutil.copy2(source,out/name)
    np.save(out/"validation_ids.npy",val.arrays["sample_id"][val_ids])
    print(f"AE input=({model_args['timesteps']}, {model_args['features']}), latent={args.latent}; "
          f"train={len(train)}, validation={len(val_ids)}/{len(val)}, "
          f"updates={args.epochs*args.steps_per_epoch}, draws={args.epochs*args.steps_per_epoch*args.batch_size}",flush=True)
    initial,_=evaluate_temporal(model,val,val_ids,args.batch_size,device)
    print(f"[initial] validation_mse={initial.mean():.6f}",flush=True)
    best=float("inf");history=[];global_step=0;begun=time.monotonic();stopped_early=False
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
            should_stop=stopping.update(row["validation_mse"])
            row["epochs_without_significant_improvement"]=stopping.wait
            labels=val.arrays["y_sequence"][val_ids]
            for label,name in ((0,"benign"),(1,"malicious")):
                row["validation_"+name+"_mse"]=float(validation[labels==label].mean()) if np.any(labels==label) else None
            row.update({"train_"+k+"_mse":group_sums[k]/group_counts[k] for k in group_sums});history.append(row)
            with (out/"history.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            checkpoint=dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            model_args=model_args,epoch=epoch,validation_mse=row["validation_mse"],
                            embedding_checkpoint_sha256=train.manifest["embedding_checkpoint_sha256"],
                            input_transform=train.input_transform)
            torch.save(checkpoint,out/"last.pt")
            if row["validation_mse"]<best:best=row["validation_mse"];torch.save(checkpoint,out/"best.pt")
            print(f"[epoch {epoch}] train={row['train_mse']:.6f} val={row['validation_mse']:.6f} "
                  f"benign={row['validation_benign_mse']} malicious={row['validation_malicious_mse']} "
                  f"wait={stopping.wait}/{stopping.patience}",flush=True)
            if should_stop:
                stopped_early=True
                print(f"[early stopping] No cumulative relative improvement > {stopping.min_delta:.2%} "
                      f"for {stopping.patience} epochs; restoring best checkpoint.",flush=True)
                break
    saved=torch.load(out/"best.pt",map_location=device,weights_only=True);model.load_state_dict(saved["model_state"])
    restored,vectors=evaluate_temporal(model,val,val_ids,args.batch_size,device,return_vectors=True)
    folder=out/"projector_validation";folder.mkdir()
    np.savetxt(folder/"vectors.tsv",vectors,delimiter="\t",fmt="%.8g")
    with (folder/"metadata.tsv").open("w",newline="") as handle:
        writer=csv.writer(handle,delimiter="\t")
        writer.writerow(["id","label","primary_attack","source_ip","active_buckets","split"])
        # Projector metadata must not copy the full validation feature tensor into RAM.
        a={k:val.arrays[k][val_ids] for k in ("sample_id","y_sequence","primary_attack","left_ip","mask")}
        for i in range(len(val_ids)):
            writer.writerow([int(a["sample_id"][i]),int(a["y_sequence"][i]),str(a["primary_attack"][i]),
                             str(a["left_ip"][i]),int(a["mask"][i].sum()),"validation"])
    dump(out/"summary.json",dict(complete=True,initial_validation_mse=float(initial.mean()),
          best_validation_mse=best,best_epoch=saved["epoch"],steps=global_step,
          sample_draws=global_step*args.batch_size,validation_examples=len(val_ids),
          epochs_completed=len(history),max_epochs=args.epochs,
          stopping_reason="early_stopping" if stopped_early else "max_epochs",
          plateau_detected=stopped_early,
          epochs_without_significant_improvement=stopping.wait,
          patience=stopping.patience,min_delta_relative=stopping.min_delta,
          restored_validation_mse=float(restored.mean()),
          best_validation_benign_mse=float(restored[labels==0].mean()) if np.any(labels==0) else None,
          best_validation_malicious_mse=float(restored[labels==1].mean()) if np.any(labels==1) else None,
          seconds=time.monotonic()-begun,test_used=False))
    print(f"Complete: best epoch={saved['epoch']}, validation_mse={best:.6f}; outputs: {out}",flush=True)


def main():
    # --dataset is the final exported directory, not raw data or an embedding model.
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--input-transform",choices=("none","bucket_l2"),default="none")
    p.add_argument("--device",choices=["cuda","cpu","mps"],default="cuda")
    p.add_argument("--seed",type=int,default=4266);p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--steps-per-epoch",type=int,default=200);p.add_argument("--width",type=int,default=32)
    p.add_argument("--latent",type=int,default=32);p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--real-only",action="store_true");p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--val-limit",type=int,default=4096)
    p.add_argument("--patience",type=int,default=20,help="Epochs without significant validation improvement; 0 disables")
    p.add_argument("--min-delta",type=float,default=.001,help="Relative improvement fraction, 0.001 = 0.1 percent")
    args=p.parse_args();validate_arguments(p,args)
    if args.batch_size%4:p.error("Batch size must be a multiple of four")
    train_temporal(args)


if __name__=="__main__":main()

