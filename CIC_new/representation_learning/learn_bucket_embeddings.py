"""Learn reusable bucket embeddings; does not prepare raw data or train a detector."""
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from CIC_new.preprocessing.dataset import Originals, FIELDS, fingerprint
from .bucket_autoencoder import BucketAE, bucket_loss
from CIC_new.training_utils import device_for, fresh, dump, config, arguments, validate_arguments

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



def main():
    p=arguments(__doc__,512)
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--validate-every",type=int,default=200)
    p.add_argument("--category-dim",type=int,default=16)
    p.add_argument("--bucket-dim",type=int,default=64)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--val-limit",type=int,default=4096)
    args=p.parse_args();validate_arguments(p,args);train_bucket(args)

if __name__=="__main__":main()

