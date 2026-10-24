"""Runtime helpers for fitting and applying the Scheme2 input encoder."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch


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


def arguments(description,batch_size):
    p=argparse.ArgumentParser(description=description)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",choices=["cuda","cpu","mps"],default="cuda")
    p.add_argument("--seed",type=int,default=4266)
    p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=batch_size)
    return p


def validate_arguments(parser,args):
    for key in ("batch_size","cpu_threads","steps","validate_every","category_dim","bucket_dim","epochs","steps_per_epoch","width","latent","log_every","val_limit","lr"):
        if hasattr(args,key) and getattr(args,key)<=0:parser.error(f"{key} must be positive")
