"""Apply a frozen bucket encoder to prepared real sequences; no optimiser runs."""
from pathlib import Path
import hashlib
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from CIC_new.preprocessing.dataset import Originals, FIELDS, fingerprint
from .bucket_autoencoder import BucketAE
from CIC_new.training_utils import device_for, fresh, dump, config, arguments, validate_arguments

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



def main():
    p=arguments(__doc__,256)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--splits",nargs="+",choices=["train","validation","test"],default=["train","validation"])
    args=p.parse_args();validate_arguments(p,args);encode(args)

if __name__=="__main__":main()

