"""Framework-independent augmentation of already learned bucket vectors.

No network definitions, optimiser, gradient updates or AE training live here.
The NumPy arrays returned here can feed PyTorch, TensorFlow or other models.
"""
from pathlib import Path
import numpy as np

from .dataset import read_json


def rotate(values,shifts):
    values=np.asarray(values);shifts=np.asarray(shifts)
    times=values.shape[1]
    index=(np.arange(times)[None,:]-shifts[:,None])%times
    index=index.reshape(*index.shape,*([1]*(values.ndim-2)))
    return np.take_along_axis(values,np.broadcast_to(index,values.shape),axis=1)


def superimpose(a,b,mask_a,mask_b,label_a,label_b,flows_a,flows_b,shifts):
    a,b,mask_a,mask_b,label_a,label_b,flows_a,flows_b=map(np.asarray,(a,b,mask_a,mask_b,label_a,label_b,flows_a,flows_b))
    mb=rotate(mask_b,shifts)
    combined=a*mask_a[...,None]+rotate(b,shifts)*mb[...,None]
    combined=combined/np.maximum(np.linalg.norm(combined,axis=-1,keepdims=True),1e-8)
    mask=np.maximum(mask_a,mb);combined*=mask[...,None]
    return combined,mask,np.maximum(label_a,rotate(label_b,shifts)),flows_a+rotate(flows_b,shifts)


class Encoded:
    """Read frozen sequence vectors without importing a neural-network library."""
    def __init__(self,folder,split):
        folder=Path(folder);self.manifest=read_json(folder/"manifest.json")
        if not self.manifest.get("complete") or not self.manifest.get("frozen"):raise ValueError("Frozen complete encoding required")
        self.meta=np.load(folder/split/"metadata.npy",mmap_mode="r")
        self.arrays={k:np.load(folder/split/(k+".npy"),mmap_mode="r") for k in ("embeddings","mask","y_bucket","flow_counts")}
        self.scaler=read_json(folder/"scaler.json")

    def take(self,indices):return {k:np.array(a[indices],copy=True) for k,a in self.arrays.items()}

    def volume(self,counts,mask):
        return ((np.log1p(counts.astype(np.float32))-self.scaler["mean"][0])/self.scaler["scale"][0])*mask


def mixed_batch(store,left,right,shifts,return_details=False):
    a=store.take(left);b=store.take(np.maximum(right,0))
    present=(right>=0).astype(np.float32)[:,None]
    b["mask"]*=present;b["y_bucket"]*=present.astype(np.int64);b["flow_counts"]*=present.astype(np.int64)
    z,mask,y,counts=superimpose(a["embeddings"],b["embeddings"],a["mask"],b["mask"],
                               a["y_bucket"],b["y_bucket"],a["flow_counts"],b["flow_counts"],shifts)
    x=np.concatenate([z,store.volume(counts,mask)[...,None]],axis=-1).astype(np.float32)
    result=(x,mask.astype(np.float32),y.max(axis=1).astype(np.int64))
    return (*result,y,counts) if return_details else result
