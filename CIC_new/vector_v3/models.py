"""Learnable categorical embeddings and stable, frozen-target superposition."""
import torch
from torch import nn
from torch.nn import functional as F

FIELDS=("protocol","l7","service")


class BucketAE(nn.Module):
    def __init__(self,slots,numeric_width=82,category_dim=16,bucket_dim=64):
        super().__init__()
        self.numeric=nn.Sequential(nn.Linear(numeric_width,32),nn.ReLU())
        self.categories=nn.ModuleDict({f:nn.Embedding(slots[f],category_dim,padding_idx=0) for f in FIELDS})
        self.encoder=nn.Sequential(nn.Linear(32+len(FIELDS)*category_dim,128),nn.ReLU(),
                                   nn.Linear(128,bucket_dim))
        self.numeric_decoder=nn.Linear(bucket_dim,numeric_width)
        self.category_decoders=nn.ModuleDict({f:nn.Linear(bucket_dim,slots[f]) for f in FIELDS})

    def encode(self,numeric,histograms):
        # Histograms are per-field category proportions; IDs themselves are never summed.
        categorical=[histograms[f]@self.categories[f].weight for f in FIELDS]
        # Train the decoder on the SAME unit representation later exported.
        # Otherwise normalising only at export could discard information the
        # bucket AE had encoded solely in vector magnitude.
        return F.normalize(self.encoder(torch.cat([self.numeric(numeric),*categorical],dim=-1)),p=2,dim=-1,eps=1e-8)

    def forward(self,numeric,histograms):
        z=self.encode(numeric,histograms)
        return self.numeric_decoder(z),{f:self.category_decoders[f](z) for f in FIELDS}


def bucket_loss(numeric,histograms,predicted_numeric,logits):
    regression=(numeric-predicted_numeric).square().mean(dim=-1)
    categorical=torch.stack([-(histograms[f]*F.log_softmax(logits[f],dim=-1)).sum(dim=-1)
                             for f in FIELDS],dim=-1).mean(dim=-1)
    return (regression+categorical).mean()


def rotate(values,shifts):
    """Per-example np.roll(...,axis=time), for arbitrary trailing dimensions."""
    times=values.shape[1]
    index=(torch.arange(times,device=values.device)[None,:]-shifts[:,None])%times
    index=index.reshape(*index.shape,*([1]*(values.ndim-2))).expand_as(values)
    return torch.gather(values,1,index)


def superimpose(a,b,mask_a,mask_b,label_a,label_b,flows_a,flows_b,shifts):
    """Rotate donor B, ADD EMBEDDINGS, then per-bucket L2 normalise.

    Returns embeddings, mask, bucket labels and flow counts. Metadata follows
    the same rotation; numerical bucket feature vectors are never added here.
    """
    mb=rotate(mask_b,shifts)
    b=rotate(b,shifts)*mb.unsqueeze(-1)
    combined=F.normalize(a*mask_a.unsqueeze(-1)+b,p=2,dim=-1,eps=1e-8)
    mask=torch.maximum(mask_a,mb)
    combined=combined*mask.unsqueeze(-1)
    labels=torch.maximum(label_a,rotate(label_b,shifts))
    counts=flows_a+rotate(flows_b,shifts)
    return combined,mask,labels,counts


class TemporalAE(nn.Module):
    def __init__(self,features=65,width=32,latent=32,timesteps=10):
        super().__init__()
        self.width=width;self.steps=(timesteps+1)//2;self.timesteps=timesteps
        self.encoder=nn.Sequential(nn.Conv1d(features+1,width,3,padding=1),nn.ReLU(),
                                   nn.Conv1d(width,width,3,stride=2,padding=1),nn.ReLU())
        self.to_latent=nn.Linear(width*self.steps,latent)
        self.from_latent=nn.Linear(latent,width*self.steps)
        self.decoder=nn.Sequential(nn.ConvTranspose1d(width,width,4,stride=2,padding=1),
                                   nn.ReLU(),nn.Conv1d(width,features,3,padding=1))

    def encode(self,x,mask):
        x=torch.cat([x*mask.unsqueeze(-1),mask.unsqueeze(-1)],dim=-1)
        return self.to_latent(self.encoder(x.transpose(1,2)).flatten(1))

    def forward(self,x,mask):
        z=self.encode(x,mask)
        return self.decoder(self.from_latent(z).reshape(-1,self.width,self.steps)).transpose(1,2)[:,:self.timesteps]


def sequence_loss(x,prediction,mask):
    error=(x-prediction).square().mean(dim=-1)
    return (error*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
