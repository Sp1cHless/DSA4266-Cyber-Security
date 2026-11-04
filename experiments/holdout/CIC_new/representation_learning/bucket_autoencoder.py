"""Self-supervised bucket-embedding model; not required for raw preprocessing.

This AE is one way to learn reusable input embeddings. It is distinct from an
optional downstream temporal AE or intrusion classifier.
"""
import torch
from torch import nn
from torch.nn import functional as F

FIELDS=("protocol","l7","service")


class BucketAE(nn.Module):
    def __init__(self,slots,numeric_width=82,category_dim=16,bucket_dim=64):
        super().__init__()
        self.numeric=nn.Sequential(nn.Linear(numeric_width,32),nn.ReLU())
        self.categories=nn.ModuleDict({f:nn.Embedding(slots[f],category_dim,padding_idx=0) for f in FIELDS})
        self.encoder=nn.Sequential(nn.Linear(32+len(FIELDS)*category_dim,128),nn.ReLU(),nn.Linear(128,bucket_dim))
        self.numeric_decoder=nn.Linear(bucket_dim,numeric_width)
        self.category_decoders=nn.ModuleDict({f:nn.Linear(bucket_dim,slots[f]) for f in FIELDS})

    def encode(self,numeric,histograms):
        categorical=[histograms[f]@self.categories[f].weight for f in FIELDS]
        return F.normalize(self.encoder(torch.cat([self.numeric(numeric),*categorical],dim=-1)),p=2,dim=-1,eps=1e-8)

    def forward(self,numeric,histograms):
        z=self.encode(numeric,histograms)
        return self.numeric_decoder(z),{f:self.category_decoders[f](z) for f in FIELDS}


def bucket_loss(numeric,histograms,predicted_numeric,logits):
    regression=(numeric-predicted_numeric).square().mean(dim=-1)
    categorical=torch.stack([-(histograms[f]*F.log_softmax(logits[f],dim=-1)).sum(dim=-1)
                             for f in FIELDS],dim=-1).mean(dim=-1)
    return (regression+categorical).mean()
