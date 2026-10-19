"""Optional temporal CNN AE; consumes prepared sequence vectors and masks."""
import torch
from torch import nn


class TemporalAE(nn.Module):
    def __init__(self,features=65,width=32,latent=32,timesteps=10):
        super().__init__()
        self.width=width;self.steps=(timesteps+1)//2;self.timesteps=timesteps
        self.encoder=nn.Sequential(nn.Conv1d(features+1,width,3,padding=1),nn.ReLU(),
                                   nn.Conv1d(width,width,3,stride=2,padding=1),nn.ReLU())
        self.to_latent=nn.Linear(width*self.steps,latent)
        self.from_latent=nn.Linear(latent,width*self.steps)
        self.decoder=nn.Sequential(nn.ConvTranspose1d(width,width,4,stride=2,padding=1),nn.ReLU(),nn.Conv1d(width,features,3,padding=1))

    def encode(self,x,mask):
        x=torch.cat([x*mask.unsqueeze(-1),mask.unsqueeze(-1)],dim=-1)
        return self.to_latent(self.encoder(x.transpose(1,2)).flatten(1))

    def forward(self,x,mask):
        z=self.encode(x,mask)
        return self.decoder(self.from_latent(z).reshape(-1,self.width,self.steps)).transpose(1,2)[:,:self.timesteps]


def sequence_loss(x,prediction,mask):
    error=(x-prediction).square().mean(dim=-1)
    return (error*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
