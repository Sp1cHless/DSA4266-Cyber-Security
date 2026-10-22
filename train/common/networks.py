"""Mask-aware pooling on the original ten-bin temporal grid."""
import torch


def sequence_input(x, mask):
    return torch.cat((x * mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)


def pool(values, mask):
    active = mask.unsqueeze(-1)
    mean = (values * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
    maximum = values.masked_fill(active == 0, torch.finfo(values.dtype).min).max(dim=1).values
    return torch.cat((mean, maximum), dim=-1)
