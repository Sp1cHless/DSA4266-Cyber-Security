"""Unchanged LSTM detector with binary-class-conditional host adversaries."""
import numpy as np
import torch
from torch import nn
from train.lstm.model import Model as Baseline
from train.common.networks import sequence_input, pool, rotate_sequence


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class Model(nn.Module):
    def __init__(self, architecture, host_sizes, seed=4266):
        super().__init__()
        self.detector = Baseline(**architecture)
        width = architecture['hidden_size'] * 3
        # Extra initialization must not change the detector/dropout RNG stream.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed + 7101)
            self.adversaries = nn.ModuleList([
                nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, size))
                for size in host_sizes])

    def encode(self, x, mask):
        x, mask = rotate_sequence(x, mask, self.detector.circular_shift_start)
        outputs, (hidden, _) = self.detector.lstm(sequence_input(x, mask))
        return torch.cat((pool(outputs, mask), hidden[-1]), dim=-1)

    def forward(self, x, mask):
        return self.detector(x, mask)

    def host_logits(self, features, label, strength):
        return self.adversaries[label](Reverse.apply(features, strength))


def host_pools(training, minimum=20):
    """Define identities from real training examples only; no OTHER catch-all."""
    result, audit = [], []
    for label, kind in enumerate(('real_benign', 'real_malicious')):
        indices = np.flatnonzero(training['types'] == kind)
        hosts, inverse, counts = np.unique(training['host'][indices], return_inverse=True,
                                          return_counts=True)
        eligible = np.flatnonzero(counts >= minimum)
        if len(eligible) < 2:
            raise ValueError('Need at least two sufficiently supported hosts in ' + kind)
        order = np.argsort(inverse, kind='stable')
        boundaries = np.r_[0, np.cumsum(counts)]
        pools = [indices[order[boundaries[h]:boundaries[h + 1]]] for h in eligible]
        result.append(pools)
        audit.append({'label': label, 'sample_type': kind, 'all_hosts': len(hosts),
                      'eligible_hosts': len(eligible), 'minimum_sequences': minimum,
                      'all_real_sequences': len(indices),
                      'eligible_sequences': sum(map(len, pools)),
                      'sequence_coverage': sum(map(len, pools)) / len(indices),
                      'hosts': hosts[eligible].tolist(), 'counts': counts[eligible].tolist()})
    return result, audit


def sample_hosts(pools, count, rng):
    targets = rng.integers(len(pools), size=count)
    indices = np.array([pools[h][rng.integers(len(pools[h]))] for h in targets])
    return indices, targets
