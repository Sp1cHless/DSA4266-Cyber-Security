"""Preserve the sequence classifier and add one linear head per recurrent state."""
import torch
from torch import nn

from train.lstm.model import Model as Baseline
from train.common.networks import pool, rotate_sequence, sequence_input


class Model(Baseline):
    def __init__(self, hidden_size=64, dropout=0.2, circular_shift_start=0, **unused):
        super().__init__(hidden_size=hidden_size, dropout=dropout, circular_shift_start=circular_shift_start)
        # Preserve the baseline RNG stream, including subsequent dropout draws.
        with torch.random.fork_rng(devices=[]):
            self.bucket_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.bucket_head.weight)
        nn.init.zeros_(self.bucket_head.bias)

    def forward_with_buckets(self, x, mask):
        x, mask = rotate_sequence(x, mask, self.circular_shift_start)
        states, (hidden, _) = self.lstm(sequence_input(x, mask))
        features = torch.cat((pool(states, mask), hidden[-1]), dim=-1)
        sequence = self.head(features).squeeze(-1)
        return sequence, self.bucket_head(states).squeeze(-1)

    def forward(self, x, mask):
        return self.forward_with_buckets(x, mask)[0]
