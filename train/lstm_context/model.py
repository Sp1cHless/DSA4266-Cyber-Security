"""LSTM ablation: replace only uniform mean pooling with learned attention."""
import torch
from torch import nn
from train.lstm.model import Model as Baseline
from train.common.networks import pool, rotate_sequence, sequence_input


class Model(Baseline):
    def __init__(self, hidden_size=64, dropout=0.2, circular_shift_start=0, **unused):
        super().__init__(hidden_size=hidden_size, dropout=dropout, circular_shift_start=circular_shift_start)
        self.attention = nn.Linear(hidden_size, 1)
        # Begin with the original uniform mean; the LSTM and head keep their original initialization.
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def forward(self, x, mask):
        x, mask = rotate_sequence(x, mask, self.circular_shift_start)
        states, (hidden, _) = self.lstm(sequence_input(x, mask))
        logits = self.attention(states).squeeze(-1).masked_fill(mask == 0, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        average = (states * weights.unsqueeze(-1)).sum(1)
        maximum = pool(states, mask)[:, states.shape[-1]:]
        return self.head(torch.cat((average, maximum, hidden[-1]), dim=-1)).squeeze(-1)
