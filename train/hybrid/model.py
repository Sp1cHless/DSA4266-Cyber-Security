import torch
from torch import nn
from train.common.networks import sequence_input, pool, rotate_sequence, validate_start


class Model(nn.Module):
    def __init__(self, channels=64, hidden_size=64, dropout=0.2, circular_shift_start=0, **unused):
        super().__init__()
        self.circular_shift_start = validate_start(circular_shift_start)
        self.conv = nn.Sequential(nn.Conv1d(66, channels, 3, padding=1), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size * 3, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, mask):
        x, mask = rotate_sequence(x, mask, self.circular_shift_start)
        h = self.conv(sequence_input(x, mask).transpose(1, 2)).transpose(1, 2)
        h = h * mask.unsqueeze(-1)
        outputs, (hidden, _) = self.lstm(h)
        return self.head(torch.cat((pool(outputs, mask), hidden[-1]), dim=-1)).squeeze(-1)
