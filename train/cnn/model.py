from torch import nn
from train.common.networks import sequence_input, pool


class Model(nn.Module):
    def __init__(self, channels=64, dropout=0.2, **unused):
        super().__init__()
        self.first = nn.Conv1d(66, channels, kernel_size=3, padding=1)
        self.second = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(channels * 4, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, mask):
        h = sequence_input(x, mask).transpose(1, 2)
        h = self.dropout(self.relu(self.first(h))) * mask.unsqueeze(1)
        h = self.relu(self.second(h)).transpose(1, 2)
        return self.head(pool(h, mask)).squeeze(-1)
