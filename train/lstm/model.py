import torch
from torch import nn
from train.common.networks import sequence_input, pool


class Model(nn.Module):
    def __init__(self, hidden_size=64, dropout=0.2, **unused):
        super().__init__()
        self.lstm = nn.LSTM(66, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size * 3, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, mask):
        # Keep empty bins: removing them would change elapsed-time spacing.
        outputs, (hidden, _) = self.lstm(sequence_input(x, mask))
        features = torch.cat((pool(outputs, mask), hidden[-1]), dim=-1)
        return self.head(features).squeeze(-1)
