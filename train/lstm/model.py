import torch
from torch import nn
from train.common.networks import sequence_input, pool, rotate_sequence, validate_start


class Model(nn.Module):
    def __init__(self, hidden_size=64, dropout=0.2, circular_shift_start=0, **unused):
        super().__init__()
        self.circular_shift_start = validate_start(circular_shift_start)
        self.lstm = nn.LSTM(66, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size * 3, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, mask):
        # Move the selected original bucket to index zero, including empty bins.
        # Keeping this in the model applies the saved rule during frozen evaluation.
        x, mask = rotate_sequence(x, mask, self.circular_shift_start)
        # Keep empty bins: removing them would change elapsed-time spacing.
        outputs, (hidden, _) = self.lstm(sequence_input(x, mask))
        features = torch.cat((pool(outputs, mask), hidden[-1]), dim=-1)
        return self.head(features).squeeze(-1)
