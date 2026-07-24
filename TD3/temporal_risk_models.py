import torch
import torch.nn as nn


class SingleFrameRiskEncoder(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        return self.network(sequence[:, -1, :]).squeeze(-1)


class TemporalRiskEncoder(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=32):
        super().__init__()
        self.recurrent = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        outputs, _ = self.recurrent(sequence)
        return self.head(outputs[:, -1, :]).squeeze(-1)
