import torch
import torch.nn as nn


class InteractionGate(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64)):
        super().__init__()
        if input_dim < 1 or len(hidden_dims) != 2 or min(hidden_dims) < 1:
            raise ValueError("gate dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], 1),
        )

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("gate features have the wrong shape")
        return self.network(features).squeeze(-1)
