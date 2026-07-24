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


class HighResolutionFrameEncoder(nn.Module):
    def __init__(self, input_dim=182, hidden_dim=64, output_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, frames):
        return self.network(frames)


class HighResolutionSingleFrameRiskEncoder(nn.Module):
    def __init__(self, input_dim=182, hidden_dim=64, frame_dim=32):
        super().__init__()
        self.frame_encoder = HighResolutionFrameEncoder(
            input_dim, hidden_dim, frame_dim
        )
        self.head = nn.Linear(frame_dim, 1)

    def forward(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        encoded = self.frame_encoder(sequence[:, -1, :])
        return self.head(encoded).squeeze(-1)


class HighResolutionTemporalRiskEncoder(nn.Module):
    def __init__(self, input_dim=182, hidden_dim=64, frame_dim=32):
        super().__init__()
        self.frame_encoder = HighResolutionFrameEncoder(
            input_dim, hidden_dim, frame_dim
        )
        self.recurrent = nn.GRU(frame_dim, frame_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, 1))

    def forward(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        batch, steps, features = sequence.shape
        encoded = self.frame_encoder(sequence.reshape(batch * steps, features))
        encoded = encoded.reshape(batch, steps, -1)
        outputs, _ = self.recurrent(encoded)
        return self.head(outputs[:, -1, :]).squeeze(-1)
