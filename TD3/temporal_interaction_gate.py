import numpy as np
import torch
import torch.nn as nn


ACTOR_COMPARISON_DIM = 6


def actor_comparison_features(standard_actions, interaction_actions):
    standard = np.asarray(standard_actions, dtype=np.float32)
    interaction = np.asarray(interaction_actions, dtype=np.float32)
    if standard.shape != interaction.shape or standard.ndim != 2:
        raise ValueError("actor actions must have matching [N, action_dim] shapes")
    if standard.shape[1] != 2:
        raise ValueError("actor actions must contain linear and angular commands")
    if not np.all(np.isfinite(standard)) or not np.all(np.isfinite(interaction)):
        raise ValueError("actor actions must be finite")

    standard_deployed = standard.copy()
    interaction_deployed = interaction.copy()
    standard_deployed[:, 0] = (standard_deployed[:, 0] + 1.0) / 2.0
    interaction_deployed[:, 0] = (interaction_deployed[:, 0] + 1.0) / 2.0
    return np.concatenate(
        (
            standard_deployed,
            interaction_deployed,
            interaction_deployed - standard_deployed,
        ),
        axis=1,
    ).astype(np.float32)


class TemporalInteractionGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("temporal Gate dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2 or 1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2 or 1, 1),
        )

    def forward(self, features):
        if features.ndim != 3 or features.shape[2] != self.input_dim:
            raise ValueError("temporal Gate features have the wrong shape")
        outputs, _ = self.gru(features)
        return self.head(outputs[:, -1]).squeeze(-1)
