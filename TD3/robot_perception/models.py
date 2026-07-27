import torch
import torch.nn as nn


class LocalRobotDetector(nn.Module):
    """Small CNN for robot identity and center refinement in a lidar patch."""

    def __init__(self, input_channels=3, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 4, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.center_regressor = nn.Linear(hidden_dim, 2)

    def forward(self, patches):
        if patches.ndim != 4 or patches.shape[1] != 3:
            raise ValueError("patches must have shape [batch, 3, height, width]")
        features = self.shared(self.features(patches))
        logits = self.classifier(features).squeeze(-1)
        offsets = self.center_regressor(features)
        return logits, offsets
