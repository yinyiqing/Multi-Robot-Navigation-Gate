import math

import torch
import torch.nn as nn


def alternating_binary_masks(input_dim, num_blocks, device=None):
    if int(input_dim) < 2:
        raise ValueError("input_dim must be at least 2")
    if int(num_blocks) < 1:
        raise ValueError("num_blocks must be positive")
    base = torch.arange(int(input_dim), device=device) % 2
    masks = []
    for index in range(int(num_blocks)):
        mask = base if index % 2 == 0 else 1 - base
        masks.append(mask.to(dtype=torch.float32))
    return masks


class AffineCouplingBlock(nn.Module):
    def __init__(self, input_dim, mask, hidden_dim=128, log_scale_limit=2.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_scale_limit = float(log_scale_limit)
        if self.input_dim < 2 or self.hidden_dim < 1:
            raise ValueError("flow dimensions must be positive")
        if self.log_scale_limit <= 0.0:
            raise ValueError("log_scale_limit must be positive")

        mask = torch.as_tensor(mask, dtype=torch.float32).reshape(-1)
        if mask.shape != (self.input_dim,):
            raise ValueError("mask shape must match input_dim")
        if not torch.all((mask == 0.0) | (mask == 1.0)):
            raise ValueError("mask must be binary")
        if int(torch.sum(mask).item()) in (0, self.input_dim):
            raise ValueError("mask must leave at least one transformed dimension")
        self.register_buffer("mask", mask)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.input_dim),
        )

    def _shift_and_log_scale(self, masked):
        shift, raw_log_scale = torch.chunk(self.network(masked), 2, dim=-1)
        log_scale = self.log_scale_limit * torch.tanh(raw_log_scale)
        active = 1.0 - self.mask
        return shift * active, log_scale * active

    def forward(self, values):
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("flow input must have shape [batch, input_dim]")
        masked = values * self.mask
        shift, log_scale = self._shift_and_log_scale(masked)
        active = 1.0 - self.mask
        transformed = masked + active * (values * torch.exp(log_scale) + shift)
        log_det = torch.sum(log_scale, dim=-1)
        return transformed, log_det

    def inverse(self, values):
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("flow input must have shape [batch, input_dim]")
        masked = values * self.mask
        shift, log_scale = self._shift_and_log_scale(masked)
        active = 1.0 - self.mask
        restored = masked + active * ((values - shift) * torch.exp(-log_scale))
        log_det = -torch.sum(log_scale, dim=-1)
        return restored, log_det


class RealNVPFlow(nn.Module):
    def __init__(
        self,
        input_dim=24,
        num_blocks=6,
        hidden_dim=128,
        log_scale_limit=2.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_blocks = int(num_blocks)
        self.hidden_dim = int(hidden_dim)
        self.log_scale_limit = float(log_scale_limit)
        masks = alternating_binary_masks(self.input_dim, self.num_blocks)
        self.blocks = nn.ModuleList(
            [
                AffineCouplingBlock(
                    self.input_dim,
                    mask,
                    hidden_dim=self.hidden_dim,
                    log_scale_limit=self.log_scale_limit,
                )
                for mask in masks
            ]
        )

    def forward(self, values):
        log_det = torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
        output = values
        for block in self.blocks:
            output, block_log_det = block(output)
            log_det = log_det + block_log_det
        return output, log_det

    def inverse(self, values):
        log_det = torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
        output = values
        for block in reversed(self.blocks):
            output, block_log_det = block.inverse(output)
            log_det = log_det + block_log_det
        return output, log_det

    def negative_log_likelihood(self, values):
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("flow input must have shape [batch, input_dim]")
        latent, log_det = self.forward(values)
        base_nll = 0.5 * torch.sum(
            latent * latent + math.log(2.0 * math.pi), dim=-1
        )
        return base_nll - log_det
