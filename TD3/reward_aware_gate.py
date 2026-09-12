import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardAwareTemporalGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        if input_dim < 1 or hidden_dim < 2:
            raise ValueError("reward-aware Gate dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        head_dim = self.hidden_dim // 2
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.interaction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, features):
        if features.ndim != 3 or features.shape[2] != self.input_dim:
            raise ValueError("reward-aware Gate features have the wrong shape")
        outputs, _ = self.gru(features)
        encoded = outputs[:, -1]
        return (
            self.interaction_head(encoded).squeeze(-1),
            self.advantage_head(encoded).squeeze(-1),
        )


class AuxiliaryRewardTemporalGate(nn.Module):
    """Single-head Router with reward difference used only as training aid.

    The deployable output is one interaction logit.  Counterfactual one-step
    reward differences supervise the same head through an auxiliary loss and
    never alter inference-time logits.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        if input_dim < 1 or hidden_dim < 2:
            raise ValueError("auxiliary reward Gate dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        head_dim = self.hidden_dim // 2
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.interaction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1)
        )

    def forward(self, features):
        if features.ndim != 3 or features.shape[2] != self.input_dim:
            raise ValueError("auxiliary reward Gate features have the wrong shape")
        outputs, _ = self.gru(features)
        return self.interaction_head(outputs[:, -1]).squeeze(-1)


class MultiTaskRewardTemporalGate(nn.Module):
    """B7 Router: full-data interaction head plus a separate reward preference head."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        if input_dim < 1 or hidden_dim < 2:
            raise ValueError("multi-task reward Gate dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        head_dim = self.hidden_dim // 2
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.interaction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1)
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1)
        )

    def forward(self, features):
        if features.ndim != 3 or features.shape[2] != self.input_dim:
            raise ValueError("multi-task reward Gate features have the wrong shape")
        outputs, _ = self.gru(features)
        encoded = outputs[:, -1]
        return (
            self.interaction_head(encoded).squeeze(-1),
            self.reward_head(encoded).squeeze(-1),
        )


class DualEncoderRewardTemporalGate(nn.Module):
    """Joint-supervision Router with non-interfering temporal branches.

    Both branches consume the same deployable feature history. The interaction
    branch learns the privileged 2 m phase label, while the reward branch learns
    a bounded transform of the same-state one-step Actor reward difference.
    Separate GRUs prevent sparse reward supervision from changing the phase
    representation learned from the full Gate dataset.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        if input_dim < 1 or hidden_dim < 2:
            raise ValueError("dual-encoder Gate dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        head_dim = self.hidden_dim // 2
        self.interaction_gru = nn.GRU(
            self.input_dim, self.hidden_dim, batch_first=True
        )
        self.reward_gru = nn.GRU(
            self.input_dim, self.hidden_dim, batch_first=True
        )
        self.interaction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
            nn.Tanh(),
        )

    def forward(self, features):
        if features.ndim != 3 or features.shape[2] != self.input_dim:
            raise ValueError("dual-encoder Gate features have the wrong shape")
        interaction_outputs, _ = self.interaction_gru(features)
        reward_outputs, _ = self.reward_gru(features)
        return (
            self.interaction_head(interaction_outputs[:, -1]).squeeze(-1),
            self.reward_head(reward_outputs[:, -1]).squeeze(-1),
        )


def joint_supervision_routing_score(
    interaction_logits, bounded_advantage, reward_weight=0.25
):
    """Combine the two predicted supervision signals at deployment."""
    if float(reward_weight) < 0.0:
        raise ValueError("reward weight must be non-negative")
    interaction = torch.as_tensor(interaction_logits)
    advantage = torch.as_tensor(
        bounded_advantage, dtype=interaction.dtype, device=interaction.device
    )
    if interaction.shape != advantage.shape:
        raise ValueError("interaction and reward predictions must have equal shapes")
    adjusted = interaction + float(reward_weight) * torch.clamp(
        advantage, -1.0, 1.0
    )
    return torch.sigmoid(adjusted)


def auxiliary_reward_gate_loss(
    interaction_logits,
    interaction_labels,
    delta_reward,
    noise_threshold=0.005,
    reward_weight=0.25,
):
    """BCE phase loss plus sign(delta-r) auxiliary loss on reliable anchors."""
    logits = torch.as_tensor(interaction_logits)
    labels = torch.as_tensor(interaction_labels, dtype=logits.dtype, device=logits.device)
    delta = torch.as_tensor(delta_reward, dtype=logits.dtype, device=logits.device)
    if logits.shape != labels.shape or logits.shape != delta.shape:
        raise ValueError("Router targets must have identical shapes")
    if noise_threshold < 0 or reward_weight < 0:
        raise ValueError("noise threshold and reward weight must be non-negative")
    primary = F.binary_cross_entropy_with_logits(logits, labels)
    reliable = torch.abs(delta) > float(noise_threshold)
    if reliable.any():
        reward_target = (delta[reliable] > 0).to(logits.dtype)
        auxiliary = F.binary_cross_entropy_with_logits(logits[reliable], reward_target)
    else:
        auxiliary = logits.sum() * 0.0
    return primary + float(reward_weight) * auxiliary, primary, auxiliary, reliable


def boundary_reward_tiebreak(
    interaction_probability,
    delta_reward,
    lower=0.40,
    upper=0.60,
    delta_threshold=0.005,
):
    """Combine Gate probability and one-step reward only in an uncertain band.

    Outside [lower, upper], the learned Gate decides. Inside the band, the
    sign of a reliable one-step reward difference breaks the tie; otherwise
    the probability is preserved.
    """
    p = torch.as_tensor(interaction_probability)
    delta = torch.as_tensor(delta_reward, dtype=p.dtype, device=p.device)
    if p.shape != delta.shape or not (0.0 <= lower < upper <= 1.0):
        raise ValueError("invalid probability band or target shapes")
    if delta_threshold < 0.0:
        raise ValueError("delta threshold must be non-negative")
    uncertain = (p >= float(lower)) & (p <= float(upper))
    reliable = torch.abs(delta) > float(delta_threshold)
    reward_probability = (delta > 0).to(p.dtype)
    return torch.where(uncertain & reliable, reward_probability, p)


def combined_routing_score(interaction_logits, normalized_advantage, clip=1.0):
    if float(clip) <= 0.0:
        raise ValueError("advantage clip must be positive")
    adjusted_logits = interaction_logits + torch.clamp(
        normalized_advantage, -float(clip), float(clip)
    )
    return torch.sigmoid(adjusted_logits)


class RobustRewardAwareTemporalGate(nn.Module):
    """B2-preserving reward-aware Gate used by the G28 offline candidate.

    The GRU and interaction head are initialized from B2 and frozen during
    training.  Only the advantage head is learned from the bounded
    counterfactual reward target, so reward supervision cannot recalibrate the
    existing phase classifier.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        if input_dim < 1 or hidden_dim < 2:
            raise ValueError("robust reward-aware Gate dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        head_dim = self.hidden_dim // 2
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.interaction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, features):
        if features.ndim != 3 or features.shape[2] != self.input_dim:
            raise ValueError("robust reward-aware Gate features have the wrong shape")
        outputs, _ = self.gru(features)
        encoded = outputs[:, -1]
        return (
            self.interaction_head(encoded).squeeze(-1),
            self.advantage_head(encoded).squeeze(-1),
        )


def bounded_reward_target(delta_reward, robust_scale):
    """Map one-step reward differences to a bounded, train-split-only target."""
    scale = float(robust_scale)
    if not scale > 0.0:
        raise ValueError("robust reward scale must be positive")
    values = torch.as_tensor(delta_reward)
    return torch.tanh(torch.clamp(values, -scale, scale) / scale)


def conservative_routing_score(
    interaction_logits,
    bounded_advantage,
    switch_on_threshold=0.43,
    fusion_alpha=0.25,
    fusion_temperature=1.0,
    advantage_dead_zone=0.10,
):
    """Fuse reward only near the frozen B2 switching boundary.

    The exponential weight is one at the B2 on-threshold and decays with
    distance in logit space.  This makes the reward head a tie-breaker rather
    than a second phase classifier.
    """
    if not 0.0 < switch_on_threshold < 1.0:
        raise ValueError("switch_on_threshold must lie in (0, 1)")
    if (
        float(fusion_alpha) < 0.0
        or float(fusion_temperature) <= 0.0
        or float(advantage_dead_zone) < 0.0
    ):
        raise ValueError("fusion parameters must be non-negative and finite")
    threshold_logit = torch.log(
        torch.as_tensor(switch_on_threshold, dtype=interaction_logits.dtype, device=interaction_logits.device)
        / torch.as_tensor(1.0 - switch_on_threshold, dtype=interaction_logits.dtype, device=interaction_logits.device)
    )
    distance = torch.abs(interaction_logits - threshold_logit)
    boundary_weight = torch.exp(-distance / float(fusion_temperature))
    bounded = torch.clamp(bounded_advantage, -1.0, 1.0)
    if float(advantage_dead_zone) > 0.0:
        bounded = torch.sign(bounded) * torch.relu(
            torch.abs(bounded) - float(advantage_dead_zone)
        )
    adjusted_logits = interaction_logits + float(fusion_alpha) * boundary_weight * bounded
    return torch.sigmoid(adjusted_logits)


def selective_reward_routing_score(
    interaction_logits,
    normalized_advantage,
    switch_on_threshold=0.43,
    fusion_alpha=0.20,
    fusion_temperature=0.75,
    advantage_confidence_threshold=0.25,
):
    """Apply one-step reward evidence only near the phase boundary and above a confidence threshold."""
    if not 0.0 < float(switch_on_threshold) < 1.0:
        raise ValueError("switch_on_threshold must lie in (0,1)")
    if float(fusion_alpha) < 0.0 or float(fusion_temperature) <= 0.0:
        raise ValueError("fusion parameters must be non-negative and finite")
    if float(advantage_confidence_threshold) < 0.0:
        raise ValueError("advantage confidence threshold must be non-negative")
    threshold_logit = torch.log(
        torch.as_tensor(switch_on_threshold, dtype=interaction_logits.dtype, device=interaction_logits.device)
        / torch.as_tensor(1.0 - switch_on_threshold, dtype=interaction_logits.dtype, device=interaction_logits.device)
    )
    boundary_weight = torch.exp(-torch.abs(interaction_logits - threshold_logit) / float(fusion_temperature))
    advantage = torch.clamp(normalized_advantage, -1.0, 1.0)
    confident = (torch.abs(advantage) >= float(advantage_confidence_threshold)).to(advantage.dtype)
    adjusted_logits = interaction_logits + float(fusion_alpha) * boundary_weight * confident * advantage
    return torch.sigmoid(adjusted_logits)
