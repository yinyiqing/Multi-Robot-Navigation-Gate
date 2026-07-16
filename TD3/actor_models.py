import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        state = F.relu(self.layer_1(state))
        state = F.relu(self.layer_2(state))
        return self.tanh(self.layer_3(state))


class ResidualActor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        residual_scale=0.15,
    ):
        super(ResidualActor, self).__init__()
        if hidden_dim < 1:
            raise ValueError("Residual hidden_dim must be positive")
        if residual_scale <= 0.0 or residual_scale > 1.0:
            raise ValueError("Residual scale must be in (0, 1]")

        self.base_actor = Actor(state_dim, action_dim)
        self.adapter_layer_1 = nn.Linear(state_dim, hidden_dim)
        self.adapter_layer_2 = nn.Linear(hidden_dim, action_dim)
        self.register_buffer(
            "residual_scale_tensor", torch.tensor(float(residual_scale))
        )

        nn.init.zeros_(self.adapter_layer_2.weight)
        nn.init.zeros_(self.adapter_layer_2.bias)
        self.freeze_base_actor()

    def freeze_base_actor(self):
        for parameter in self.base_actor.parameters():
            parameter.requires_grad = False

    def load_base_state_dict(self, state_dict):
        self.base_actor.load_state_dict(state_dict)
        self.freeze_base_actor()

    def residual(self, state):
        hidden = F.relu(self.adapter_layer_1(state))
        return self.residual_scale_tensor * torch.tanh(self.adapter_layer_2(hidden))

    @property
    def residual_scale(self):
        return float(self.residual_scale_tensor.item())

    def forward(self, state):
        base_action = self.base_actor(state)
        return torch.clamp(base_action + self.residual(state), -1.0, 1.0)


def is_residual_actor_state_dict(state_dict):
    return any(key.startswith("base_actor.") for key in state_dict)
