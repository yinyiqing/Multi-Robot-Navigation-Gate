import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim_1=800, hidden_dim_2=600):
        super(Actor, self).__init__()
        self.hidden_dim_1 = int(hidden_dim_1)
        self.hidden_dim_2 = int(hidden_dim_2)
        if self.hidden_dim_1 < 1 or self.hidden_dim_2 < 1:
            raise ValueError("Actor hidden dimensions must be positive")

        self.layer_1 = nn.Linear(state_dim, self.hidden_dim_1)
        self.layer_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.layer_3 = nn.Linear(self.hidden_dim_2, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        state = F.relu(self.layer_1(state))
        state = F.relu(self.layer_2(state))
        return self.tanh(self.layer_3(state))


def actor_hidden_dims_from_state_dict(state_dict):
    if is_residual_actor_state_dict(state_dict):
        state_dict = {
            key[len("base_actor.") :]: value
            for key, value in state_dict.items()
            if key.startswith("base_actor.")
        }
    required = ("layer_1.weight", "layer_2.weight", "layer_3.weight")
    if any(key not in state_dict for key in required):
        raise ValueError("Actor state dict does not contain the expected linear layers")
    hidden_dim_1 = int(state_dict["layer_1.weight"].shape[0])
    hidden_dim_2 = int(state_dict["layer_2.weight"].shape[0])
    return hidden_dim_1, hidden_dim_2


def function_preserving_expand_actor_state_dict(source_state_dict, target_actor):
    """Embed a narrower Actor while leaving added paths trainable.

    Added first-layer features cannot affect the copied second layer initially, and
    added second-layer features have zero output weights. Their internal weights stay
    randomly initialized, so gradients can activate the extra branch after warm start.
    """
    if is_residual_actor_state_dict(source_state_dict):
        raise ValueError("Cannot expand a residual Actor into a full Actor")

    source_hidden_1, source_hidden_2 = actor_hidden_dims_from_state_dict(
        source_state_dict
    )
    target_state = target_actor.state_dict()
    target_hidden_1, target_hidden_2 = actor_hidden_dims_from_state_dict(target_state)
    if target_hidden_1 < source_hidden_1 or target_hidden_2 < source_hidden_2:
        raise ValueError(
            "Target Actor must not be narrower than the warm-start Actor: "
            f"source=({source_hidden_1}, {source_hidden_2}) "
            f"target=({target_hidden_1}, {target_hidden_2})"
        )
    if target_state["layer_1.weight"].shape[1] != source_state_dict[
        "layer_1.weight"
    ].shape[1]:
        raise ValueError("Actor input dimensions do not match")
    if target_state["layer_3.weight"].shape[0] != source_state_dict[
        "layer_3.weight"
    ].shape[0]:
        raise ValueError("Actor output dimensions do not match")

    expanded = {key: value.clone() for key, value in target_state.items()}
    expanded["layer_1.weight"][:source_hidden_1].copy_(
        source_state_dict["layer_1.weight"]
    )
    expanded["layer_1.bias"][:source_hidden_1].copy_(
        source_state_dict["layer_1.bias"]
    )

    expanded["layer_2.weight"][:source_hidden_2, :source_hidden_1].copy_(
        source_state_dict["layer_2.weight"]
    )
    expanded["layer_2.bias"][:source_hidden_2].copy_(
        source_state_dict["layer_2.bias"]
    )
    expanded["layer_2.weight"][:source_hidden_2, source_hidden_1:].zero_()

    expanded["layer_3.weight"][:, :source_hidden_2].copy_(
        source_state_dict["layer_3.weight"]
    )
    expanded["layer_3.weight"][:, source_hidden_2:].zero_()
    expanded["layer_3.bias"].copy_(source_state_dict["layer_3.bias"])
    return expanded


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
