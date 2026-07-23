import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from actor_models import Actor


class TemporalStrongActor(nn.Module):
    def __init__(
        self,
        base_actor_state,
        state_dim=24,
        action_dim=2,
        history_len=8,
        hidden_dim=128,
        linear_residual_scale=0.5,
        angular_residual_scale=0.35,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.history_len = int(history_len)
        # This is an independent warm-started backbone, not the frozen weak actor.
        self.backbone = Actor(state_dim, action_dim)
        self.backbone.load_state_dict(base_actor_state)
        self.encoder = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.register_buffer(
            "residual_scales",
            torch.tensor(
                [linear_residual_scale, angular_residual_scale], dtype=torch.float32
            ),
        )
        nn.init.zeros_(self.residual_head[2].weight)
        nn.init.zeros_(self.residual_head[2].bias)

    def residual(self, history):
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, time, state]")
        if history.shape[1:] != (self.history_len, self.state_dim):
            raise ValueError(
                "expected history tail (%d, %d), got %s"
                % (self.history_len, self.state_dim, tuple(history.shape[1:]))
            )
        _, hidden = self.encoder(history)
        return self.residual_head(hidden[-1]) * self.residual_scales

    def forward(self, history, return_details=False):
        base_action = self.backbone(history[:, -1])
        residual = self.residual(history)
        action = torch.clamp(base_action + residual, -1.0, 1.0)
        if return_details:
            return action, base_action, residual
        return action


class TemporalQNetwork(nn.Module):
    def __init__(self, state_dim=24, action_dim=2, hidden_dim=128):
        super().__init__()
        self.encoder = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, history, action):
        _, hidden = self.encoder(history)
        return self.value(torch.cat((hidden[-1], action), dim=-1))


class TwinTemporalCritic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.q1 = TemporalQNetwork(**kwargs)
        self.q2 = TemporalQNetwork(**kwargs)

    def forward(self, history, action):
        return self.q1(history, action), self.q2(history, action)

    def first(self, history, action):
        return self.q1(history, action)


class StrongInteractionTD3:
    def __init__(
        self,
        base_actor_state,
        history_len=8,
        state_dim=24,
        action_dim=2,
        hidden_dim=128,
        actor_lr=1e-5,
        critic_lr=5e-5,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        actor_kwargs = {
            "base_actor_state": base_actor_state,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "history_len": history_len,
            "hidden_dim": hidden_dim,
        }
        self.actor = TemporalStrongActor(**actor_kwargs).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.weak_reference = Actor(state_dim, action_dim).to(self.device)
        self.weak_reference.load_state_dict(base_actor_state)
        self.weak_reference.eval()
        for parameter in self.weak_reference.parameters():
            parameter.requires_grad = False
        critic_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
        }
        self.critic = TwinTemporalCritic(**critic_kwargs).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.total_updates = 0

    def select_action(self, history):
        value = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(value.unsqueeze(0))
        self.actor.train()
        return action.cpu().numpy().reshape(-1)

    def train_step(
        self,
        replay_buffer,
        batch_size=64,
        discount=0.999,
        tau=0.005,
        policy_noise=0.1,
        noise_clip=0.25,
        policy_delay=2,
        reward_scale=0.1,
        update_actor=True,
    ):
        histories, actions, rewards, dones, next_histories, groups = replay_buffer.sample(
            batch_size
        )
        history = torch.as_tensor(histories, device=self.device)
        action = torch.as_tensor(actions, device=self.device)
        reward = torch.as_tensor(rewards, device=self.device) * reward_scale
        done = torch.as_tensor(dones, device=self.device)
        next_history = torch.as_tensor(next_histories, device=self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(
                -noise_clip, noise_clip
            )
            next_action = (self.actor_target(next_history) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_history, next_action)
            target_q = reward + (1.0 - done) * discount * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(history, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        self.total_updates += 1
        metrics = {"critic_loss": float(critic_loss.detach().cpu())}
        if update_actor and self.total_updates % policy_delay == 0:
            self._set_requires_grad(self.critic, False)
            actor_action, backbone_action, residual = self.actor(
                history, return_details=True
            )
            with torch.no_grad():
                weak_action = self.weak_reference(history[:, -1])
            actor_q = self.critic.first(history, actor_action)
            q_scale = actor_q.detach().abs().mean().clamp(min=1.0)
            policy_loss = -actor_q.mean() / q_scale
            group_weights = {"deep": 0.05, "close": 0.25, "margin": 1.0}
            weights = torch.as_tensor(
                [group_weights.get(str(group), 1.0) for group in groups],
                dtype=history.dtype,
                device=self.device,
            ).unsqueeze(1)
            preserve_loss = (weights * (actor_action - weak_action).pow(2)).mean()
            residual_loss = residual.pow(2).mean()
            actor_loss = policy_loss + preserve_loss + 0.05 * residual_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.actor.parameters() if p.requires_grad), 1.0
            )
            self.actor_optimizer.step()
            self._set_requires_grad(self.critic, True)
            self._soft_update(self.actor, self.actor_target, tau)
            metrics.update(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "preserve_loss": float(preserve_loss.detach().cpu()),
                    "residual_linear_mean": float(residual[:, 0].mean().detach().cpu()),
                    "residual_angular_mean": float(residual[:, 1].mean().detach().cpu()),
                    "backbone_drift": float(
                        (backbone_action - weak_action).abs().mean().detach().cpu()
                    ),
                }
            )
        self._soft_update(self.critic, self.critic_target, tau)
        return metrics

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.total_updates = int(state["total_updates"])

    @staticmethod
    def _soft_update(source, target, tau):
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.mul_(1.0 - tau)
            target_parameter.data.add_(tau * source_parameter.data)

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad = enabled
