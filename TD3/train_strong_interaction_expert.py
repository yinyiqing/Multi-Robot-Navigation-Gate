import json
import os
import random
import socket
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from interaction_expert import StrongInteractionTD3
from multi_agent_velodyne_env import MultiAgentGazeboEnv
from outcome_utils import resolve_terminal_outcome
from sequence_replay_buffer import SequenceReplayBuffer


def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)


def env_float(name, default):
    value = os.environ.get(name)
    return default if value is None or not value.strip() else float(value)


def make_histories(states, history_len):
    return {
        index: deque(
            [np.asarray(state, dtype=np.float32)] * history_len,
            maxlen=history_len,
        )
        for index, state in enumerate(states)
    }


def history_array(histories, index):
    return np.stack(histories[index]).astype(np.float32, copy=False)


def empty_metrics():
    return {
        "episodes": 0,
        "successes": 0,
        "collisions": 0,
        "unresolved": 0,
        "full_successes": 0,
        "timeouts": 0,
        "steps": 0,
    }


def finalize_metrics(values, num_agents):
    episodes = values["episodes"]
    agents = episodes * num_agents
    return {
        "episodes": episodes,
        "agent_success_rate": values["successes"] / agents,
        "collision_rate": values["collisions"] / agents,
        "unresolved_rate": values["unresolved"] / agents,
        "full_success_rate": values["full_successes"] / episodes,
        "timeout_rate": values["timeouts"] / episodes,
        "mean_steps": values["steps"] / episodes,
    }


def evaluate(agent, env, manifest_path, train_manifest_path, history_len, max_steps):
    os.environ["DRL_MULTI_MANIFEST_SAMPLING"] = "cycle"
    env.set_manifest_path(manifest_path)
    totals = {band: empty_metrics() for band in ("deep", "close", "margin")}
    episode_count = len(env.curriculum_cases)
    try:
        for episode_index in range(episode_count):
            states = env.reset()
            band = env.current_curriculum_case["view"]["interaction_band"]
            histories = make_histories(states, history_len)
            active = [True] * env.num_agents
            success_flags = np.zeros(env.num_agents, dtype=np.int32)
            collision_flags = np.zeros(env.num_agents, dtype=np.int32)
            steps = 0
            while any(active) and steps < max_steps:
                actions = []
                for index in range(env.num_agents):
                    if not active[index]:
                        actions.append([0.0, 0.0])
                        continue
                    raw_action = agent.select_action(history_array(histories, index))
                    actions.append([(raw_action[0] + 1.0) / 2.0, raw_action[1]])
                next_states, _, dones, targets, collisions = env.step(actions, active)
                for index, next_state in enumerate(next_states):
                    histories[index].append(np.asarray(next_state, dtype=np.float32))
                    if not active[index]:
                        continue
                    success, collision = resolve_terminal_outcome(
                        success_flags[index],
                        collision_flags[index],
                        targets[index],
                        collisions[index],
                    )
                    success_flags[index] = int(success)
                    collision_flags[index] = int(collision)
                    if dones[index]:
                        active[index] = False
                steps += 1
                states = next_states
            values = totals[band]
            success_count = int(success_flags.sum())
            collision_count = int(collision_flags.sum())
            values["episodes"] += 1
            values["successes"] += success_count
            values["collisions"] += collision_count
            values["unresolved"] += max(
                env.num_agents - success_count - collision_count, 0
            )
            values["full_successes"] += int(success_count == env.num_agents)
            values["timeouts"] += int(steps >= max_steps)
            values["steps"] += steps
            if (episode_index + 1) % 20 == 0 or episode_index + 1 == episode_count:
                print(
                    "Validation progress | episodes=%d/%d"
                    % (episode_index + 1, episode_count)
                )
    finally:
        os.environ["DRL_MULTI_MANIFEST_SAMPLING"] = "random"
        env.set_manifest_path(train_manifest_path)
    return {band: finalize_metrics(values, env.num_agents) for band, values in totals.items()}


def candidate_accepted(candidate, baseline):
    return (
        candidate["deep"]["full_success_rate"]
        >= baseline["deep"]["full_success_rate"] + 0.15
        and candidate["deep"]["collision_rate"]
        < baseline["deep"]["collision_rate"]
        and candidate["close"]["full_success_rate"]
        >= baseline["close"]["full_success_rate"] - 0.05
        and candidate["margin"]["full_success_rate"]
        >= baseline["margin"]["full_success_rate"] - 0.03
    )


seed = env_int("DRL_STRONG_SEED", 20260723)
history_len = env_int("DRL_STRONG_HISTORY_LEN", 8)
hidden_dim = env_int("DRL_STRONG_HIDDEN_DIM", 128)
max_agent_samples = env_int("DRL_STRONG_MAX_AGENT_SAMPLES", 40000)
eval_interval = env_int("DRL_STRONG_EVAL_INTERVAL", 20000)
actor_start_samples = env_int("DRL_STRONG_ACTOR_START_SAMPLES", 8000)
learning_starts = env_int("DRL_STRONG_LEARNING_STARTS", 2000)
max_episode_steps = env_int("DRL_STRONG_MAX_EPISODE_STEPS", 300)
batch_size = env_int("DRL_STRONG_BATCH_SIZE", 64)
exploration_noise = env_float("DRL_STRONG_EXPLORATION_NOISE", 0.05)
actor_lr = env_float("DRL_STRONG_ACTOR_LR", 1e-5)
critic_lr = env_float("DRL_STRONG_CRITIC_LR", 5e-5)
model_name = os.environ.get(
    "DRL_STRONG_MODEL_NAME", "interaction_expert_temporal_gru_pilot_s20260723"
)
base_model = os.environ.get(
    "DRL_STRONG_BASE_MODEL",
    "TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best",
)
train_manifest = os.environ["DRL_MULTI_MANIFEST_PATH"]
validation_manifest = os.environ["DRL_MULTI_EVAL_MANIFEST_PATH"]
launchfile = os.environ.get(
    "DRL_STRONG_LAUNCHFILE", "multi_robot_scenario_strong_interaction_pilot_5.launch"
)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

base_actor_path = Path("pytorch_models") / f"{base_model}_actor.pth"
base_actor_state = torch.load(base_actor_path, map_location="cpu", weights_only=True)
agent = StrongInteractionTD3(
    base_actor_state,
    history_len=history_len,
    hidden_dim=hidden_dim,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
)
replay = SequenceReplayBuffer(
    200000,
    seed=seed,
    group_ratios={"deep": 0.6, "close": 0.2, "margin": 0.2},
)
env = MultiAgentGazeboEnv(
    launchfile,
    20,
    agent_names=[f"r{index}" for index in range(1, 6)],
    cooperative_reward=True,
    cooperative_reward_self_weight=0.8,
    cooperative_reward_distance_weighted=True,
    cooperative_reward_sigma=2.0,
    cooperative_reward_mode="average_plus_interaction",
    interaction_safe_distance=1.2,
    interaction_close_penalty=2.0,
    interaction_stagnation_penalty=0.1,
    weak_coupling_layout=True,
    scenario_mode="manifest",
    active_neighbors_only=True,
)
time.sleep(5)

checkpoint_path = Path("checkpoints") / f"{model_name}_latest.pt"
best_path = Path("checkpoints") / f"{model_name}_best.pt"
result_path = Path("results") / f"{model_name}.json"
for directory in (Path("checkpoints"), Path("pytorch_models"), Path("results"), Path("runs")):
    directory.mkdir(exist_ok=True)
writer = SummaryWriter(
    log_dir=str(Path("runs") / f"strong_{datetime.now():%b%d_%H-%M-%S}_{socket.gethostname()}")
)

agent_samples = 0
environment_steps = 0
episode = 0
next_eval = eval_interval
evaluations = []


def save_checkpoint(path, baseline):
    torch.save(
        {
            "agent": agent.state_dict(),
            "replay_buffer": replay.state_dict(),
            "agent_samples": agent_samples,
            "environment_steps": environment_steps,
            "episode": episode,
            "baseline": baseline,
            "evaluations": evaluations,
            "config": {
                "version": 1,
                "base_model": base_model,
                "history_len": history_len,
                "hidden_dim": hidden_dim,
                "train_manifest": train_manifest,
                "validation_manifest": validation_manifest,
                "seed": seed,
            },
        },
        path,
    )


print("==============================================")
print("Strong-interaction expert pilot")
print("Weak-interaction actor: frozen", base_model)
print("Temporal history / hidden:", history_len, "/", hidden_dim)
print("Reward: 0.8 self + 0.2 distance-weighted neighbor + interaction shaping")
print("Train manifest:", train_manifest)
print("Validation manifest:", validation_manifest)
print("Actor starts at agent samples:", actor_start_samples)
print("Max agent samples:", max_agent_samples)
print("Device:", agent.device)
print("==============================================")

baseline = evaluate(
    agent, env, validation_manifest, train_manifest, history_len, max_episode_steps
)
evaluations.append({"agent_samples": 0, "metrics": baseline, "accepted": False})
print("Frozen-5D same-protocol baseline:", json.dumps(baseline, ensure_ascii=False))
save_checkpoint(checkpoint_path, baseline)

try:
    while agent_samples < max_agent_samples:
        states = env.reset()
        band = env.current_curriculum_case["view"]["interaction_band"]
        histories = make_histories(states, history_len)
        active = [True] * env.num_agents
        episode_success = np.zeros(env.num_agents, dtype=np.int32)
        episode_collision = np.zeros(env.num_agents, dtype=np.int32)
        episode += 1
        for step in range(max_episode_steps):
            current_histories = {
                index: history_array(histories, index) for index in range(env.num_agents)
            }
            raw_actions = []
            env_actions = []
            for index in range(env.num_agents):
                if not active[index]:
                    raw_actions.append(np.zeros(2, dtype=np.float32))
                    env_actions.append([0.0, 0.0])
                    continue
                raw_action = agent.select_action(current_histories[index])
                raw_action = np.clip(
                    raw_action + np.random.normal(0.0, exploration_noise, 2), -1.0, 1.0
                ).astype(np.float32)
                raw_actions.append(raw_action)
                env_actions.append([(raw_action[0] + 1.0) / 2.0, raw_action[1]])

            next_states, rewards, dones, targets, collisions = env.step(env_actions, active)
            environment_steps += 1
            truncated = step + 1 >= max_episode_steps
            for index, next_state in enumerate(next_states):
                histories[index].append(np.asarray(next_state, dtype=np.float32))
                if not active[index]:
                    continue
                replay.add(
                    current_histories[index],
                    raw_actions[index],
                    rewards[index],
                    dones[index] or truncated,
                    history_array(histories, index),
                    band,
                )
                agent_samples += 1
                success, collision = resolve_terminal_outcome(
                    episode_success[index],
                    episode_collision[index],
                    targets[index],
                    collisions[index],
                )
                episode_success[index] = int(success)
                episode_collision[index] = int(collision)
                if dones[index] or truncated:
                    active[index] = False
            if len(replay) >= learning_starts:
                metrics = agent.train_step(
                    replay,
                    batch_size=batch_size,
                    update_actor=agent_samples >= actor_start_samples,
                )
                for name, value in metrics.items():
                    writer.add_scalar(f"optimization/{name}", value, environment_steps)
            if not any(active) or agent_samples >= max_agent_samples:
                break

        print(
            "Episode %d | band=%s | agent_samples=%d | env_steps=%d | success=%d/5 | collision=%d/5"
            % (
                episode,
                band,
                agent_samples,
                environment_steps,
                int(episode_success.sum()),
                int(episode_collision.sum()),
            )
        )
        if agent_samples < next_eval and agent_samples < max_agent_samples:
            if episode % 10 == 0:
                save_checkpoint(checkpoint_path, baseline)
            continue

        candidate = evaluate(
            agent, env, validation_manifest, train_manifest, history_len, max_episode_steps
        )
        accepted = candidate_accepted(candidate, baseline)
        evaluations.append(
            {"agent_samples": agent_samples, "metrics": candidate, "accepted": accepted}
        )
        print("Validation:", json.dumps(candidate, ensure_ascii=False))
        print("Acceptance criteria met:", accepted)
        for band_name, band_metrics in candidate.items():
            for metric_name, value in band_metrics.items():
                if metric_name != "episodes":
                    writer.add_scalar(
                        f"validation/{band_name}_{metric_name}", value, agent_samples
                    )
        if accepted:
            torch.save(agent.actor.state_dict(), Path("pytorch_models") / f"{model_name}_actor.pth")
            save_checkpoint(best_path, baseline)
        save_checkpoint(checkpoint_path, baseline)
        result_path.write_text(
            json.dumps(
                {"baseline": baseline, "evaluations": evaluations},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        next_eval += eval_interval
finally:
    save_checkpoint(checkpoint_path, baseline)
    writer.close()
