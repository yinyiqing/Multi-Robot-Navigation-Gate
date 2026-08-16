#!/usr/bin/env python3
"""Reject any G19 resume checkpoint that is not the audited pre-unfreeze state."""

import argparse
from pathlib import Path

import torch


EXPECTED_EVAL_PROTOCOL = (
    "eval-v1|scenario=manifest|episodes=120|max_steps=300|manifest_sha256="
    "52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635"
    "|sampling=cycle"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    network = checkpoint["network"]
    replay = checkpoint["replay_buffer"]["buffer"]
    expected = {
        "timestep": 33430,
        "env_step_count": 14046,
        "timesteps_since_eval": 13430,
        "episode_num": 620,
        "epoch": 2,
        "evaluation_count": 1,
        "replay_size": 33430,
        "actor_optimizer_states": 0,
        "actor_hidden_dim_1": 800,
        "actor_hidden_dim_2": 600,
        "eval_protocol_id": EXPECTED_EVAL_PROTOCOL,
    }
    actual = {
        "timestep": checkpoint.get("timestep"),
        "env_step_count": checkpoint.get("env_step_count"),
        "timesteps_since_eval": checkpoint.get("timesteps_since_eval"),
        "episode_num": checkpoint.get("episode_num"),
        "epoch": checkpoint.get("epoch"),
        "evaluation_count": len(checkpoint.get("evaluations", [])),
        "replay_size": len(replay),
        "actor_optimizer_states": len(network["actor_optimizer"]["state"]),
        "actor_hidden_dim_1": network.get("actor_hidden_dim_1"),
        "actor_hidden_dim_2": network.get("actor_hidden_dim_2"),
        "eval_protocol_id": checkpoint.get("eval_protocol_id"),
    }
    if actual != expected:
        raise SystemExit(f"G19 resume audit failed: expected={expected}, actual={actual}")
    print(f"G19 resume audit passed: {actual}")


if __name__ == "__main__":
    main()
