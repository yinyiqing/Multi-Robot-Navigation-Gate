#!/usr/bin/env python3
"""Fork the audited pre-update R2B checkpoint with fixed stability objectives."""

import argparse
import copy
from pathlib import Path

import torch


EXPECTED_PROTOCOL = (
    "eval-v1|scenario=standard|episodes=120|max_steps=300|manifest_sha256="
    "e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7"
    "|sampling=cycle"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite R2D checkpoint: {args.output}")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    network = checkpoint["network"]
    actual = {
        "timestep": checkpoint.get("timestep"),
        "epoch": checkpoint.get("epoch"),
        "replay_size": len(checkpoint["replay_buffer"]["buffer"]),
        "actor_optimizer_states": len(network["actor_optimizer"]["state"]),
        "actor_hidden_dims": (
            network.get("actor_hidden_dim_1"),
            network.get("actor_hidden_dim_2"),
        ),
        "eval_protocol_id": checkpoint.get("eval_protocol_id"),
    }
    expected = {
        "timestep": 10086,
        "epoch": 1,
        "replay_size": 10086,
        "actor_optimizer_states": 0,
        "actor_hidden_dims": (1137, 855),
        "eval_protocol_id": EXPECTED_PROTOCOL,
    }
    if actual != expected:
        raise SystemExit(f"R2B source audit failed: expected={expected}, actual={actual}")

    network["actor_reference"] = copy.deepcopy(network["actor"])
    network["actor_anchor_weight"] = 1.0
    network["actor_q_normalization_alpha"] = 1.0
    network["actor_grad_norm_clip"] = 1.0
    checkpoint["evaluations"] = []
    checkpoint["evaluation_history"] = []
    checkpoint["best_eval_summary"] = None
    checkpoint["best_epoch"] = None
    checkpoint["bad_eval_count"] = 0
    checkpoint["epoch"] = 1
    checkpoint["fork_provenance"] = {
        "protocol": "r2d-pre5a-stable-v1",
        "source": args.source.name,
        "source_timestep": 10086,
        "actor_reference": "source actor at fork",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"R2D checkpoint forked: {args.output}")
    print(actual)


if __name__ == "__main__":
    main()
