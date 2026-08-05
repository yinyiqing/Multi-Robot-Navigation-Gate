#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from actor_models import Actor, function_preserving_expand_actor_state_dict


BASE_ACTOR = (
    PROJECT_ROOT
    / "TD3/pytorch_models"
    / "TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
)
EXPECTED_BASE_SHA256 = (
    "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线"
    / "12_参数匹配单Actor容量对照/local_data/initialization_audit.json"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def actor_macs(model):
    return sum(
        layer.in_features * layer.out_features
        for layer in (model.layer_1, model.layer_2, model.layer_3)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Audit the G12 function-preserving Actor expansion."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    actual_hash = sha256(BASE_ACTOR)
    if actual_hash != EXPECTED_BASE_SHA256:
        raise ValueError(
            f"5A Actor hash mismatch: expected={EXPECTED_BASE_SHA256} actual={actual_hash}"
        )

    source_state = torch.load(BASE_ACTOR, map_location="cpu", weights_only=False)
    base_actor = Actor(24, 2)
    base_actor.load_state_dict(source_state)

    torch.manual_seed(20260810)
    wide_actor = Actor(24, 2, hidden_dim_1=1137, hidden_dim_2=855)
    wide_actor.load_state_dict(
        function_preserving_expand_actor_state_dict(source_state, wide_actor)
    )

    generator = torch.Generator(device="cpu").manual_seed(20260810)
    states = torch.randn(8192, 24, generator=generator)
    with torch.no_grad():
        difference = (wide_actor(states) - base_actor(states)).abs()

    loss = wide_actor(states[:128]).square().mean()
    loss.backward()
    added_output_gradient = float(
        wide_actor.layer_3.weight.grad[:, 600:].abs().sum().item()
    )
    base_parameters = parameter_count(base_actor)
    wide_parameters = parameter_count(wide_actor)
    two_actor_parameters = 2 * base_parameters
    base_macs = actor_macs(base_actor)
    wide_macs = actor_macs(wide_actor)
    max_abs_output_error = float(difference.max().item())
    passed = bool(
        base_parameters == 501_802
        and wide_parameters == 1_003_127
        and two_actor_parameters - wide_parameters == 477
        and base_macs == 500_400
        and wide_macs - 2 * base_macs == 333
        and max_abs_output_error <= 1e-6
        and added_output_gradient > 0.0
    )
    result = {
        "protocol": "g12-capacity-matched-function-preserving-expansion-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_actor": str(BASE_ACTOR.relative_to(PROJECT_ROOT)),
        "base_actor_sha256": actual_hash,
        "base_architecture": [24, 800, 600, 2],
        "wide_architecture": [24, 1137, 855, 2],
        "base_parameters": base_parameters,
        "two_actor_parameters": two_actor_parameters,
        "wide_parameters": wide_parameters,
        "parameter_gap": two_actor_parameters - wide_parameters,
        "base_macs_per_action": base_macs,
        "two_actor_macs_per_step": 2 * base_macs,
        "wide_macs_per_action": wide_macs,
        "mac_gap": wide_macs - 2 * base_macs,
        "audit_states": int(states.shape[0]),
        "max_abs_output_error": max_abs_output_error,
        "mean_abs_output_error": float(difference.mean().item()),
        "added_output_gradient_l1": added_output_gradient,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("G12 Actor initialization audit failed")


if __name__ == "__main__":
    main()
