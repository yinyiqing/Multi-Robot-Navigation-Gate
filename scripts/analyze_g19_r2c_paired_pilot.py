#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

import sys


ROOT = Path(__file__).resolve().parents[1]
TD3 = ROOT / "TD3"
RUN_DIR = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线/19_R2C公平容量对照/local_data"
)
SOURCE = TD3 / "pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
MODELS = {
    "original": "capacity_original_g19_r2c_n5_seed20260826",
    "wide": "capacity_wide_g19_r2c_n5_seed20260826",
}


sys.path.insert(0, str(TD3))
from actor_models import Actor, function_preserving_expand_actor_state_dict


def actor_delta(actor, reference):
    common = []
    for name, tensor in actor.items():
        if name in reference and tensor.shape == reference[name].shape:
            common.append((tensor.float() - reference[name].float()).abs().reshape(-1))
    values = torch.cat(common)
    return {
        "common_mean_abs_delta": float(values.mean()),
        "common_max_abs_delta": float(values.max()),
    }


def read_model(kind, source):
    path = TD3 / "checkpoints" / f"{MODELS[kind]}_latest.pt"
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    evaluations = checkpoint.get("evaluations", [])
    if len(evaluations) != 3:
        raise ValueError(f"{kind} expected 3 evaluations, got {len(evaluations)}")
    records = []
    for sample_boundary, row in zip((20000, 40000, 60000), evaluations):
        records.append(
            {
                "sample_boundary": sample_boundary,
                "reward": float(row[0]),
                "agent_success": float(row[1]),
                "collision": float(row[2]),
                "mean_steps": float(row[3]),
                "mean_final_distance": float(row[4]),
                "unresolved": float(row[5]),
                "full_success": float(row[6]),
                "timeout": float(row[7]),
            }
        )
    frozen = records[1]
    trained = records[2]
    if kind == "wide":
        target = Actor(24, 2, hidden_dim_1=1137, hidden_dim_2=855)
        reference = function_preserving_expand_actor_state_dict(source, target)
    else:
        reference = source
    delta = actor_delta(checkpoint["network"]["actor"], reference)
    stability = {
        "full_success_delta": trained["full_success"] - frozen["full_success"],
        "agent_success_delta": trained["agent_success"] - frozen["agent_success"],
        "collision_delta": trained["collision"] - frozen["collision"],
        "timeout_delta": trained["timeout"] - frozen["timeout"],
    }
    passed = (
        stability["full_success_delta"] >= -0.05
        and stability["agent_success_delta"] >= -0.03
        and stability["collision_delta"] <= 0.03
        and stability["timeout_delta"] <= 0.02
        and delta["common_max_abs_delta"] > 1e-6
    )
    return {
        "checkpoint": str(path.relative_to(ROOT)),
        "timestep": int(checkpoint["timestep"]),
        "evaluations": records,
        "actor_delta": delta,
        "stability": stability,
        "stability_passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-control-pass", action="store_true")
    args = parser.parse_args()
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    results = {kind: read_model(kind, source) for kind in MODELS}
    summary = {
        "protocol": {
            "experiment_id": "G19-R2C-paired-stability-pilot",
            "seed": 20260826,
            "actor_update_boundary": 41000,
            "samples_per_model": 60000,
            "sealed_test_read": False,
        },
        "models": results,
    }
    if results["original"] and results["wide"]:
        original = results["original"]["evaluations"][2]
        wide = results["wide"]["evaluations"][2]
        summary["trained_wide_minus_original"] = {
            key: wide[key] - original[key]
            for key in ("full_success", "agent_success", "collision", "timeout", "mean_steps")
        }
        summary["capacity_candidate_passed"] = bool(
            results["wide"]["stability_passed"]
            and wide["full_success"] > original["full_success"]
            and wide["collision"] <= original["collision"] + 0.02
            and wide["timeout"] <= original["timeout"] + 0.02
        )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output = RUN_DIR / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_control_pass and (
        not results["original"] or not results["original"]["stability_passed"]
    ):
        raise SystemExit("R2C original-width stability gate failed; wide run is blocked")


if __name__ == "__main__":
    main()
