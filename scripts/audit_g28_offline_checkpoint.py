#!/usr/bin/env python3
"""Audit that the G28 checkpoint preserves the frozen B2 phase branch."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from g27_dataset import canonical_sha256, sha256_file
from reward_aware_gate import RobustRewardAwareTemporalGate
from temporal_interaction_gate import TemporalInteractionGate


B2 = ROOT / "experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
DEFAULT_CHECKPOINT = ROOT / "experiments/03_保留专门化/02_论文主线/28_RewardAwareGate稳健修正版/local_data/training/seed20260911_final/best.pt"
DECISION = ROOT / "experiments/03_保留专门化/02_论文主线/28_RewardAwareGate稳健修正版/local_data/protocol/router_decision.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    b2_payload = torch.load(B2, map_location="cpu", weights_only=False)
    model = RobustRewardAwareTemporalGate(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    source = TemporalInteractionGate(**b2_payload["model_config"])
    source.load_state_dict(b2_payload["model_state_dict"])
    diffs = {}
    for name, value in model.gru.state_dict().items():
        diffs["gru." + name] = float(torch.max(torch.abs(value - source.gru.state_dict()[name])).item())
    for name, value in model.interaction_head.state_dict().items():
        diffs["interaction_head." + name] = float(torch.max(torch.abs(value - source.head.state_dict()[name])).item())
    decision_hash = sha256_file(DECISION)
    result = {
        "format_version": 1,
        "protocol": "G28-offline-checkpoint-audit-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "b2_checkpoint_sha256": sha256_file(B2),
        "frozen_b2_parameter_max_abs_diff": max(diffs.values()),
        "frozen_b2_parameter_check_passed": max(diffs.values()) <= 1e-7,
        "decision_sha256": decision_hash,
        "embedded_decision_hash_matches": payload.get("training_data", {}).get("decision_sha256") == decision_hash,
    }
    result["audit_passed"] = bool(result["frozen_b2_parameter_check_passed"] and result["embedded_decision_hash_matches"])
    result["summary_sha256"] = canonical_sha256(result)
    output = args.output or args.checkpoint.parents[2] / "validation" / "offline_checkpoint_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Wrote", output)


if __name__ == "__main__":
    main()
