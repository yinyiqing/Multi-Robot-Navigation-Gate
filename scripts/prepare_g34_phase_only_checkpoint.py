#!/usr/bin/env python3
"""Create the same-weight G34 checkpoint used for the phase-only ablation."""
import copy
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
SOURCE = BASE / "34_双头RewardAwareGate/local_data/training/seed20260912/best_runtime.pt"
RUN = BASE / "37_最终双监督消融"
TARGET = RUN / "local_data/checkpoint/g34_phase_only.pt"
RECORD = RUN / "local_data/checkpoint/record.json"
EXPECTED_SOURCE_SHA256 = "4cb3626289c5b353d3ccf858f02c0e06d828106aeadc9f551f62ed6c90e4e95d"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise ValueError("frozen G34 checkpoint hash mismatch")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if source.get("model_id") != "G34-dual-encoder-reward-aware":
        raise ValueError("unexpected G34 model id")
    decision = source.get("router_decision", {})
    if decision.get("reward_weight") != 0.25:
        raise ValueError("source checkpoint does not use the frozen reward weight")

    phase_only = copy.deepcopy(source)
    phase_only["label"] = "G34 same-checkpoint phase-only decision ablation"
    phase_only["router_decision"]["reward_weight"] = 0.0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    torch.save(phase_only, TARGET)
    loaded = torch.load(TARGET, map_location="cpu", weights_only=False)
    if loaded["router_decision"]["reward_weight"] != 0.0:
        raise ValueError("phase-only checkpoint did not retain reward_weight=0")
    if loaded["model_state_dict"].keys() != source["model_state_dict"].keys():
        raise ValueError("model state keys changed")
    for name, tensor in source["model_state_dict"].items():
        if not torch.equal(tensor, loaded["model_state_dict"][name]):
            raise ValueError("model weight changed: %s" % name)

    record = {
        "format_version": 1,
        "experiment_id": "G34-final-phase-only-ablation",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "derived_checkpoint": str(TARGET.relative_to(ROOT)),
        "derived_sha256": sha256(TARGET),
        "model_weights_identical": True,
        "only_runtime_metadata_change": {"router_decision.reward_weight": [0.25, 0.0]},
        "actor_or_router_trained": False,
    }
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
