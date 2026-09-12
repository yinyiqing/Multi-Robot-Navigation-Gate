#!/usr/bin/env python3
"""Validate the frozen G34 artifact and create a metadata-correct runtime copy."""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from learned_gate_controller import LearnedInteractionGateController


BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
TRAINING = BASE / "34_双头RewardAwareGate/local_data/training/seed20260912"
SOURCE = TRAINING / "best.pt"
RUNTIME = TRAINING / "best_runtime.pt"
SUMMARY = TRAINING / "summary.json"
AUDIT = TRAINING / "checkpoint_audit.json"
DETECTOR = (
    BASE
    / "results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class FixedPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def get_action(self, _state):
        return self.action.copy()


def state_dict_equal(left, right):
    if set(left) != set(right):
        return False
    return all(torch.equal(left[name], right[name]) for name in left)


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    expected_summary_hash = summary.pop("summary_sha256")
    if canonical_sha256(summary) != expected_summary_hash:
        raise ValueError("G34 summary canonical hash mismatch")
    summary["summary_sha256"] = expected_summary_hash

    source_hash = sha256(SOURCE)
    if summary["checkpoint"]["sha256"] != source_hash:
        raise ValueError("G34 source checkpoint hash mismatch")
    for relative_path, expected_hash in summary["inputs"].items():
        path = ROOT / relative_path
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError("G34 frozen input hash mismatch: %s" % relative_path)
    if not summary.get("offline_admission_passed"):
        raise ValueError("G34 did not pass offline admission")

    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if source.get("model_id") not in (
        "G33-dual-encoder-reward-aware",
        "G34-dual-encoder-reward-aware",
    ):
        raise ValueError("unexpected G34 source model id")
    runtime = dict(source)
    runtime["model_id"] = "G34-dual-encoder-reward-aware"
    if RUNTIME.exists():
        saved_runtime = torch.load(RUNTIME, map_location="cpu", weights_only=False)
        if saved_runtime.get("model_id") != runtime["model_id"]:
            raise ValueError("G34 runtime checkpoint has the wrong model id")
        if not state_dict_equal(
            saved_runtime["model_state_dict"], source["model_state_dict"]
        ):
            raise ValueError("G34 runtime weights differ from the trained checkpoint")
        runtime = saved_runtime
    else:
        torch.save(runtime, RUNTIME)

    controller = LearnedInteractionGateController(
        FixedPolicy([-0.5, 0.1]),
        FixedPolicy([0.5, -0.2]),
        DETECTOR,
        RUNTIME,
        "cpu",
        switch_on_threshold=0.43,
        switch_off_threshold=0.33,
        minimum_hold_steps=3,
        evaluation_stride=2,
    )
    controller.reset(["r1"])
    probes = [
        controller.feature_mean,
        controller.feature_mean + controller.feature_std,
        controller.feature_mean - controller.feature_std,
    ]
    outputs = [controller._gate_outputs("r1", probe) for probe in probes]
    finite = all(math.isfinite(value) for row in outputs for value in row)
    bounded = all(
        0.0 <= row[0] <= 1.0
        and 0.0 <= row[1] <= 1.0
        and -1.0 <= row[2] <= 1.0
        for row in outputs
    )
    if not finite or not bounded:
        raise ValueError("G34 runtime outputs are invalid")

    result = {
        "format_version": 1,
        "protocol": "G34-checkpoint-audit-v1",
        "passed": True,
        "source_checkpoint": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": source_hash,
            "model_id": source["model_id"],
        },
        "runtime_checkpoint": {
            "path": str(RUNTIME.relative_to(ROOT)),
            "sha256": sha256(RUNTIME),
            "model_id": runtime["model_id"],
            "weights_exactly_equal_to_source": state_dict_equal(
                runtime["model_state_dict"], source["model_state_dict"]
            ),
        },
        "normalization": {
            "finite": bool(
                np.all(np.isfinite(controller.feature_mean))
                and np.all(np.isfinite(controller.feature_std))
            ),
            "strictly_positive_scale": bool(np.all(controller.feature_std > 0.0)),
            "input_dim": int(len(controller.feature_mean)),
        },
        "runtime_smoke_outputs": [list(map(float, row)) for row in outputs],
        "deployment_contract": {
            "inputs": [
                "ego local state",
                "local LiDAR candidates and deterministic tracking",
                "two frozen-Actor candidate actions",
                "eight-frame local feature history",
            ],
            "true_interaction_label_used": False,
            "true_counterfactual_reward_difference_used": False,
            "predictions_fused": [
                "interaction logit",
                "bounded one-step reward advantage",
            ],
        },
    }
    result["audit_sha256"] = canonical_sha256(result)
    AUDIT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
