#!/usr/bin/env python3
import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))
sys.path.insert(0, str(ROOT / "scripts"))

from robot_perception.gate_features import build_gate_feature
from robot_perception.models import LocalRobotDetector
from robot_perception.tracker import RobotCandidateTracker
from temporal_interaction_gate import (
    TemporalInteractionGate,
    actor_comparison_features,
)
from train_temporal_interaction_gate import load_actor, sha256_file


BASE = ROOT / "experiments/03_保留专门化/02_论文主线"


def parse_args():
    parser = argparse.ArgumentParser(description="Audit G25 CPU deployment cost.")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "25_最终消融与Sealed评测/local_data/deployment_cost.json",
    )
    return parser.parse_args()


def parameter_count(model):
    return int(sum(item.numel() for item in model.parameters()))


def timed(function, warmup, iterations):
    for _ in range(warmup):
        function()
    values = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(values)
    return {
        "iterations": iterations,
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[int(0.95 * (iterations - 1))],
        "p99_ms": ordered[int(0.99 * (iterations - 1))],
    }


def actor_macs(input_dim=24, hidden_1=800, hidden_2=600, output_dim=2):
    return input_dim * hidden_1 + hidden_1 * hidden_2 + hidden_2 * output_dim


def detector_macs():
    conv1 = 16 * 16 * 64 * (3 * 3 * 3)
    conv2 = 32 * 8 * 32 * (16 * 3 * 3)
    conv3 = 64 * 4 * 16 * (32 * 3 * 3)
    shared = (64 * 2 * 4) * 128
    heads = 128 * 3
    return conv1 + conv2 + conv3 + shared + heads


def router_macs(input_dim=82, hidden_dim=64, sequence_length=8):
    gru_per_step = 3 * (input_dim * hidden_dim + hidden_dim * hidden_dim)
    return sequence_length * gru_per_step + hidden_dim * 32 + 32


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def representative_frame():
    shard_dir = (
        BASE
        / "11_可部署在线Gate研究/G11_B_student_rollout_v1"
        / "local_data/student_shards/train"
    )
    path = sorted(shard_dir.glob("*.npz"))[0]
    with np.load(path, allow_pickle=True) as shard:
        counts = np.bincount(
            shard["frame_record_indices"],
            minlength=len(shard["frame_actor_states"]),
        )
        frame = int(np.argmax(counts))
        rows = np.flatnonzero(shard["frame_record_indices"] == frame)[:12]
        return {
            "shard": str(path),
            "patches": shard["patches"][rows].astype(np.float32),
            "centers": shard["candidate_centers"][rows].astype(np.float32),
            "pose": shard["frame_ego_poses"][frame].astype(np.float32),
            "state": shard["frame_actor_states"][frame].astype(np.float32),
        }


def main():
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")

    standard_path = ROOT / "TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
    interaction_path = ROOT / "TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
    detector_path = BASE / "results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt"
    router_path = BASE / "11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt"
    standard = load_actor(standard_path, "cpu").eval()
    interaction = load_actor(interaction_path, "cpu").eval()
    detector_payload = torch.load(detector_path, map_location=device, weights_only=False)
    detector = LocalRobotDetector(**detector_payload.get("model_config", {})).eval()
    detector.load_state_dict(detector_payload["model_state_dict"])
    router_payload = torch.load(router_path, map_location=device, weights_only=False)
    router = TemporalInteractionGate(**router_payload["model_config"]).eval()
    router.load_state_dict(router_payload["model_state_dict"])

    frame = representative_frame()
    state = torch.from_numpy(frame["state"]).reshape(1, -1)
    patches = torch.from_numpy(frame["patches"])
    if len(patches) == 0:
        raise ValueError("representative frame has no candidates")
    router_input = torch.zeros(1, 8, 82)
    tracker = RobotCandidateTracker()
    fixed_probabilities = torch.sigmoid(detector(patches)[0]).detach().numpy()
    timestamp = [0.0]

    @torch.no_grad()
    def actor_once():
        return standard(state)

    @torch.no_grad()
    def actors_sequential():
        return standard(state), interaction(state)

    @torch.no_grad()
    def detector_once():
        return detector(patches)

    def tracker_feature_once():
        timestamp[0] += 0.2
        tracked = tracker.update(
            frame["centers"], fixed_probabilities, frame["pose"], timestamp[0]
        )
        return build_gate_feature(frame["state"], tracked, max_tracks=4)

    @torch.no_grad()
    def router_once():
        return router(router_input)

    route_tracker = RobotCandidateTracker()
    route_timestamp = [0.0]

    @torch.no_grad()
    def route_frame_once():
        route_timestamp[0] += 0.2
        probabilities = torch.sigmoid(detector(patches)[0]).numpy()
        tracked = route_tracker.update(
            frame["centers"], probabilities, frame["pose"], route_timestamp[0]
        )
        standard_action = standard(state).numpy()
        interaction_action = interaction(state).numpy()
        feature = build_gate_feature(frame["state"], tracked, max_tracks=4)
        actions = actor_comparison_features(standard_action, interaction_action)[0]
        feature = np.concatenate((feature, actions)).astype(np.float32)
        normalized = (
            feature - np.asarray(router_payload["feature_mean"], dtype=np.float32)
        ) / np.asarray(router_payload["feature_std"], dtype=np.float32)
        window = torch.from_numpy(np.repeat(normalized[None, :], 8, axis=0)[None])
        return router(window)

    measurements = {
        "single_actor": timed(actor_once, args.warmup, args.iterations),
        "two_actors_sequential": timed(
            actors_sequential, args.warmup, args.iterations
        ),
        "g0_detector_representative_candidates": timed(
            detector_once, args.warmup, args.iterations
        ),
        "g1_tracker_and_feature": timed(
            tracker_feature_once, args.warmup, args.iterations
        ),
        "router_8_frame": timed(router_once, args.warmup, args.iterations),
        "complete_routing_frame": timed(
            route_frame_once, args.warmup, args.iterations
        ),
    }
    result = {
        "protocol": {
            "experiment_id": "G25-deployment-cost",
            "sealed_test_read": False,
            "device": "cpu",
            "batch_size": 1,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "torch_threads": 1,
            "cpu": cpu_model(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "gate_stride": 2,
            "two_actor_parallel": None,
            "two_actor_parallel_reason": "no parallel Actor execution path is implemented",
            "peak_gpu_memory_mib": 0,
            "representative_candidate_count": int(len(patches)),
            "representative_shard": frame["shard"],
            "artifact_sha256": {
                "standard_actor": sha256_file(standard_path),
                "interaction_actor": sha256_file(interaction_path),
                "detector": sha256_file(detector_path),
                "router": sha256_file(router_path),
            },
        },
        "parameters": {
            "single_actor": parameter_count(standard),
            "two_actors": parameter_count(standard) + parameter_count(interaction),
            "g0_detector": parameter_count(detector),
            "router": parameter_count(router),
        },
        "theoretical_macs": {
            "single_actor": actor_macs(),
            "two_actors": 2 * actor_macs(),
            "g0_detector_per_candidate": detector_macs(),
            "g0_detector_representative_frame": detector_macs() * len(patches),
            "router_8_frame": router_macs(),
        },
        "latency": measurements,
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
