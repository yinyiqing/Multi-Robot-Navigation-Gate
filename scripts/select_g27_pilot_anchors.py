#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_models import Actor
from g27_counterfactual import (
    action_stratum,
    canonical_sha256,
    deployed_action,
    distance_stratum,
    select_stratified_anchors,
    select_stratified_anchors_matching,
    selection_hash,
)
from robot_perception.dataset import load_shard


BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
DEFAULT_MANIFEST = BASE / "datasets/fixed_v1/views/g11_a1_gate_v1/train.json.gz"
DEFAULT_SHARDS = (
    BASE
    / "11_可部署在线Gate研究/G11_A1_当前协议时序pilot/local_data/shards/train"
)
DEFAULT_OUTPUT = BASE / "27_反事实Reward增强Gate监督/local_data/protocol/p0_anchors.json"
DEFAULT_NAVIGATION = (
    ROOT
    / "TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
)
DEFAULT_INTERACTION = (
    ROOT
    / "TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze the 32 G27 P0 anchors.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--navigation-actor", type=Path, default=DEFAULT_NAVIGATION)
    parser.add_argument("--interaction-actor", type=Path, default=DEFAULT_INTERACTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--minimum-step", type=int, default=4)
    parser.add_argument("--maximum-step", type=int, default=4)
    parser.add_argument("--action-threshold", type=float, default=0.25)
    parser.add_argument("--per-cell", type=int, default=4)
    parser.add_argument("--matching", action="store_true")
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        action="append",
        default=[],
        help="selection JSON files whose scenario IDs must be excluded",
    )
    parser.add_argument("--protocol", default="G27-P0-fresh-process-one-step-v1")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_actor(path):
    model = Actor(24, 2)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def load_manifest(path):
    if Path(path).suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    scenarios = payload.get("scenarios", [])
    return payload, {str(item["scenario_id"]): index for index, item in enumerate(scenarios)}


def collect_candidates(args, scenario_indices, navigation, interaction):
    candidates = []
    shard_paths = sorted(args.shard_dir.glob("*.npz"))
    if set(path.stem for path in shard_paths) != set(scenario_indices):
        raise ValueError("training shard coverage does not match the G11-A1 manifest")
    with torch.no_grad():
        for path in shard_paths:
            shard = load_shard(path)
            scenario_id = str(shard["scenario_id"])
            states = torch.from_numpy(shard["frame_actor_states"].astype(np.float32))
            navigation_raw = navigation(states).cpu().numpy()
            interaction_raw = interaction(states).cpu().numpy()
            for row in range(len(states)):
                step = int(shard["frame_indices_unique"][row])
                if not args.minimum_step <= step <= args.maximum_step:
                    continue
                ego_index = int(shard["frame_ego_indices"][row])
                distance = float(shard["frame_nearest_robot_distances"][row])
                if not np.isfinite(distance):
                    continue
                action_n = deployed_action(navigation_raw[row])
                action_i = deployed_action(interaction_raw[row])
                disagreement = float(np.linalg.norm(action_i - action_n))
                item = {
                    "scenario_id": scenario_id,
                    "scenario_index": int(scenario_indices[scenario_id]),
                    "anchor_step": step,
                    "ego_index": ego_index,
                    "ego_name": "r%d" % (ego_index + 1),
                    "prior_nearest_robot_distance": distance,
                    "prior_interaction_label": int(distance <= 2.0),
                    "prior_action_disagreement_l2": disagreement,
                    "distance_stratum": distance_stratum(distance),
                    "action_stratum": action_stratum(
                        disagreement, threshold=args.action_threshold
                    ),
                    "selection_hash": selection_hash(
                        args.seed, scenario_id, step, ego_index
                    ),
                    "source_shard": str(path.relative_to(ROOT)),
                }
                item["anchor_id"] = item["selection_hash"][:16]
                candidates.append(item)
    return candidates


def main():
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.shard_dir = args.shard_dir.resolve()
    args.navigation_actor = args.navigation_actor.resolve()
    args.interaction_actor = args.interaction_actor.resolve()
    args.output = args.output.resolve()
    if args.minimum_step < 0 or args.maximum_step < args.minimum_step:
        raise ValueError("invalid anchor-step interval")
    if args.action_threshold <= 0.0:
        raise ValueError("action threshold must be positive")
    for path in (args.manifest, args.navigation_actor, args.interaction_actor):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.shard_dir.is_dir():
        raise FileNotFoundError(args.shard_dir)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen anchor file: %s" % args.output)

    torch.set_num_threads(1)
    manifest, scenario_indices = load_manifest(args.manifest)
    navigation = load_actor(args.navigation_actor)
    interaction = load_actor(args.interaction_actor)
    candidates = collect_candidates(
        args, scenario_indices, navigation, interaction
    )
    excluded_scenes = set()
    for exclusion_path in args.exclude_selection:
        exclusion = load_json(exclusion_path)
        _validate = exclusion.get("anchors", [])
        excluded_scenes.update(str(item["scenario_id"]) for item in _validate)
    if excluded_scenes:
        candidates = [
            item for item in candidates if str(item["scenario_id"]) not in excluded_scenes
        ]
    selector = (
        select_stratified_anchors_matching if args.matching else select_stratified_anchors
    )
    selected = selector(candidates, per_cell=args.per_cell)
    cell_counts = Counter(
        "%s/%s" % (item["distance_stratum"], item["action_stratum"])
        for item in selected
    )
    payload = {
        "format_version": 1,
        "protocol": args.protocol,
        "selection": {
            "seed": args.seed,
            "minimum_step": args.minimum_step,
            "maximum_step": args.maximum_step,
            "distance_boundaries_m": [1.2, 2.0, 3.0],
            "action_disagreement_l2_threshold": args.action_threshold,
            "per_cell": args.per_cell,
            "candidate_count": len(candidates),
            "excluded_scene_count": len(excluded_scenes),
            "selected_count": len(selected),
            "cell_counts": dict(sorted(cell_counts.items())),
            "outcome_or_reward_used": False,
            "unique_scene_matching": bool(args.matching),
        },
        "inputs": {
            "manifest": str(args.manifest.relative_to(ROOT)),
            "manifest_dataset_id": manifest.get("dataset_id"),
            "manifest_sha256": sha256(args.manifest),
            "shard_dir": str(args.shard_dir.relative_to(ROOT)),
            "navigation_actor": str(args.navigation_actor.relative_to(ROOT)),
            "navigation_actor_sha256": sha256(args.navigation_actor),
            "interaction_actor": str(args.interaction_actor.relative_to(ROOT)),
            "interaction_actor_sha256": sha256(args.interaction_actor),
        },
        "anchors": selected,
    }
    payload["selection_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(payload["selection"], ensure_ascii=False, indent=2))
    print("Selection SHA-256:", payload["selection_sha256"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
