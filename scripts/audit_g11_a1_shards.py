#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from robot_perception.dataset import load_shard


DEFAULT_ROUTE = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
    / "G11_A1_当前协议时序pilot"
)
DEFAULT_VIEW = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
    / "g11_a1_gate_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit complete G11-A1 shards.")
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument(
        "--shard-root", type=Path, default=DEFAULT_ROUTE / "local_data/shards"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROUTE / "local_data/shard_audit.json",
    )
    return parser.parse_args()


def load_manifest(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {item["scenario_id"]: item for item in payload["scenarios"]}


def update_digest(digest, path):
    digest.update(path.name.encode("utf-8"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def assert_finite(name, values, path):
    if not np.all(np.isfinite(values)):
        raise ValueError("%s has non-finite %s" % (path, name))


def audit_shard(path, scenario, split):
    shard = load_shard(path)
    scenario_id = str(shard["scenario_id"])
    expected_pool = scenario["view"]["gate_pool"]
    expected_band = scenario["view"]["interaction_band"]
    if scenario_id != scenario["scenario_id"]:
        raise ValueError("shard scenario ID mismatch: %s" % path)
    if str(shard["split"]) != split:
        raise ValueError("shard split mismatch: %s" % path)
    if str(shard["scenario_pool"]) != expected_pool:
        raise ValueError("shard pool mismatch: %s" % path)
    if str(shard["interaction_band"]) != expected_band:
        raise ValueError("shard interaction band mismatch: %s" % path)

    frame_count = len(shard["frame_actor_states"])
    frame_keys = (
        "frame_ego_indices",
        "frame_indices_unique",
        "frame_ego_poses",
        "frame_timestamps",
        "frame_nearest_robot_distances",
        "frame_oracle_interaction_labels",
        "frame_nearest_front_robot_distances",
        "frame_front_interaction_labels",
    )
    if frame_count < 1 or shard["frame_actor_states"].shape != (frame_count, 24):
        raise ValueError("invalid Actor frame matrix: %s" % path)
    for key in frame_keys:
        if len(shard[key]) != frame_count:
            raise ValueError("frame length mismatch for %s: %s" % (key, path))
    assert_finite("frame_actor_states", shard["frame_actor_states"], path)
    assert_finite("frame_ego_poses", shard["frame_ego_poses"], path)
    assert_finite("frame_timestamps", shard["frame_timestamps"], path)

    labels = shard["frame_oracle_interaction_labels"]
    front_labels = shard["frame_front_interaction_labels"]
    if not set(np.unique(labels).tolist()) <= {0, 1}:
        raise ValueError("invalid Oracle labels: %s" % path)
    if not set(np.unique(front_labels).tolist()) <= {0, 1}:
        raise ValueError("invalid front labels: %s" % path)

    candidate_count = len(shard["patches"])
    candidate_keys = (
        "labels",
        "candidate_centers",
        "candidate_ranges",
        "frame_record_indices",
    )
    if candidate_count < 1 or shard["patches"].shape[1:] != (3, 16, 64):
        raise ValueError("invalid candidate patches: %s" % path)
    for key in candidate_keys:
        if len(shard[key]) != candidate_count:
            raise ValueError("candidate length mismatch for %s: %s" % (key, path))
    assert_finite("patches", shard["patches"], path)
    assert_finite("candidate_centers", shard["candidate_centers"], path)
    record_indices = shard["frame_record_indices"].astype(np.int64)
    if np.min(record_indices) < 0 or np.max(record_indices) >= frame_count:
        raise ValueError("candidate frame index out of range: %s" % path)
    frames_with_candidates = len(np.unique(record_indices))

    for ego_index in np.unique(shard["frame_ego_indices"]):
        rows = np.flatnonzero(shard["frame_ego_indices"] == ego_index)
        frame_indices = shard["frame_indices_unique"][rows]
        timestamps = shard["frame_timestamps"][rows]
        if len(rows) > 1 and (
            np.any(np.diff(frame_indices) <= 0) or np.any(np.diff(timestamps) < 0.0)
        ):
            raise ValueError("non-monotonic ego sequence: %s" % path)

    return {
        "frames": frame_count,
        "candidates": candidate_count,
        "frames_without_candidates": frame_count - frames_with_candidates,
        "oracle_positives": int(np.sum(labels)),
        "front_positives": int(np.sum(front_labels)),
        "visible_robots": int(shard["visible_robot_count"]),
        "missed_visible_robots": int(shard["missed_visible_robot_count"]),
        "pool": expected_pool,
        "band": expected_band,
    }


def audit_split(manifest_path, shard_dir, split):
    scenarios = load_manifest(manifest_path)
    paths = sorted(Path(shard_dir).glob("*.npz"))
    files = {path.stem: path for path in paths}
    missing = sorted(set(scenarios) - set(files))
    extra = sorted(set(files) - set(scenarios))
    if missing or extra or len(paths) != len(files):
        raise ValueError(
            "%s shard coverage mismatch: missing=%s extra=%s files=%d unique=%d"
            % (split, missing[:5], extra[:5], len(paths), len(files))
        )

    totals = Counter()
    strata = defaultdict(Counter)
    digest = hashlib.sha256()
    for scenario_id in sorted(scenarios):
        path = files[scenario_id]
        metrics = audit_shard(path, scenarios[scenario_id], split)
        update_digest(digest, path)
        totals.update({key: value for key, value in metrics.items() if isinstance(value, int)})
        stratum = "%s_%s" % (metrics["pool"], metrics["band"])
        strata[stratum].update(
            {key: value for key, value in metrics.items() if isinstance(value, int)}
        )
        strata[stratum]["shards"] += 1
    return {
        "manifest": str(manifest_path),
        "shard_dir": str(shard_dir),
        "shards": len(paths),
        "dataset_sha256": digest.hexdigest(),
        "totals": dict(totals),
        "strata": {key: dict(value) for key, value in sorted(strata.items())},
    }


def main():
    args = parse_args()
    result = {
        split: audit_split(
            args.view_dir / (split + ".json.gz"),
            args.shard_root / split,
            split,
        )
        for split in ("train", "validation")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
