#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_g11_a1_gate_views import write_gzip_json
from scenario_manifests import AGENT_NAMES, load_manifest_dataset, validate_manifest_scenarios


DEFAULT_ROOT = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "views/g12_r3_mixed_v1/train.json.gz"
STRONG_BAND_SCHEDULE = ("deep", "deep", "close", "close", "margin")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the deterministic four-slot G12-R3 training schedule."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def load_source(path):
    payload = load_manifest_dataset(path)
    scenarios = validate_manifest_scenarios(payload["scenarios"], AGENT_NAMES)
    return payload, scenarios


def source_record(path, payload, scenarios, name):
    return {
        "name": name,
        "path": relative_path(path),
        "dataset_id": payload.get("dataset_id"),
        "split": payload.get("split"),
        "sha256": sha256_file(path),
        "scenarios": len(scenarios),
    }


def balanced_strong_sequence(scenarios, count):
    by_band = {band: [] for band in set(STRONG_BAND_SCHEDULE)}
    for scenario in scenarios:
        band = scenario.get("view", {}).get("interaction_band")
        if band not in by_band:
            raise ValueError("strong scenario is missing a registered interaction band")
        by_band[band].append(scenario)
    if any(not items for items in by_band.values()):
        raise ValueError("every strong interaction band must be non-empty")

    indices = {band: 0 for band in by_band}
    selected = []
    for slot in range(count):
        band = STRONG_BAND_SCHEDULE[slot % len(STRONG_BAND_SCHEDULE)]
        items = by_band[band]
        selected.append(items[indices[band] % len(items)])
        indices[band] += 1
    return selected


def annotate(source, stream, schedule_index):
    scenario = copy.deepcopy(source)
    source_id = str(source["scenario_id"])
    alias = "g12-r3-%05d-%s" % (schedule_index, source_id)
    scenario["scenario_id"] = alias
    scenario["name"] = alias
    scenario["navigation_split"] = source.get("split", "train")
    scenario["split"] = "train"
    scenario["view"] = {
        **scenario.get("view", {}),
        "g12_r3_protocol": "g12-r3-mixed-v1",
        "g12_r3_stream": stream,
        "g12_r3_source_scenario_id": source_id,
        "g12_r3_schedule_index": schedule_index,
    }
    return scenario


def build_view(args):
    paths = {
        "standard": args.dataset_root / "standard/train.json.gz",
        "dense": args.dataset_root / "dense/train.json.gz",
        "strong": (
            args.dataset_root
            / "views/strong_interaction_curriculum_v1/full_train.json.gz"
        ),
    }
    payloads = {}
    pools = {}
    records = []
    for name, path in paths.items():
        payload, scenarios = load_source(path)
        payloads[name] = payload
        pools[name] = scenarios
        records.append(source_record(path, payload, scenarios, name))

    if len(pools["standard"]) != 3000 or len(pools["dense"]) != 6000:
        raise ValueError("R3 expects the frozen 3000/6000 broad train pools")
    broad_ids = {
        item["scenario_id"] for item in pools["standard"] + pools["dense"]
    }
    strong_ids = {item["scenario_id"] for item in pools["strong"]}
    if not strong_ids <= broad_ids:
        raise ValueError("strong interaction replay must be derived from broad train")

    broad_slots = len(pools["dense"])
    strong = balanced_strong_sequence(pools["strong"], broad_slots * 2)
    schedule = []
    for index in range(broad_slots):
        slot_sources = (
            ("standard", pools["standard"][index % len(pools["standard"])]),
            ("strong", strong[index * 2]),
            ("dense", pools["dense"][index]),
            ("strong", strong[index * 2 + 1]),
        )
        for stream, source in slot_sources:
            schedule.append(annotate(source, stream, len(schedule)))

    stream_counts = Counter(item["view"]["g12_r3_stream"] for item in schedule)
    band_counts = Counter(
        item["view"].get("interaction_band")
        for item in schedule
        if item["view"]["g12_r3_stream"] == "strong"
    )
    if stream_counts != {"standard": 6000, "dense": 6000, "strong": 12000}:
        raise ValueError("R3 four-slot schedule accounting mismatch")
    if len({item["scenario_id"] for item in schedule}) != len(schedule):
        raise ValueError("R3 schedule aliases must be unique")

    output = {
        "dataset_version": 1,
        "dataset_id": "g12-r3-mixed-v1-train",
        "split": "train",
        "view_config": {
            "protocol": "g12-r3-mixed-v1",
            "schedule": ["standard", "strong", "dense", "strong"],
            "sampling": "cycle",
            "broad_stream_share": 0.5,
            "broad_source_shares": {"standard": 0.5, "dense": 0.5},
            "strong_stream_share": 0.5,
            "strong_band_schedule": list(STRONG_BAND_SCHEDULE),
            "stream_counts": dict(stream_counts),
            "strong_band_counts": dict(band_counts),
            "source_scenario_ids_preserved_in_view": True,
            "sealed_test_read": False,
        },
        "source_manifests": records,
        "scenarios": schedule,
    }
    write_gzip_json(args.output, output)
    return {
        "output": relative_path(args.output),
        "sha256": sha256_file(args.output),
        "scenarios": len(schedule),
        "stream_counts": dict(stream_counts),
        "strong_band_counts": dict(band_counts),
    }


def main():
    print(json.dumps(build_view(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
