#!/usr/bin/env python3
import copy
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
    / "g11_a1_gate_v1/validation.json.gz"
)
OUTPUT_DIR = SOURCE.parent.parent / "g11_c_pilot_v1"
OUTPUT = OUTPUT_DIR / "validation.json.gz"
QUOTAS = {
    ("standard", "zero"): 13,
    ("standard", "edge1"): 12,
    ("dense", "zero"): 12,
    ("dense", "edge1"): 13,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_scenarios(scenarios, quotas=QUOTAS):
    selected = []
    counts = Counter()
    for source in scenarios:
        key = (source["view"]["gate_pool"], source["view"]["gate_topology"])
        if key not in quotas or counts[key] >= quotas[key]:
            continue
        scenario = copy.deepcopy(source)
        scenario["view"]["g11_c_pilot"] = True
        scenario["view"]["g11_c_stratum"] = "%s_%s" % key
        selected.append(scenario)
        counts[key] += 1
    if dict(counts) != quotas or len(selected) != sum(quotas.values()):
        raise ValueError("source validation cannot satisfy G11-C pilot quotas")
    if len({item["scenario_id"] for item in selected}) != len(selected):
        raise ValueError("G11-C pilot contains duplicate scenario IDs")
    return selected


def write_gzip_json(path, payload):
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(encoded)


def main():
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    selected = select_scenarios(source["scenarios"])
    payload = {
        "dataset_version": 1,
        "dataset_id": "g11-c-closed-loop-pilot-v1-validation",
        "split": "validation",
        "view_config": {
            "protocol": "g11-c-pilot-v1",
            "source_manifest": str(SOURCE.relative_to(PROJECT_ROOT)),
            "source_manifest_sha256": sha256_file(SOURCE),
            "selection": "first scenarios in frozen source order satisfying quotas",
            "quotas": {"%s_%s" % key: value for key, value in QUOTAS.items()},
            "episodes": len(selected),
            "repeats": 2,
            "sealed_test_read": False,
        },
        "scenarios": selected,
    }
    write_gzip_json(OUTPUT, payload)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256_file(OUTPUT),
                "scenarios": len(selected),
                "quotas": payload["view_config"]["quotas"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
