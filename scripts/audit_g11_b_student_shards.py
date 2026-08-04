#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from audit_g11_a1_shards import audit_shard, update_digest
from robot_perception.dataset import load_shard


ROUTE = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
    / "G11_B_student_rollout_v1"
)
MANIFEST = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
    / "g11_a1_gate_v1/train.json.gz"
)
RUN_METADATA = ROUTE / "student_run_metadata.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Audit G11-B student shards.")
    parser.add_argument("--profile", choices=("smoke", "train"), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        manifest_items = json.load(handle)["scenarios"]
    expected_metadata = json.loads(RUN_METADATA.read_text(encoding="utf-8"))
    expected_metadata_json = json.dumps(
        expected_metadata, sort_keys=True, separators=(",", ":")
    )
    expected_items = manifest_items[:1] if args.profile == "smoke" else manifest_items
    expected = {item["scenario_id"]: item for item in expected_items}
    shard_dir = (
        ROUTE / "local_data/smoke/student_shards"
        if args.profile == "smoke"
        else ROUTE / "local_data/student_shards/train"
    )
    paths = sorted(shard_dir.glob("*.npz"))
    files = {path.stem: path for path in paths}
    if set(files) != set(expected) or len(files) != len(paths):
        raise ValueError(
            "G11-B %s coverage mismatch: expected=%d files=%d missing=%s extra=%s"
            % (
                args.profile,
                len(expected),
                len(paths),
                sorted(set(expected) - set(files))[:5],
                sorted(set(files) - set(expected))[:5],
            )
        )

    totals = Counter()
    strata = defaultdict(Counter)
    digest = hashlib.sha256()
    for scenario_id in sorted(expected):
        path = files[scenario_id]
        metrics = audit_shard(path, expected[scenario_id], "train")
        shard = load_shard(path)
        if "run_metadata_json" not in shard:
            raise ValueError("G11-B shard lacks run metadata: %s" % path)
        if str(shard["run_metadata_json"]) != expected_metadata_json:
            raise ValueError("G11-B shard run metadata mismatch: %s" % path)
        update_digest(digest, path)
        numeric = {
            key: value for key, value in metrics.items() if isinstance(value, int)
        }
        totals.update(numeric)
        stratum = "%s_%s" % (metrics["pool"], metrics["band"])
        strata[stratum].update(numeric)
        strata[stratum]["shards"] += 1

    result = {
        "profile": args.profile,
        "manifest": str(MANIFEST),
        "shard_dir": str(shard_dir),
        "shards": len(paths),
        "dataset_sha256": digest.hexdigest(),
        "totals": dict(totals),
        "strata": {key: dict(value) for key, value in sorted(strata.items())},
    }
    output = ROUTE / "local_data" / ("%s_audit.json" % args.profile)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
