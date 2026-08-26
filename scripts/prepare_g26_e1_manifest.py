#!/usr/bin/env python3
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
SOURCE = BASE / "datasets/fixed_v1/dense/test.json.gz"
E1 = BASE / "26_数量泛化与外部切换基线/local_data/e1"
OUTPUT = E1 / "manifests/dense_test_256_384.json.gz"
RECORD = E1 / "manifests/manifest_record.json"
START = 256
STOP = 384


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not SOURCE.is_file():
        raise SystemExit("missing frozen dense test source: %s" % SOURCE)
    if OUTPUT.exists() or RECORD.exists():
        raise SystemExit("E1 manifest already exists; refusing to overwrite it")

    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = payload.get("scenarios", [])
    if len(scenarios) < STOP:
        raise ValueError("dense test source has fewer than %d scenes" % STOP)
    selected = scenarios[START:STOP]
    if len(selected) != STOP - START:
        raise ValueError("E1 selection does not contain exactly 128 scenes")
    if any(int(item.get("num_agents", -1)) != 5 for item in selected):
        raise ValueError("E1 Dense test slice must contain five-robot scenes")
    scenario_ids = [str(item["scenario_id"]) for item in selected]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("E1 manifest contains duplicate scenario IDs")

    payload["scenarios"] = selected
    payload["g26_e1_selection"] = {
        "rule": "original frozen dense/test order, Python slice [256:384]",
        "start_index_in_source": START,
        "stop_index_in_source": STOP,
        "scene_count": len(selected),
        "policy_independent": True,
        "source_sha256": sha256_file(SOURCE),
        "sealed_test_read": True,
    }
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as output_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output_handle, mtime=0
        ) as zipped:
            zipped.write(raw)

    record = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
        "start_index_in_source": START,
        "stop_index_in_source": STOP,
        "scenes": len(selected),
        "first_scenario_id": scenario_ids[0],
        "last_scenario_id": scenario_ids[-1],
        "selection": "original frozen order, source slice [256:384]",
        "policy_independent": True,
        "sealed_test_read": True,
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
