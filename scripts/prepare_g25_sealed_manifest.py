#!/usr/bin/env python3
import gzip
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G25 = BASE / "25_最终消融与Sealed评测/local_data"
DRY_RUN = G25 / "dry_run/dry_run_completion.json"
SOURCE = BASE / "datasets/fixed_v1/dense/test.json.gz"
OUTPUT = G25 / "sealed_manifest/dense_test_first256.json.gz"


def main():
    if not DRY_RUN.is_file():
        raise SystemExit("sealed manifest remains locked until the dry-run completes")
    dry_run = json.loads(DRY_RUN.read_text(encoding="utf-8"))
    if dry_run.get("status") != "complete" or dry_run.get("sealed_test_read"):
        raise SystemExit("invalid dry-run completion record")
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = payload.get("scenarios", [])
    if len(scenarios) < 256:
        raise ValueError("dense sealed source has fewer than 256 scenes")
    payload["scenarios"] = scenarios[:256]
    payload["g25_sealed_selection"] = {
        "rule": "original frozen order, first 256 scenes",
        "policy_independent": True,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as file_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=file_handle, mtime=0) as zipped:
            zipped.write(raw)
    record = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "scenes": 256,
        "selection": "first 256 in original frozen order",
        "sealed_test_read": True,
    }
    record_path = OUTPUT.parent / "manifest_record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
