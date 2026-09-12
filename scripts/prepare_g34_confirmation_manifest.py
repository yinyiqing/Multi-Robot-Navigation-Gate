#!/usr/bin/env python3
"""Freeze the disjoint 64-scene Dense development confirmation manifest."""

import gzip
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
SOURCE = BASE / "datasets/fixed_v1/dense/validation.json.gz"
OUTPUT = BASE / "34_双头RewardAwareGate/local_data/protocol/dense_confirmation64.json.gz"
SOURCE_SHA256 = "2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7"
START = 32
COUNT = 64


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_gzip(payload):
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


def main():
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("Dense validation source manifest hash mismatch")
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    selected = source["scenarios"][START : START + COUNT]
    if len(selected) != COUNT or len({row["scenario_id"] for row in selected}) != COUNT:
        raise ValueError("invalid G34 confirmation slice")
    pilot_ids = {row["scenario_id"] for row in source["scenarios"][:START]}
    if pilot_ids & {row["scenario_id"] for row in selected}:
        raise ValueError("G34 pilot and confirmation scenes overlap")

    output = dict(source)
    output["dataset_id"] = "g34-dense-development-confirmation64-v1"
    output["split"] = "development-confirmation"
    output["candidate_count"] = COUNT
    output["rejection_counts"] = {}
    output["scenarios"] = selected
    output["source_selection"] = {
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "zero_based_slice": [START, START + COUNT],
        "selected_count": COUNT,
        "overlap_with_dense32_pilot": 0,
    }
    encoded = encode_gzip(output)
    if OUTPUT.exists() and OUTPUT.read_bytes() != encoded:
        raise FileExistsError("refusing to replace a different G34 confirmation manifest")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(encoded)
    print(json.dumps({
        "path": str(OUTPUT.relative_to(ROOT)),
        "sha256": sha256(OUTPUT),
        "scenes": COUNT,
        "source_slice": [START, START + COUNT],
        "pilot_overlap": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
