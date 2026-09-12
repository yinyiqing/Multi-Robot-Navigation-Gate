#!/usr/bin/env python3
"""Freeze and audit the preregistered G34 independent Dense test slice."""

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G34 = BASE / "34_双头RewardAwareGate"
SOURCE = BASE / "datasets/fixed_v1/dense/test.json.gz"
TRAIN = BASE / "datasets/fixed_v1/dense/train.json.gz"
VALIDATION = BASE / "datasets/fixed_v1/dense/validation.json.gz"
G25 = BASE / "25_最终消融与Sealed评测/local_data/sealed_manifest/dense_test_first256.json.gz"
G26 = BASE / "26_数量泛化与外部切换基线/local_data/e1/manifests/dense_test_256_384.json.gz"
G27 = BASE / "27_反事实Reward增强Gate监督/local_data/test/manifests/dense_test_384_640.json.gz"
PROTOCOL = G34 / "local_data/independent_test/protocol.json"
CONFIRMATION = G34 / "local_data/confirmation64/summary.json"
CHECKPOINT = G34 / "local_data/training/seed20260912/best_runtime.pt"
OUTPUT = G34 / "local_data/independent_test/manifest/dense_test_640_896.json.gz"
RECORD = G34 / "local_data/independent_test/manifest/record.json"
START = 640
STOP = 896


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_manifest(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def geometry_signature(item):
    agents = item.get("agents", {})
    geometry = {
        "map_id": item.get("map_id"),
        "num_agents": item.get("num_agents"),
        "start_half_width_m": item.get("start_half_width_m"),
        "boxes": item.get("boxes"),
        "agents": {
            name: {
                "start": values.get("start"),
                "goal": values.get("goal"),
                "heading": values.get("heading"),
            }
            for name, values in sorted(agents.items())
        },
    }
    return canonical_sha256(geometry)


def main():
    required = (SOURCE, TRAIN, VALIDATION, G25, G26, G27, PROTOCOL, CONFIRMATION, CHECKPOINT)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists() or RECORD.exists():
        raise FileExistsError("refusing to overwrite the G34 independent manifest")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    confirmation = json.loads(CONFIRMATION.read_text(encoding="utf-8"))
    if protocol.get("decision") != "freeze_g34_and_run_independent_evaluation":
        raise ValueError("G34 independent evaluation is not authorized")
    if protocol["evaluation"].get("source_slice_zero_based") != [START, STOP]:
        raise ValueError("G34 protocol has a different test slice")
    if protocol["method"].get("checkpoint_sha256") != sha256_file(CHECKPOINT):
        raise ValueError("G34 protocol checkpoint hash mismatch")
    if confirmation.get("decision") != "pass_for_independent_evaluation_design":
        raise ValueError("G34 Dense64 did not authorize independent evaluation")
    if protocol["development_evidence"].get("dense64_summary_sha256") != sha256_file(CONFIRMATION):
        raise ValueError("G34 protocol does not match the confirmation summary")
    status = protocol.get("test_data_status_at_registration", {})
    if status.get("policy_outcomes_for_slice_read") is not False or status.get("g34_rollouts_started") is not False:
        raise ValueError("G34 test was not registered before outcomes")

    source_hash = sha256_file(SOURCE)
    if protocol["evaluation"].get("source_manifest_sha256") != source_hash:
        raise ValueError("Dense test source hash mismatch")
    source = load_manifest(SOURCE)
    selected = source.get("scenarios", [])[START:STOP]
    if len(selected) != STOP - START:
        raise ValueError("G34 independent selection must have 256 scenes")
    if any(int(item.get("num_agents", -1)) != 5 for item in selected):
        raise ValueError("G34 independent selection contains non-five-robot scenes")
    selected_ids = [str(item["scenario_id"]) for item in selected]
    selected_geometry = {geometry_signature(item) for item in selected}
    if len(set(selected_ids)) != len(selected) or len(selected_geometry) != len(selected):
        raise ValueError("G34 independent selection contains duplicates")

    overlap_audit = {}
    for name, path in (
        ("dense_train", TRAIN),
        ("dense_validation", VALIDATION),
        ("g25_test_0_256", G25),
        ("g26_test_256_384", G26),
        ("g27_test_384_640", G27),
    ):
        other = load_manifest(path).get("scenarios", [])
        id_overlap = set(selected_ids) & {str(item["scenario_id"]) for item in other}
        geometry_overlap = selected_geometry & {geometry_signature(item) for item in other}
        overlap_audit[name] = {
            "scene_id_overlap": len(id_overlap),
            "complete_geometry_overlap": len(geometry_overlap),
        }
        if id_overlap or geometry_overlap:
            raise ValueError("G34 independent test overlaps %s" % name)

    output = dict(source)
    output["dataset_id"] = "g34-independent-dense-test-640-896-v1"
    output["split"] = "independent-test"
    output["candidate_count"] = len(selected)
    output["rejection_counts"] = {}
    output["scenarios"] = selected
    output["g34_selection"] = {
        "rule": "original frozen Dense test order, Python slice [640:896]",
        "source_sha256": source_hash,
        "protocol_sha256": sha256_file(PROTOCOL),
        "policy_independent": True,
        "overlap_audit": overlap_audit,
    }
    raw = (json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as output_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output_handle, mtime=0) as zipped:
            zipped.write(raw)

    record = {
        "format_version": 1,
        "experiment_id": "G34-P3-independent-dense-test",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
        "source_slice_zero_based": [START, STOP],
        "scenes": len(selected),
        "first_scenario_id": selected_ids[0],
        "last_scenario_id": selected_ids[-1],
        "protocol_sha256": sha256_file(PROTOCOL),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "confirmation_summary_sha256": sha256_file(CONFIRMATION),
        "overlap_audit": overlap_audit,
    }
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
