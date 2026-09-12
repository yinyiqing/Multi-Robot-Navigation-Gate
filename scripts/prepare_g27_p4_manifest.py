#!/usr/bin/env python3
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G27 = BASE / "27_反事实Reward增强Gate监督"
SOURCE = BASE / "datasets/fixed_v1/dense/test.json.gz"
TRAIN = BASE / "datasets/fixed_v1/dense/train.json.gz"
VALIDATION = BASE / "datasets/fixed_v1/dense/validation.json.gz"
G25 = BASE / "25_最终消融与Sealed评测/local_data/sealed_manifest/dense_test_first256.json.gz"
G26 = BASE / "26_数量泛化与外部切换基线/local_data/e1/manifests/dense_test_256_384.json.gz"
P3_SUMMARY = G27 / "local_data/validation/p3_summary.json"
P3_RESULT = G27 / "local_data/validation/results/g27_dense256_b3_s20260810.npy"
AMENDMENT = G27 / "local_data/protocol/p4_continuation_amendment.json"
B3 = G27 / "local_data/training/seed20260910/best.pt"
OUTPUT = G27 / "local_data/test/manifests/dense_test_384_640.json.gz"
RECORD = G27 / "local_data/test/manifests/manifest_record.json"
START = 384
STOP = 640


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def verify_p4_authorization(summary, amendment):
    recorded = summary.get("summary_sha256")
    payload = dict(summary)
    payload.pop("summary_sha256", None)
    if not recorded or canonical_sha256(payload) != recorded:
        raise ValueError("P3 summary hash is invalid")
    criteria = summary.get("admission_criteria", {})
    required_p3_safety = (
        "full_success_not_below_b2",
        "robot_collision_increase_at_most_2pp",
        "episode_timeout_increase_at_most_1pp",
    )
    if not all(criteria.get(name) is True for name in required_p3_safety):
        raise ValueError("P3 did not pass the frozen success/safety checks")
    if criteria.get("interaction_share_down_10pp_or_paired_steps_down_5") is not False:
        raise ValueError("P3 history does not match the registered amendment")
    if summary.get("admission_passed") is not False:
        raise ValueError("P3 historical admission record was unexpectedly changed")

    if amendment.get("experiment_id") != "G27-P4-continuation-amendment":
        raise ValueError("invalid P4 continuation amendment")
    if amendment.get("decision") != "continue_to_independent_test_with_frozen_b3":
        raise ValueError("the amendment does not authorize P4")
    if amendment.get("removed_admission_condition") != (
        "interaction selection share down by 10 percentage points or "
        "paired-success steps down by 5"
    ):
        raise ValueError("the amendment does not remove the historical efficiency veto")
    status = amendment.get("test_data_status_at_registration", {})
    if status != {
        "p4_manifest_generated": False,
        "dense_test_slice_384_640_read": False,
        "p4_rollouts_started": False,
    }:
        raise ValueError("the amendment was not registered before P4 test access")
    if amendment.get("p3_summary_sha256") != sha256_file(P3_SUMMARY):
        raise ValueError("the amendment does not match the frozen P3 summary")
    if amendment.get("p3_result_sha256") != sha256_file(P3_RESULT):
        raise ValueError("the amendment does not match the frozen P3 result")
    b3_hash = sha256_file(B3)
    if summary.get("b3_checkpoint_sha256") != b3_hash:
        raise ValueError("P3 summary does not match the frozen B3 checkpoint")
    if amendment.get("b3_checkpoint_sha256") != b3_hash:
        raise ValueError("the amendment does not match the frozen B3 checkpoint")


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
    required = (
        SOURCE,
        TRAIN,
        VALIDATION,
        G25,
        G26,
        P3_SUMMARY,
        P3_RESULT,
        AMENDMENT,
        B3,
    )
    for path in required:
        if not path.is_file():
            raise SystemExit("missing P4 prerequisite: %s" % path)
    if OUTPUT.exists() or RECORD.exists():
        raise SystemExit("P4 manifest already exists; refusing to overwrite it")
    test_dir = G27 / "local_data/test"
    if any((test_dir / name).exists() for name in ("results", "checkpoints", "p4_completion.json", "p4_statistics.json")):
        raise SystemExit("P4 outputs already exist before manifest preparation")

    p3_summary = load_json(P3_SUMMARY)
    amendment = load_json(AMENDMENT)
    verify_p4_authorization(p3_summary, amendment)
    source_payload = load_manifest(SOURCE)
    scenarios = source_payload.get("scenarios", [])
    if len(scenarios) < STOP:
        raise ValueError("dense test source has fewer than %d scenes" % STOP)
    selected = scenarios[START:STOP]
    if len(selected) != STOP - START:
        raise ValueError("P4 selection must contain exactly 256 scenes")
    if any(int(item.get("num_agents", -1)) != 5 for item in selected):
        raise ValueError("P4 selection contains a non-five-robot scene")

    selected_ids = [str(item["scenario_id"]) for item in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("P4 selection contains duplicate scene IDs")
    selected_geometry = {geometry_signature(item) for item in selected}
    if len(selected_geometry) != len(selected):
        raise ValueError("P4 selection contains duplicate complete geometries")

    overlap_audit = {}
    for name, path in (
        ("dense_train", TRAIN),
        ("dense_validation", VALIDATION),
        ("g25_sealed", G25),
        ("g26_external", G26),
    ):
        other = load_manifest(path).get("scenarios", [])
        other_ids = {str(item["scenario_id"]) for item in other}
        other_geometry = {geometry_signature(item) for item in other}
        id_overlap = sorted(set(selected_ids) & other_ids)
        geometry_overlap = selected_geometry & other_geometry
        overlap_audit[name] = {
            "scene_id_overlap": len(id_overlap),
            "complete_geometry_overlap": len(geometry_overlap),
        }
        if id_overlap or geometry_overlap:
            raise ValueError("P4 overlaps %s" % name)

    output_payload = dict(source_payload)
    output_payload["scenarios"] = selected
    output_payload["g27_p4_selection"] = {
        "rule": "original frozen dense/test order, Python slice [384:640]",
        "start_index_in_source": START,
        "stop_index_in_source": STOP,
        "scene_count": len(selected),
        "policy_independent": True,
        "source_sha256": sha256_file(SOURCE),
        "p3_summary_sha256": sha256_file(P3_SUMMARY),
        "p3_result_sha256": sha256_file(P3_RESULT),
        "continuation_amendment_sha256": sha256_file(AMENDMENT),
        "overlap_audit": overlap_audit,
        "sealed_test_read": True,
    }
    raw = (json.dumps(output_payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as output_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output_handle, mtime=0
        ) as zipped:
            zipped.write(raw)

    record = {
        "format_version": 1,
        "experiment_id": "G27-P4-independent-test",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
        "start_index_in_source": START,
        "stop_index_in_source": STOP,
        "scenes": len(selected),
        "first_scenario_id": selected_ids[0],
        "last_scenario_id": selected_ids[-1],
        "selection": "original frozen order, source slice [384:640]",
        "policy_independent": True,
        "p3_summary_sha256": sha256_file(P3_SUMMARY),
        "p3_result_sha256": sha256_file(P3_RESULT),
        "continuation_amendment": str(AMENDMENT.relative_to(ROOT)),
        "continuation_amendment_sha256": sha256_file(AMENDMENT),
        "b3_checkpoint_sha256": p3_summary["b3_checkpoint_sha256"],
        "overlap_audit": overlap_audit,
        "sealed_test_read": True,
    }
    RECORD.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
