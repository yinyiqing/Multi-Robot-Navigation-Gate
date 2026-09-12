import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from g27_counterfactual import BRANCH_NAMES, canonical_sha256


REWARD_COMPONENTS = (
    "goal",
    "collision",
    "progress",
    "forward",
    "turn",
    "obstacle",
    "stagnation",
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_payload_hash(payload, field):
    if field not in payload:
        raise ValueError("payload is missing %s" % field)
    expected = str(payload[field])
    unhashed = dict(payload)
    del unhashed[field]
    actual = canonical_sha256(unhashed)
    if actual != expected:
        raise ValueError("%s mismatch: expected %s, got %s" % (field, expected, actual))
    return expected


def _stratum(anchor):
    return "%s/%s" % (anchor["distance_stratum"], anchor["action_stratum"])


def _validate_reward(branch, anchor_id, branch_name):
    components = branch.get("reward_components", {})
    missing = sorted(set(REWARD_COMPONENTS + ("total",)) - set(components))
    if missing:
        raise ValueError(
            "%s/%s reward is missing components %s"
            % (anchor_id, branch_name, missing)
        )
    component_total = float(sum(float(components[name]) for name in REWARD_COMPONENTS))
    if not np.isclose(component_total, float(components["total"]), atol=1e-8, rtol=0.0):
        raise ValueError("%s/%s reward components do not sum" % (anchor_id, branch_name))
    if not np.isclose(
        float(branch["reward"]), float(components["total"]), atol=1e-8, rtol=0.0
    ):
        raise ValueError("%s/%s reward differs from component total" % (anchor_id, branch_name))


def _validate_features(reference, anchor_id):
    features = reference.get("gate_features")
    if not isinstance(features, dict):
        raise ValueError("%s has no Gate features" % anchor_id)
    if int(features.get("sampling_stride_environment_steps", -1)) != 2:
        raise ValueError("%s has the wrong Gate sampling stride" % anchor_id)
    if int(features.get("sequence_length", -1)) != 8:
        raise ValueError("%s has the wrong Gate sequence length" % anchor_id)
    history = np.asarray(features.get("combined_feature_history"), dtype=np.float32)
    window = np.asarray(features.get("combined_feature_window"), dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 82 or len(history) < 1:
        raise ValueError("%s has invalid combined feature history" % anchor_id)
    if window.shape != (8, 82):
        raise ValueError("%s Gate window is not 8x82" % anchor_id)
    if not np.all(np.isfinite(history)) or not np.all(np.isfinite(window)):
        raise ValueError("%s Gate features contain non-finite values" % anchor_id)
    retained_length = min(len(history), 8)
    if not np.array_equal(window[-retained_length:], history[-retained_length:]):
        raise ValueError("%s padded Gate window does not end with its history" % anchor_id)
    if retained_length < 8 and np.any(window[:-retained_length] != 0.0):
        raise ValueError("%s Gate window has nonzero left padding" % anchor_id)
    return window, retained_length


def _validate_selection(selection):
    verify_payload_hash(selection, "selection_sha256")
    anchors = selection.get("anchors", [])
    expected_count = int(selection.get("selection", {}).get("selected_count", -1))
    if len(anchors) != expected_count:
        raise ValueError("selection count does not match its metadata")
    anchor_ids = [str(item["anchor_id"]) for item in anchors]
    scene_ids = [str(item["scenario_id"]) for item in anchors]
    if len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("selection contains duplicate anchor IDs")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("selection contains duplicate scenario IDs")
    observed_cells = Counter(_stratum(item) for item in anchors)
    declared_cells = {
        str(name): int(count)
        for name, count in selection.get("selection", {}).get("cell_counts", {}).items()
    }
    if dict(sorted(observed_cells.items())) != dict(sorted(declared_cells.items())):
        raise ValueError("selection stratum counts do not match their metadata")
    return anchors


def audit_split(selection_path, collection_dir):
    selection_path = Path(selection_path)
    collection_dir = Path(collection_dir)
    selection = load_json(selection_path)
    anchors = _validate_selection(selection)
    by_id = {str(item["anchor_id"]): item for item in anchors}

    run_manifest = load_json(collection_dir / "run_manifest.json")
    verify_payload_hash(run_manifest, "run_manifest_sha256")
    if run_manifest.get("selection_sha256") != selection["selection_sha256"]:
        raise ValueError("run manifest uses a different selection")
    if list(run_manifest.get("anchor_ids", [])) != [item["anchor_id"] for item in anchors]:
        raise ValueError("run manifest anchor order differs from the selection")

    collection_summary = load_json(collection_dir / "summary.json")
    verify_payload_hash(collection_summary, "summary_sha256")
    if not collection_summary.get("analysis", {}).get("completed"):
        raise ValueError("collection summary is not complete")

    status_counts = Counter()
    selected_by_cell = Counter(_stratum(item) for item in anchors)
    usable_by_cell = Counter()
    invalid_by_cell = Counter()
    alignment_excluded_by_cell = Counter()
    records = []
    record_hashes = collection_summary.get("anchor_record_sha256", {})

    for anchor_id, anchor in by_id.items():
        combined_path = collection_dir / "anchors" / anchor_id / "combined.json"
        if not combined_path.is_file():
            raise ValueError("missing combined record for %s" % anchor_id)
        combined = load_json(combined_path)
        record_hash = verify_payload_hash(combined, "record_sha256")
        if record_hashes.get(anchor_id) != record_hash:
            raise ValueError("summary record hash differs for %s" % anchor_id)
        expected_candidate = dict(anchor)
        expected_candidate["protocol"] = selection["protocol"]
        if combined.get("candidate") != expected_candidate:
            raise ValueError("saved candidate differs for %s" % anchor_id)
        reference = combined.get("reference", {})
        for key in (
            "anchor_id",
            "scenario_id",
            "scenario_index",
            "anchor_step",
            "ego_index",
            "selection_hash",
        ):
            if reference.get(key) != expected_candidate.get(key):
                raise ValueError("reference %s differs for %s" % (key, anchor_id))

        status = str(combined.get("status"))
        status_counts[status] += 1
        cell = _stratum(anchor)
        if status == "invalid_reference":
            if reference.get("valid", True) or combined.get("branches"):
                raise ValueError("invalid reference record is inconsistent for %s" % anchor_id)
            invalid_by_cell[cell] += 1
            continue
        if status != "complete" or not reference.get("valid", False):
            raise ValueError("unexpected collection status for %s: %s" % (anchor_id, status))

        spec_hash = verify_payload_hash(reference, "spec_sha256")
        distance = float(reference["reference_nearest_robot_distance"])
        interaction_label = int(reference["reference_interaction_label"])
        if interaction_label != int(distance <= 2.0):
            raise ValueError("reference interaction label is wrong for %s" % anchor_id)
        window, history_length = _validate_features(reference, anchor_id)
        branches = combined.get("branches", {})
        if set(branches) != set(BRANCH_NAMES):
            raise ValueError("%s does not contain N1/N2/I1/I2" % anchor_id)
        all_aligned = True
        for branch_name in BRANCH_NAMES:
            branch = branches[branch_name]
            verify_payload_hash(branch, "result_sha256")
            if branch.get("spec_sha256") != spec_hash:
                raise ValueError("%s/%s uses a different reference" % (anchor_id, branch_name))
            expected_actor = "interaction" if branch_name.startswith("I") else "navigation"
            if branch.get("branch") != branch_name or branch.get("actor") != expected_actor:
                raise ValueError("%s/%s branch identity is wrong" % (anchor_id, branch_name))
            branch_aligned = bool(branch.get("alignment", {}).get("passed"))
            all_aligned &= branch_aligned
            if branch_aligned:
                _validate_reward(branch, anchor_id, branch_name)
        if not all_aligned:
            alignment_excluded_by_cell[cell] += 1
            continue

        delta_1 = float(branches["I1"]["reward"]) - float(branches["N1"]["reward"])
        delta_2 = float(branches["I2"]["reward"]) - float(branches["N2"]["reward"])
        records.append(
            {
                "anchor_id": anchor_id,
                "scenario_id": str(anchor["scenario_id"]),
                "stratum": cell,
                "anchor_step": int(anchor["anchor_step"]),
                "ego_index": int(anchor["ego_index"]),
                "gate_window": window,
                "history_length": history_length,
                "interaction_label": interaction_label,
                "delta_r_1": delta_1,
                "delta_r_2": delta_2,
                "mean_delta_r": 0.5 * (delta_1 + delta_2),
            }
        )
        usable_by_cell[cell] += 1

    declared = collection_summary["analysis"]
    expected_counts = {
        "anchors_total": len(anchors),
        "reference_valid": status_counts["complete"],
        "reference_invalid": status_counts["invalid_reference"],
        "usable_anchors": len(records),
    }
    for key, value in expected_counts.items():
        if int(declared.get(key, -1)) != value:
            raise ValueError("collection summary %s does not match audited data" % key)

    mean_deltas = np.asarray([item["mean_delta_r"] for item in records], dtype=np.float64)
    split_summary = {
        "selection": str(selection_path),
        "selection_sha256": selection["selection_sha256"],
        "collection": str(collection_dir),
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "collection_summary_sha256": collection_summary["summary_sha256"],
        "selected": len(anchors),
        "reference_invalid": status_counts["invalid_reference"],
        "alignment_excluded": int(sum(alignment_excluded_by_cell.values())),
        "usable": len(records),
        "coverage": {
            "selected": dict(sorted(selected_by_cell.items())),
            "usable": dict(sorted(usable_by_cell.items())),
            "invalid_reference": dict(sorted(invalid_by_cell.items())),
            "alignment_excluded": dict(sorted(alignment_excluded_by_cell.items())),
        },
        "mean_delta_r": {
            "minimum": float(np.min(mean_deltas)) if len(mean_deltas) else None,
            "median": float(np.median(mean_deltas)) if len(mean_deltas) else None,
            "maximum": float(np.max(mean_deltas)) if len(mean_deltas) else None,
            "positive": int(np.sum(mean_deltas > 0.0)),
            "nonpositive": int(np.sum(mean_deltas <= 0.0)),
        },
    }
    return records, split_summary


def records_to_arrays(records):
    if not records:
        raise ValueError("cannot build an empty G27 dataset")
    return {
        "anchor_ids": np.asarray([item["anchor_id"] for item in records]),
        "scenario_ids": np.asarray([item["scenario_id"] for item in records]),
        "strata": np.asarray([item["stratum"] for item in records]),
        "anchor_steps": np.asarray([item["anchor_step"] for item in records], dtype=np.int32),
        "ego_indices": np.asarray([item["ego_index"] for item in records], dtype=np.int32),
        "gate_windows": np.stack([item["gate_window"] for item in records]).astype(np.float32),
        "history_lengths": np.asarray(
            [item["history_length"] for item in records], dtype=np.int32
        ),
        "interaction_labels": np.asarray(
            [item["interaction_label"] for item in records], dtype=np.float32
        ),
        "delta_r_1": np.asarray([item["delta_r_1"] for item in records], dtype=np.float32),
        "delta_r_2": np.asarray([item["delta_r_2"] for item in records], dtype=np.float32),
        "mean_delta_r": np.asarray(
            [item["mean_delta_r"] for item in records], dtype=np.float32
        ),
    }
