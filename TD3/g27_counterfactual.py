import hashlib
import json
import math
from collections import Counter

import numpy as np


AGENT_NAMES = ("r1", "r2", "r3", "r4", "r5")
BRANCH_NAMES = ("N1", "N2", "I1", "I2")
ALIGNMENT_LIMITS = {
    "position_error": 0.02,
    "yaw_error": 0.02,
    "linear_velocity_error": 0.02,
    "angular_velocity_error": 0.03,
}
REWARD_WEIGHTS = {
    "goal": 100.0,
    "collision": -100.0,
    "progress": 20.0,
    "forward": 0.5,
    "turn": 0.2,
    "obstacle": 0.5,
    "stagnation": 0.03,
}


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selection_hash(seed, scenario_id, step, ego_index):
    value = "%s|%s|%s|%s" % (seed, scenario_id, int(step), int(ego_index))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def distance_stratum(distance):
    value = float(distance)
    if value <= 1.2:
        return "deep"
    if value <= 2.0:
        return "interaction"
    if value <= 3.0:
        return "near"
    return "far"


def action_stratum(disagreement, threshold=0.25):
    return "low" if float(disagreement) < float(threshold) else "high"


def select_stratified_anchors(candidates, per_cell=4):
    if int(per_cell) < 1:
        raise ValueError("per_cell must be positive")
    cells = [
        (distance, action)
        for distance in ("deep", "interaction", "near", "far")
        for action in ("low", "high")
    ]
    selected = []
    used_scenarios = set()
    for cell in cells:
        eligible = sorted(
            (
                item
                for item in candidates
                if (item["distance_stratum"], item["action_stratum"]) == cell
            ),
            key=lambda item: item["selection_hash"],
        )
        chosen = []
        for item in eligible:
            if item["scenario_id"] in used_scenarios:
                continue
            chosen.append(dict(item))
            used_scenarios.add(item["scenario_id"])
            if len(chosen) == int(per_cell):
                break
        if len(chosen) != int(per_cell):
            raise ValueError(
                "not enough unique scenarios for cell %s/%s: %d/%d"
                % (cell[0], cell[1], len(chosen), int(per_cell))
            )
        selected.extend(chosen)
    return selected


def select_stratified_anchors_matching(candidates, per_cell):
    """Fill every stratum quota while assigning each scene at most once."""
    if int(per_cell) < 1:
        raise ValueError("per_cell must be positive")
    cells = [
        (distance, action)
        for distance in ("deep", "interaction", "near", "far")
        for action in ("low", "high")
    ]
    best = {}
    for item in candidates:
        cell = (item["distance_stratum"], item["action_stratum"])
        if cell not in cells:
            continue
        key = (cell, item["scenario_id"])
        if key not in best or item["selection_hash"] < best[key]["selection_hash"]:
            best[key] = dict(item)
    neighbors = {
        cell: sorted(
            (scene for candidate_cell, scene in best if candidate_cell == cell),
            key=lambda scene: best[(cell, scene)]["selection_hash"],
        )
        for cell in cells
    }
    slots = [cell for cell in cells for _ in range(int(per_cell))]
    scene_to_slot = {}
    slot_to_scene = {}

    def assign(slot_index, seen_scenes):
        cell = slots[slot_index]
        for scene in neighbors[cell]:
            if scene in seen_scenes:
                continue
            seen_scenes.add(scene)
            previous = scene_to_slot.get(scene)
            if previous is None or assign(previous, seen_scenes):
                scene_to_slot[scene] = slot_index
                slot_to_scene[slot_index] = scene
                return True
        return False

    for slot_index in range(len(slots)):
        if not assign(slot_index, set()):
            cell = slots[slot_index]
            filled = sum(
                int(slot_to_scene.get(index) is not None)
                for index, other_cell in enumerate(slots)
                if other_cell == cell
            )
            raise ValueError(
                "cannot satisfy unique-scene quota for cell %s/%s: %d/%d"
                % (cell[0], cell[1], filled, int(per_cell))
            )
    selected = [
        best[(slots[index], slot_to_scene[index])] for index in range(len(slots))
    ]
    if len({item["scenario_id"] for item in selected}) != len(selected):
        raise AssertionError("matching reused a scenario")
    return selected


def deployed_action(raw_action):
    values = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("raw Actor action must contain two finite values")
    return np.asarray([(values[0] + 1.0) / 2.0, values[1]], dtype=np.float64)


def base_reward_components(target, collision, action, min_laser, progress):
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        raise ValueError("deployed action must contain two finite values")
    values = {
        "goal": 0.0,
        "collision": 0.0,
        "progress": 0.0,
        "forward": 0.0,
        "turn": 0.0,
        "obstacle": 0.0,
        "stagnation": 0.0,
    }
    if bool(target):
        values["goal"] = REWARD_WEIGHTS["goal"]
    elif bool(collision):
        values["collision"] = REWARD_WEIGHTS["collision"]
    else:
        obstacle = 1.0 - float(min_laser) if float(min_laser) < 1.0 else 0.0
        values["progress"] = REWARD_WEIGHTS["progress"] * float(progress)
        values["forward"] = REWARD_WEIGHTS["forward"] * float(action[0])
        values["turn"] = -REWARD_WEIGHTS["turn"] * abs(float(action[1]))
        values["obstacle"] = -REWARD_WEIGHTS["obstacle"] * obstacle
        if float(action[0]) < 0.1 and abs(float(progress)) < 0.01:
            values["stagnation"] = -REWARD_WEIGHTS["stagnation"]
    values["total"] = float(sum(values.values()))
    return values


def angle_error(first, second):
    return abs(float((float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi))


def alignment_metrics(reference, observed):
    if tuple(reference["agent_names"]) != tuple(observed["agent_names"]):
        raise ValueError("snapshot agent order differs")
    names = tuple(reference["agent_names"])
    metrics = {
        "active_mask_match": list(reference["active_mask"])
        == list(observed["active_mask"]),
        "position_error": max(
            float(
                np.linalg.norm(
                    np.asarray(observed["agents"][name]["position"], dtype=float)
                    - np.asarray(reference["agents"][name]["position"], dtype=float)
                )
            )
            for name in names
        ),
        "yaw_error": max(
            angle_error(
                observed["agents"][name]["yaw"], reference["agents"][name]["yaw"]
            )
            for name in names
        ),
        "linear_velocity_error": max(
            float(
                np.linalg.norm(
                    np.asarray(
                        observed["agents"][name]["linear_velocity"], dtype=float
                    )
                    - np.asarray(
                        reference["agents"][name]["linear_velocity"], dtype=float
                    )
                )
            )
            for name in names
        ),
        "angular_velocity_error": max(
            abs(
                float(observed["agents"][name]["angular_velocity"])
                - float(reference["agents"][name]["angular_velocity"])
            )
            for name in names
        ),
    }
    metrics["passed"] = bool(
        metrics["active_mask_match"]
        and all(metrics[key] <= limit for key, limit in ALIGNMENT_LIMITS.items())
    )
    return metrics


def validate_branch_spec(specification):
    required = {
        "format_version",
        "scenario_id",
        "scenario_index",
        "anchor_step",
        "ego_index",
        "agent_names",
        "action_prefix",
        "anchor_snapshot",
        "navigation_actions",
        "interaction_actions",
    }
    missing = sorted(required - set(specification))
    if missing:
        raise ValueError("branch specification is missing: %s" % ", ".join(missing))
    if int(specification["format_version"]) != 1:
        raise ValueError("unsupported branch specification format")
    if tuple(specification["agent_names"]) != AGENT_NAMES:
        raise ValueError("branch specification requires r1--r5 in order")
    ego_index = int(specification["ego_index"])
    if not 0 <= ego_index < len(AGENT_NAMES):
        raise ValueError("ego_index is out of range")
    if len(specification["action_prefix"]) != int(specification["anchor_step"]):
        raise ValueError("action prefix length does not match anchor_step")
    for key in ("navigation_actions", "interaction_actions"):
        values = np.asarray(specification[key], dtype=float)
        if values.shape != (len(AGENT_NAMES), 2) or not np.all(np.isfinite(values)):
            raise ValueError("%s must have shape [5,2] with finite values" % key)
    for actions in specification["action_prefix"]:
        values = np.asarray(actions, dtype=float)
        if values.shape != (len(AGENT_NAMES), 2) or not np.all(np.isfinite(values)):
            raise ValueError("each action-prefix item must have shape [5,2]")
    return specification


def percentile_higher(values, percentile):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("percentile values must be a non-empty finite vector")
    try:
        return float(np.percentile(values, percentile, method="higher"))
    except TypeError:
        return float(np.percentile(values, percentile, interpolation="higher"))


def analyze_pilot(anchor_records):
    if not anchor_records:
        raise ValueError("pilot analysis requires anchor records")
    aligned = []
    same_actor_exact = True
    alignment_failures = Counter()
    for anchor in anchor_records:
        branches = anchor.get("branches", {})
        if set(branches) != set(BRANCH_NAMES):
            raise ValueError("every anchor must contain N1/N2/I1/I2")
        anchor_aligned = all(bool(branches[name]["alignment"]["passed"]) for name in BRANCH_NAMES)
        if not anchor_aligned:
            for name in BRANCH_NAMES:
                if not branches[name]["alignment"]["passed"]:
                    alignment_failures[name] += 1
            continue
        aligned.append(anchor)
        for left, right in (("N1", "N2"), ("I1", "I2")):
            same_actor_exact &= (
                bool(branches[left]["target"]) == bool(branches[right]["target"])
                and bool(branches[left]["collision"])
                == bool(branches[right]["collision"])
            )

    same_actor_noise = []
    for anchor in aligned:
        branches = anchor["branches"]
        same_actor_noise.extend(
            [
                abs(float(branches["N1"]["reward"]) - float(branches["N2"]["reward"])),
                abs(float(branches["I1"]["reward"]) - float(branches["I2"]["reward"])),
            ]
        )
    noise_threshold = percentile_higher(same_actor_noise, 95.0) if same_actor_noise else None
    comparisons = []
    if noise_threshold is not None:
        for anchor in aligned:
            branches = anchor["branches"]
            delta_1 = float(branches["I1"]["reward"]) - float(branches["N1"]["reward"])
            delta_2 = float(branches["I2"]["reward"]) - float(branches["N2"]["reward"])
            mean_delta = 0.5 * (delta_1 + delta_2)
            resolved = abs(mean_delta) > noise_threshold
            direction_agrees = bool(delta_1 * delta_2 > 0.0)
            comparisons.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "delta_r_1": delta_1,
                    "delta_r_2": delta_2,
                    "mean_delta_r": mean_delta,
                    "resolved": resolved,
                    "direction_agrees": direction_agrees,
                    "preferred_actor": (
                        "interaction" if mean_delta > 0.0 else "navigation"
                    )
                    if resolved
                    else "unresolved",
                }
            )
    resolved = [item for item in comparisons if item["resolved"]]
    agreement = (
        float(np.mean([item["direction_agrees"] for item in resolved]))
        if resolved
        else 0.0
    )
    support = Counter(item["preferred_actor"] for item in resolved)
    total = len(anchor_records)
    aligned_fraction = len(aligned) / total
    resolved_fraction = len(resolved) / len(aligned) if aligned else 0.0
    criteria = {
        "alignment_at_least_90pct": aligned_fraction >= 0.90,
        "same_actor_terminal_exact": bool(same_actor_exact and bool(aligned)),
        "noise_threshold_defined": noise_threshold is not None,
        "resolved_at_least_25pct": resolved_fraction >= 0.25,
        "direction_agreement_at_least_80pct": agreement >= 0.80,
        "both_actors_supported": support["navigation"] > 0 and support["interaction"] > 0,
    }
    return {
        "format_version": 1,
        "anchors_total": total,
        "anchors_aligned": len(aligned),
        "alignment_fraction": aligned_fraction,
        "alignment_failures_by_branch": dict(alignment_failures),
        "same_actor_terminal_exact": bool(same_actor_exact and bool(aligned)),
        "same_actor_noise_absolute_differences": same_actor_noise,
        "noise_threshold_p95_higher": noise_threshold,
        "comparisons": comparisons,
        "resolved_anchors": len(resolved),
        "resolved_fraction_of_aligned": resolved_fraction,
        "resolved_direction_agreement": agreement,
        "resolved_actor_support": dict(support),
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
