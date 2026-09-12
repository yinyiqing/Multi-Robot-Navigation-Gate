#!/usr/bin/env python3
import gzip
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G27 = BASE / "27_反事实Reward增强Gate监督"
MANIFEST = G27 / "local_data/test/manifests/dense_test_384_640.json.gz"
MANIFEST_RECORD = G27 / "local_data/test/manifests/manifest_record.json"
RESULT_DIR = G27 / "local_data/test/results"
COMPLETION = G27 / "local_data/test/p4_completion.json"
P3_SUMMARY = G27 / "local_data/validation/p3_summary.json"
P3_RESULT = G27 / "local_data/validation/results/g27_dense256_b3_s20260810.npy"
AMENDMENT = G27 / "local_data/protocol/p4_continuation_amendment.json"
B3 = G27 / "local_data/training/seed20260910/best.pt"
METHODS = ("5a", "b2", "b3")
SEEDS = (20260911, 20260912, 20260913)
SCENES = 256
AGENTS = 5
HORIZON = 300
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_SAMPLES = 20000
SIGN_FLIP_SAMPLES = 100000


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bca_mean_interval(values, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=float)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        bootstrap[start : start + count] = values[indices].mean(axis=1)
    observed = float(values.mean())
    normal = NormalDist()
    fraction = (
        np.sum(bootstrap < observed) + 0.5 * np.sum(bootstrap == observed)
    ) / samples
    fraction = min(
        max(float(fraction), 1.0 / (2 * samples)),
        1.0 - 1.0 / (2 * samples),
    )
    z0 = normal.inv_cdf(fraction)
    total = float(values.sum())
    jackknife = (total - values) / (len(values) - 1)
    centered = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.sum(centered**2)) ** 1.5
    acceleration = float(np.sum(centered**3)) / denominator if denominator else 0.0
    probabilities = []
    for alpha in (0.025, 0.975):
        z_alpha = normal.inv_cdf(alpha)
        adjusted = normal.cdf(
            z0 + (z0 + z_alpha) / (1.0 - acceleration * (z0 + z_alpha))
        )
        probabilities.append(min(max(adjusted, 0.0), 1.0))
    return [float(np.quantile(bootstrap, probability)) for probability in probabilities]


def sign_flip_p(values, samples=SIGN_FLIP_SAMPLES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        statistics = np.abs((signs * values).mean(axis=1))
        extreme += int(np.sum(statistics >= observed - 1e-15))
    return float((1 + extreme) / (samples + 1))


def aggregate(rows):
    return {
        "full_success": float(rows[:, 8].astype(float).mean()),
        "agent_success": float(rows[:, 6].astype(float).sum() / (len(rows) * AGENTS)),
        "robot_collision": float(rows[:, 7].astype(float).sum() / (len(rows) * AGENTS)),
        "robot_unresolved": float(rows[:, 10].astype(float).sum() / (len(rows) * AGENTS)),
        "episode_timeout": float(rows[:, 11].astype(float).mean()),
        "raw_steps": float(rows[:, 3].astype(float).mean()),
        "interaction_selection_share": float(rows[:, 14].astype(float).mean()),
        "switches": float(rows[:, 15].astype(float).mean()),
    }


def metric_matrix(runs, method, column, scale=1.0):
    return np.stack(
        [runs[(method, seed)][:, column].astype(float) / scale for seed in SEEDS],
        axis=1,
    )


def effect_summary(values, sign_flip=False):
    values = np.asarray(values, dtype=float)
    output = {
        "mean_difference": float(values.mean()),
        "scene_cluster_bca_95_ci": bca_mean_interval(values),
    }
    if sign_flip:
        output["sign_flip_two_sided_p"] = sign_flip_p(values)
    return output


def comparison(runs, baseline, candidate):
    metrics = {
        "full_success": (8, 1.0),
        "agent_success": (6, float(AGENTS)),
        "robot_collision": (7, float(AGENTS)),
        "robot_unresolved": (10, float(AGENTS)),
        "episode_timeout": (11, 1.0),
        "raw_steps": (3, 1.0),
        "interaction_selection_share": (14, 1.0),
        "switches": (15, 1.0),
    }
    output = {}
    for name, (column, scale) in metrics.items():
        base = metric_matrix(runs, baseline, column, scale)
        value = metric_matrix(runs, candidate, column, scale)
        output[name] = effect_summary(
            (value - base).mean(axis=1), sign_flip=(name == "full_success")
        )

    base_full = metric_matrix(runs, baseline, 8)
    value_full = metric_matrix(runs, candidate, 8)
    repeat_pairs = {}
    for index, seed in enumerate(SEEDS):
        delta = value_full[:, index] - base_full[:, index]
        repeat_pairs[str(seed)] = {
            "improved": int(np.sum(delta > 0)),
            "degraded": int(np.sum(delta < 0)),
            "tied": int(np.sum(delta == 0)),
        }
    output["full_success"]["per_repeat_pairs"] = repeat_pairs

    base_steps = metric_matrix(runs, baseline, 3)
    value_steps = metric_matrix(runs, candidate, 3)
    joint_success = (base_full == 1) & (value_full == 1)
    paired_scene = []
    for scene in range(SCENES):
        mask = joint_success[scene]
        if np.any(mask):
            paired_scene.append(float(np.mean((value_steps - base_steps)[scene, mask])))
    output["paired_success_steps"] = {
        "joint_success_scene_repeat_pairs": int(np.sum(joint_success)),
        "scene_clusters_with_pairs": len(paired_scene),
        **effect_summary(paired_scene),
    }

    base_penalized = np.where(base_full == 1, base_steps, HORIZON)
    value_penalized = np.where(value_full == 1, value_steps, HORIZON)
    output["penalized_completion_steps"] = {
        "horizon": HORIZON,
        **effect_summary((value_penalized - base_penalized).mean(axis=1)),
    }
    return output


def load_runs():
    for path in (
        MANIFEST,
        MANIFEST_RECORD,
        COMPLETION,
        P3_SUMMARY,
        P3_RESULT,
        AMENDMENT,
        B3,
    ):
        if not path.is_file():
            raise SystemExit("P4 is incomplete: missing %s" % path)
    completion = json.loads(COMPLETION.read_text(encoding="utf-8"))
    record = json.loads(MANIFEST_RECORD.read_text(encoding="utf-8"))
    p3 = json.loads(P3_SUMMARY.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("sealed_test_read") is not True:
        raise ValueError("invalid P4 completion record")
    criteria = p3.get("admission_criteria", {})
    required = (
        "full_success_not_below_b2",
        "robot_collision_increase_at_most_2pp",
        "episode_timeout_increase_at_most_1pp",
    )
    if p3.get("admission_passed") is not False or not all(
        criteria.get(name) is True for name in required
    ):
        raise ValueError("P3 success/safety history does not match the amendment")
    if amendment.get("decision") != "continue_to_independent_test_with_frozen_b3":
        raise ValueError("P4 continuation amendment is invalid")
    amendment_hash = sha256_file(AMENDMENT)
    p3_hash = sha256_file(P3_SUMMARY)
    p3_result_hash = sha256_file(P3_RESULT)
    b3_hash = sha256_file(B3)
    if record.get("p3_summary_sha256") != p3_hash:
        raise ValueError("manifest record does not match the frozen P3 summary")
    if amendment.get("p3_summary_sha256") != p3_hash:
        raise ValueError("amendment does not match the frozen P3 summary")
    if record.get("p3_result_sha256") != p3_result_hash:
        raise ValueError("manifest record does not match the frozen P3 result")
    if amendment.get("p3_result_sha256") != p3_result_hash:
        raise ValueError("amendment does not match the frozen P3 result")
    if p3.get("b3_checkpoint_sha256") != b3_hash:
        raise ValueError("P3 summary does not match the frozen B3 checkpoint")
    if amendment.get("b3_checkpoint_sha256") != b3_hash:
        raise ValueError("amendment does not match the frozen B3 checkpoint")
    if record.get("b3_checkpoint_sha256") != b3_hash:
        raise ValueError("manifest record does not match the frozen B3 checkpoint")
    if record.get("continuation_amendment_sha256") != amendment_hash:
        raise ValueError("manifest record does not match the P4 amendment")
    if completion.get("continuation_amendment_sha256") != amendment_hash:
        raise ValueError("completion record does not match the P4 amendment")
    manifest_hash = sha256_file(MANIFEST)
    if completion.get("manifest_sha256") != manifest_hash or record.get("output_sha256") != manifest_hash:
        raise ValueError("P4 manifest hash mismatch")
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    expected_ids = [str(item["scenario_id"]) for item in scenarios]
    if len(expected_ids) != SCENES or len(set(expected_ids)) != SCENES:
        raise ValueError("P4 manifest must contain 256 unique scenes")

    runs = {}
    hashes = {}
    for method in METHODS:
        for seed in SEEDS:
            path = RESULT_DIR / ("g27_p4_%s_s%d.npy" % (method, seed))
            rows = np.load(path, allow_pickle=True)
            if rows.shape != (SCENES, 17):
                raise ValueError("invalid result shape: %s" % path)
            if [str(item) for item in rows[:, 12]] != expected_ids:
                raise ValueError("scenario order mismatch: %s" % path)
            if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != SCENES * AGENTS:
                raise ValueError("terminal accounting mismatch: %s" % path)
            key = "%s_s%d" % (method, seed)
            hashes[key] = sha256_file(path)
            if completion.get("result_sha256", {}).get(key) != hashes[key]:
                raise ValueError("completion hash mismatch: %s" % path)
            runs[(method, seed)] = rows
    return scenarios, runs, hashes, completion, record, p3, amendment


def evaluate_confirmation_criteria(comparisons):
    main_effect = comparisons["b3_minus_5a"]
    replacement_effect = comparisons["b3_minus_b2"]
    return {
        "b3_full_success_over_5a_confirmed": (
            main_effect["full_success"]["scene_cluster_bca_95_ci"][0] > 0.0
            and main_effect["full_success"]["sign_flip_two_sided_p"] < 0.05
        ),
        "b3_collision_below_5a_confirmed": (
            main_effect["robot_collision"]["scene_cluster_bca_95_ci"][1] < 0.0
        ),
        "b3_full_success_not_below_b2": (
            replacement_effect["full_success"]["mean_difference"] >= 0.0
        ),
        "b3_collision_increase_vs_b2_at_most_2pp": (
            replacement_effect["robot_collision"]["mean_difference"] <= 0.02
        ),
        "b3_timeout_increase_vs_b2_at_most_1pp": (
            replacement_effect["episode_timeout"]["mean_difference"] <= 0.01
        ),
    }


def main():
    scenarios, runs, hashes, completion, record, p3, amendment = load_runs()
    p3_hash = sha256_file(P3_SUMMARY)
    p3_result_hash = sha256_file(P3_RESULT)
    b3_hash = sha256_file(B3)
    pooled = {
        method: aggregate(np.concatenate([runs[(method, seed)] for seed in SEEDS]))
        for method in METHODS
    }
    comparisons = {
        "b3_minus_5a": comparison(runs, "5a", "b3"),
        "b3_minus_b2": comparison(runs, "b2", "b3"),
        "b2_minus_5a": comparison(runs, "5a", "b2"),
    }
    criteria = evaluate_confirmation_criteria(comparisons)

    edge_counts = np.asarray(
        [int(item["metrics"]["conflict_edge_count"]) for item in scenarios]
    )
    by_edges = {}
    for band, mask in (
        ("0", edge_counts == 0),
        ("1", edge_counts == 1),
        ("2", edge_counts == 2),
        ("3+", edge_counts >= 3),
    ):
        repeated_mask = np.tile(mask, len(SEEDS))
        by_edges[band] = {"scenes": int(np.sum(mask))}
        for method in METHODS:
            rows = np.concatenate([runs[(method, seed)] for seed in SEEDS])
            by_edges[band][method] = aggregate(rows[repeated_mask]) if np.any(mask) else None

    output = {
        "format_version": 1,
        "protocol": {
            "experiment_id": "G27-P4-independent-test",
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "scene_clusters": SCENES,
            "source_slice": "dense/test original order [384:640]",
            "total_episodes": SCENES * len(SEEDS) * len(METHODS),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "sign_flip_samples": SIGN_FLIP_SAMPLES,
            "random_seed": BOOTSTRAP_SEED,
            "primary_alpha_two_sided": 0.05,
            "sealed_test_read": True,
            "actor_or_router_updated": False,
        },
        "manifest_sha256": sha256_file(MANIFEST),
        "manifest_record_sha256": sha256_file(MANIFEST_RECORD),
        "p3_summary_sha256": p3_hash,
        "p3_result_sha256": p3_result_hash,
        "continuation_amendment_sha256": sha256_file(AMENDMENT),
        "b3_checkpoint_sha256": b3_hash,
        "completion_sha256": sha256_file(COMPLETION),
        "result_sha256": hashes,
        "pooled_descriptive": pooled,
        "comparisons": comparisons,
        "by_conflict_edges_descriptive": by_edges,
        "confirmation_criteria": criteria,
        "confirmation_passed": all(criteria.values()),
        "next_action": (
            "replace the paper method and main results with the reward-aware router"
            if all(criteria.values())
            else "retain the proximity-only router and G25 as the paper method and confirmatory result"
        ),
        "guardrails": [
            "P4 is not pooled with G25 or G26.",
            "All comparisons use matched scenes and environment repeats.",
            "Only the prespecified B3 checkpoint is evaluated.",
            "Efficiency and interaction-selection metrics are reported but are not adoption criteria.",
        ],
    }
    output_path = G27 / "local_data/test/p4_statistics.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["confirmation_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
