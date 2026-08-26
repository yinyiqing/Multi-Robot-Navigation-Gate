#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
G26 = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线/26_数量泛化与外部切换基线"
)
Q1 = G26 / "local_data/q1"
VEHICLE_COUNTS = (3, 7)
METHODS = ("5a", "b2")
SEEDS = (20260911, 20260912)
SCENES = 128
HORIZON = 300
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_SAMPLES = 20000
SIGN_FLIP_SAMPLES = 100000


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bca_mean_interval(values, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=float)
    chunk = 1000
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
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
            z0
            + (z0 + z_alpha)
            / (1.0 - acceleration * (z0 + z_alpha))
        )
        probabilities.append(min(max(adjusted, 0.0), 1.0))
    return [float(np.quantile(bootstrap, probability)) for probability in probabilities]


def sign_flip_p(values, samples=SIGN_FLIP_SAMPLES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 1000
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        statistics = np.abs((signs * values).mean(axis=1))
        extreme += int(np.sum(statistics >= observed - 1e-15))
    return float((1 + extreme) / (samples + 1))


def aggregate(rows, num_agents):
    return {
        "full_success": float(rows[:, 8].astype(float).mean()),
        "agent_success": float(
            rows[:, 6].astype(float).sum() / (len(rows) * num_agents)
        ),
        "collision": float(
            rows[:, 7].astype(float).sum() / (len(rows) * num_agents)
        ),
        "unresolved": float(
            rows[:, 10].astype(float).sum() / (len(rows) * num_agents)
        ),
        "timeout": float(rows[:, 11].astype(float).mean()),
        "raw_steps": float(rows[:, 3].astype(float).mean()),
        "interaction_share": float(rows[:, 14].astype(float).mean()),
        "switches": float(rows[:, 15].astype(float).mean()),
    }


def metric_matrix(runs, method, column, scale=1.0):
    return np.stack(
        [runs[(method, seed)][:, column].astype(float) / scale for seed in SEEDS],
        axis=1,
    )


def effect_summary(values, include_sign_flip=False):
    output = {
        "mean_difference": float(np.mean(values)),
        "scene_cluster_bca_95_ci": bca_mean_interval(values),
    }
    if include_sign_flip:
        output["exploratory_sign_flip_two_sided_p"] = sign_flip_p(values)
    return output


def load_vehicle_count(num_agents):
    manifest_path = Q1 / f"manifests/n{num_agents}/test.json"
    completion_path = Q1 / f"results/q1_n{num_agents}_completion.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    scenarios = manifest["scenarios"]
    expected_ids = [str(item["scenario_id"]) for item in scenarios]

    if len(scenarios) != SCENES or len(set(expected_ids)) != SCENES:
        raise ValueError(f"n{num_agents} manifest must contain {SCENES} unique scenes")
    if any(int(item["num_agents"]) != num_agents for item in scenarios):
        raise ValueError(f"n{num_agents} manifest has inconsistent agent counts")
    if any(not item["validity"].get("gazebo_reset", False) for item in scenarios):
        raise ValueError(f"n{num_agents} manifest contains a failed Gazebo reset")
    if completion.get("status") != "complete":
        raise ValueError(f"n{num_agents} completion record is not complete")
    if completion.get("sealed_test_read") is not False:
        raise ValueError(f"n{num_agents} completion record has invalid sealed flag")
    if int(completion.get("num_agents", -1)) != num_agents:
        raise ValueError(f"n{num_agents} completion record has the wrong agent count")
    if int(completion.get("episodes_per_method", -1)) != SCENES:
        raise ValueError(f"n{num_agents} completion record has the wrong scene count")
    if tuple(completion.get("seeds", [])) != SEEDS:
        raise ValueError(f"n{num_agents} completion record has the wrong seeds")
    if tuple(completion.get("methods", [])) != METHODS:
        raise ValueError(f"n{num_agents} completion record has the wrong methods")
    manifest_sha256 = sha256_file(manifest_path)
    if completion.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"n{num_agents} completion record manifest hash mismatch")

    runs = {}
    result_hashes = {}
    for method in METHODS:
        for seed in SEEDS:
            path = Q1 / f"results/n{num_agents}/q1_n{num_agents}_{method}_s{seed}.npy"
            rows = np.load(path, allow_pickle=True)
            if rows.shape != (SCENES, 17):
                raise ValueError(f"invalid result shape: {path}")
            if [str(item) for item in rows[:, 12]] != expected_ids:
                raise ValueError(f"scenario order mismatch: {path}")
            terminal_count = sum(
                int(row[6]) + int(row[7]) + int(row[10]) for row in rows
            )
            if terminal_count != SCENES * num_agents:
                raise ValueError(f"terminal accounting mismatch: {path}")
            runs[(method, seed)] = rows
            result_hashes[f"{method}_s{seed}"] = sha256_file(path)

    return manifest_path, completion_path, manifest, scenarios, runs, result_hashes


def analyze_vehicle_count(num_agents):
    (
        manifest_path,
        completion_path,
        manifest,
        scenarios,
        runs,
        result_hashes,
    ) = load_vehicle_count(num_agents)
    per_repeat = {
        method: {
            str(seed): aggregate(runs[(method, seed)], num_agents) for seed in SEEDS
        }
        for method in METHODS
    }
    pooled = {
        method: aggregate(
            np.concatenate([runs[(method, seed)] for seed in SEEDS]), num_agents
        )
        for method in METHODS
    }

    matrices = {
        "full_success": (
            metric_matrix(runs, "5a", 8),
            metric_matrix(runs, "b2", 8),
        ),
        "agent_success": (
            metric_matrix(runs, "5a", 6, num_agents),
            metric_matrix(runs, "b2", 6, num_agents),
        ),
        "collision": (
            metric_matrix(runs, "5a", 7, num_agents),
            metric_matrix(runs, "b2", 7, num_agents),
        ),
        "unresolved": (
            metric_matrix(runs, "5a", 10, num_agents),
            metric_matrix(runs, "b2", 10, num_agents),
        ),
        "timeout": (
            metric_matrix(runs, "5a", 11),
            metric_matrix(runs, "b2", 11),
        ),
        "raw_steps": (
            metric_matrix(runs, "5a", 3),
            metric_matrix(runs, "b2", 3),
        ),
        "interaction_share": (
            metric_matrix(runs, "5a", 14),
            metric_matrix(runs, "b2", 14),
        ),
        "switches": (
            metric_matrix(runs, "5a", 15),
            metric_matrix(runs, "b2", 15),
        ),
    }
    effects = {}
    for name, (baseline, candidate) in matrices.items():
        scene_effect = (candidate - baseline).mean(axis=1)
        effects[name] = effect_summary(
            scene_effect, include_sign_flip=(name == "full_success")
        )

    baseline_full, candidate_full = matrices["full_success"]
    repeat_pairs = {}
    for index, seed in enumerate(SEEDS):
        delta = candidate_full[:, index] - baseline_full[:, index]
        repeat_pairs[str(seed)] = {
            "improved": int(np.sum(delta > 0)),
            "degraded": int(np.sum(delta < 0)),
            "tied": int(np.sum(delta == 0)),
        }
    effects["full_success"]["per_repeat_pairs"] = repeat_pairs

    baseline_steps, candidate_steps = matrices["raw_steps"]
    joint_success = (baseline_full == 1) & (candidate_full == 1)
    paired_scene = []
    for scene in range(SCENES):
        mask = joint_success[scene]
        if np.any(mask):
            paired_scene.append(
                float(np.mean((candidate_steps - baseline_steps)[scene, mask]))
            )
    effects["paired_success_steps"] = {
        "joint_success_scene_repeat_pairs": int(np.sum(joint_success)),
        "scene_clusters_with_pairs": len(paired_scene),
        **effect_summary(paired_scene),
    }
    baseline_penalized = np.where(baseline_full == 1, baseline_steps, HORIZON)
    candidate_penalized = np.where(candidate_full == 1, candidate_steps, HORIZON)
    penalized_scene = (candidate_penalized - baseline_penalized).mean(axis=1)
    effects["penalized_completion_steps"] = {
        "horizon": HORIZON,
        **effect_summary(penalized_scene),
    }

    edge_counts = np.asarray(
        [int(item["metrics"]["conflict_edge_count"]) for item in scenarios]
    )
    edge_bands = {}
    for band, mask in (
        ("0", edge_counts == 0),
        ("1", edge_counts == 1),
        ("2", edge_counts == 2),
        ("3+", edge_counts >= 3),
    ):
        count = int(np.sum(mask))
        if not count:
            edge_bands[band] = {"scenes": 0}
            continue
        metrics = {}
        for name in (
            "full_success",
            "agent_success",
            "collision",
            "unresolved",
            "timeout",
            "raw_steps",
            "interaction_share",
            "switches",
        ):
            baseline, candidate = matrices[name]
            metrics[name] = {
                "5a": float(baseline[mask].mean()),
                "b2": float(candidate[mask].mean()),
                "mean_difference": float((candidate - baseline)[mask].mean()),
            }
        edge_bands[band] = {"scenes": count, **metrics}

    gazebo = manifest["gazebo_validation"]
    return {
        "manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(manifest_path),
            "generation_seed": int(manifest["master_seed"]),
            "generated_candidate_count": int(manifest["candidate_count"]),
            "gazebo_candidate_count": int(gazebo["candidate_count"]),
            "gazebo_accepted_count": int(gazebo["accepted_count"]),
            "gazebo_rejected_count": int(gazebo["rejected_count"]),
            "conflict_edge_count_min": int(edge_counts.min()),
            "conflict_edge_count_max": int(edge_counts.max()),
            "conflict_edge_count_mean": float(edge_counts.mean()),
        },
        "completion_sha256": sha256_file(completion_path),
        "result_sha256": result_hashes,
        "per_repeat": per_repeat,
        "pooled_descriptive": pooled,
        "exploratory_b2_minus_5a": effects,
        "by_conflict_edges_descriptive": edge_bands,
    }


def main():
    output = {
        "protocol": {
            "experiment_id": "G26-Q1-quantity-generalization",
            "methods": list(METHODS),
            "vehicle_counts": list(VEHICLE_COUNTS),
            "seeds": list(SEEDS),
            "scene_clusters_per_vehicle_count": SCENES,
            "repeats": len(SEEDS),
            "total_episodes": len(VEHICLE_COUNTS)
            * SCENES
            * len(SEEDS)
            * len(METHODS),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "sign_flip_samples": SIGN_FLIP_SAMPLES,
            "random_seed": BOOTSTRAP_SEED,
            "inference_status": "exploratory_post_hoc_g25_matched",
            "g25_confirmatory_statistics_modified": False,
            "actor_or_router_updated": False,
        },
        "vehicle_counts": {
            str(num_agents): analyze_vehicle_count(num_agents)
            for num_agents in VEHICLE_COUNTS
        },
        "interpretation_guardrails": [
            "Compare B2 with 5A only within the same vehicle count, scene, and repeat.",
            "Do not compare absolute 3-, 5-, and 7-robot full-success rates as a quantity effect.",
            "The BCa intervals and sign-flip p-values are exploratory supplemental inference, not G25 confirmatory tests.",
        ],
    }
    output_path = Q1 / "results/q1_statistics.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"sha256 {sha256_file(output_path)}  {output_path}")


if __name__ == "__main__":
    main()
