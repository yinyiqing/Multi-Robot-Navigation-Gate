#!/usr/bin/env python3
import gzip
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
G25 = BASE / "25_最终消融与Sealed评测/local_data"
MANIFEST = G25 / "sealed_manifest/dense_test_first256.json.gz"
RESULT_DIR = G25 / "sealed/results"
COMPLETION = G25 / "sealed/sealed_completion.json"
METHODS = ("5a", "epoch16", "min_lidar", "ttc_cpa", "b2", "privileged_2m", "r2b")
SEEDS = (20260901, 20260902, 20260903)
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
    fraction = (np.sum(bootstrap < observed) + 0.5 * np.sum(bootstrap == observed)) / samples
    fraction = min(max(float(fraction), 1.0 / (2 * samples)), 1.0 - 1.0 / (2 * samples))
    z0 = normal.inv_cdf(fraction)
    total = float(values.sum())
    jackknife = (total - values) / (len(values) - 1)
    centered = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.sum(centered**2)) ** 1.5
    acceleration = float(np.sum(centered**3)) / denominator if denominator else 0.0
    probabilities = []
    for alpha in (0.025, 0.975):
        z_alpha = normal.inv_cdf(alpha)
        adjusted = normal.cdf(z0 + (z0 + z_alpha) / (1.0 - acceleration * (z0 + z_alpha)))
        probabilities.append(min(max(adjusted, 0.0), 1.0))
    return [float(np.quantile(bootstrap, p)) for p in probabilities]


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


def aggregate(rows):
    return {
        "full_success": float(rows[:, 8].astype(float).mean()),
        "agent_success": float(rows[:, 6].astype(float).sum() / (len(rows) * 5)),
        "collision": float(rows[:, 7].astype(float).sum() / (len(rows) * 5)),
        "unresolved": float(rows[:, 10].astype(float).sum() / (len(rows) * 5)),
        "timeout": float(rows[:, 11].astype(float).mean()),
        "raw_steps": float(rows[:, 3].astype(float).mean()),
        "interaction_share": float(rows[:, 14].astype(float).mean()),
        "switches": float(rows[:, 15].astype(float).mean()),
    }


def metric_matrix(runs, method, column, scale=1.0):
    return np.stack([runs[(method, seed)][:, column].astype(float) / scale for seed in SEEDS], axis=1)


def main():
    if not COMPLETION.is_file():
        raise SystemExit("sealed evaluation is incomplete")
    completion = json.loads(COMPLETION.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or not completion.get("sealed_test_read"):
        raise SystemExit("invalid sealed completion record")
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    expected_ids = [str(item["scenario_id"]) for item in scenarios]
    if len(expected_ids) != 256:
        raise ValueError("sealed manifest must contain exactly 256 scenes")
    runs = {}
    hashes = {}
    for method in METHODS:
        for seed in SEEDS:
            path = RESULT_DIR / ("g25_sealed_%s_s%d.npy" % (method, seed))
            rows = np.load(path, allow_pickle=True)
            if rows.shape != (256, 17):
                raise ValueError("invalid result shape: %s" % path)
            if [str(item) for item in rows[:, 12]] != expected_ids:
                raise ValueError("scenario order mismatch: %s" % path)
            if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != 1280:
                raise ValueError("terminal accounting mismatch: %s" % path)
            runs[(method, seed)] = rows
            hashes["%s_s%d" % (method, seed)] = sha256_file(path)

    per_repeat = {
        method: {str(seed): aggregate(runs[(method, seed)]) for seed in SEEDS}
        for method in METHODS
    }
    pooled = {
        method: aggregate(np.concatenate([runs[(method, seed)] for seed in SEEDS]))
        for method in METHODS
    }
    b2_full = metric_matrix(runs, "b2", 8)
    base_full = metric_matrix(runs, "5a", 8)
    full_scene_effect = (b2_full - base_full).mean(axis=1)
    repeat_pairs = {}
    for index, seed in enumerate(SEEDS):
        delta = b2_full[:, index] - base_full[:, index]
        repeat_pairs[str(seed)] = {
            "improved": int(np.sum(delta > 0)),
            "degraded": int(np.sum(delta < 0)),
            "tied": int(np.sum(delta == 0)),
        }

    effects = {}
    for name, column, scale in (
        ("full_success", 8, 1.0),
        ("collision", 7, 5.0),
        ("timeout", 11, 1.0),
        ("raw_steps", 3, 1.0),
        ("interaction_share", 14, 1.0),
    ):
        scene_effect = (
            metric_matrix(runs, "b2", column, scale)
            - metric_matrix(runs, "5a", column, scale)
        ).mean(axis=1)
        effects[name] = {
            "mean_difference": float(scene_effect.mean()),
            "scene_cluster_bca_95_ci": bca_mean_interval(scene_effect),
        }

    b2_steps = metric_matrix(runs, "b2", 3)
    base_steps = metric_matrix(runs, "5a", 3)
    joint_success = (b2_full == 1) & (base_full == 1)
    paired_scene = []
    for scene in range(256):
        mask = joint_success[scene]
        if np.any(mask):
            paired_scene.append(float(np.mean((b2_steps - base_steps)[scene, mask])))
    effects["paired_success_steps"] = {
        "joint_success_scene_repeat_pairs": int(np.sum(joint_success)),
        "scene_clusters_with_pairs": len(paired_scene),
        "mean_difference": float(np.mean(paired_scene)) if paired_scene else float("nan"),
        "scene_cluster_bca_95_ci": bca_mean_interval(paired_scene),
    }
    b2_penalized = np.where(b2_full == 1, b2_steps, 300.0)
    base_penalized = np.where(base_full == 1, base_steps, 300.0)
    penalized_scene = (b2_penalized - base_penalized).mean(axis=1)
    effects["penalized_completion_steps"] = {
        "horizon": 300,
        "mean_difference": float(penalized_scene.mean()),
        "scene_cluster_bca_95_ci": bca_mean_interval(penalized_scene),
    }

    edge_bands = []
    for item in scenarios:
        count = int(item["metrics"]["conflict_edge_count"])
        edge_bands.append(str(count) if count < 3 else "3+")
    heterogeneity = {}
    for band in ("0", "1", "2", "3+"):
        mask = np.asarray([item == band for item in edge_bands])
        heterogeneity[band] = {
            "scenes": int(np.sum(mask)),
            "5a_full_success": float(base_full[mask].mean()) if np.any(mask) else None,
            "b2_full_success": float(b2_full[mask].mean()) if np.any(mask) else None,
            "mean_difference": float((b2_full - base_full)[mask].mean()) if np.any(mask) else None,
        }

    output = {
        "protocol": {
            "experiment_id": "G25-sealed-statistics",
            "manifest_sha256": sha256_file(MANIFEST),
            "seeds": list(SEEDS),
            "scene_clusters": 256,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "sign_flip_samples": SIGN_FLIP_SAMPLES,
            "random_seed": BOOTSTRAP_SEED,
            "primary_alpha_two_sided": 0.05,
            "sealed_test_read": True,
        },
        "result_sha256": hashes,
        "per_repeat": per_repeat,
        "pooled_descriptive": pooled,
        "primary_b2_minus_5a": {
            "mean_full_success_difference": float(full_scene_effect.mean()),
            "scene_cluster_bca_95_ci": bca_mean_interval(full_scene_effect),
            "sign_flip_two_sided_p": sign_flip_p(full_scene_effect),
            "per_repeat_pairs": repeat_pairs,
        },
        "secondary_b2_minus_5a": effects,
        "by_conflict_edges_descriptive": heterogeneity,
    }
    path = G25 / "sealed/sealed_statistics.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
