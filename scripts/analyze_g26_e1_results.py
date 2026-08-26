#!/usr/bin/env python3
import gzip
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
G26 = ROOT / "experiments/03_保留专门化/02_论文主线/26_数量泛化与外部切换基线"
E1 = G26 / "local_data/e1"
MANIFEST = E1 / "manifests/dense_test_256_384.json.gz"
RESULT_DIR = E1 / "results"
COMPLETION = E1 / "e1_completion.json"
METHODS = ("5a", "nf_switch", "b2")
SEEDS = (20260921, 20260922)
SCENES = 128
AGENTS = 5
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
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        bootstrap[start : start + count] = values[indices].mean(axis=1)
    observed = float(values.mean())
    normal = NormalDist()
    fraction = (
        np.sum(bootstrap < observed) + 0.5 * np.sum(bootstrap == observed)
    ) / samples
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
        "collision": float(rows[:, 7].astype(float).sum() / (len(rows) * AGENTS)),
        "unresolved": float(rows[:, 10].astype(float).sum() / (len(rows) * AGENTS)),
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


def load_runs():
    if not COMPLETION.is_file():
        raise SystemExit("E1 evaluation is incomplete")
    completion = json.loads(COMPLETION.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("sealed_test_read") is not True:
        raise SystemExit("invalid E1 completion record")
    manifest_hash = sha256_file(MANIFEST)
    if completion.get("manifest_sha256") != manifest_hash:
        raise ValueError("E1 completion manifest hash mismatch")
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    expected_ids = [str(item["scenario_id"]) for item in scenarios]
    if len(expected_ids) != SCENES or len(set(expected_ids)) != SCENES:
        raise ValueError("E1 manifest must contain 128 unique scenes")

    runs = {}
    hashes = {}
    for method in METHODS:
        for seed in SEEDS:
            path = RESULT_DIR / f"g26_e1_{method}_s{seed}.npy"
            rows = np.load(path, allow_pickle=True)
            if rows.shape != (SCENES, 17):
                raise ValueError(f"invalid result shape: {path}")
            if [str(item) for item in rows[:, 12]] != expected_ids:
                raise ValueError(f"scenario order mismatch: {path}")
            if sum(int(row[6]) + int(row[7]) + int(row[10]) for row in rows) != SCENES * AGENTS:
                raise ValueError(f"terminal accounting mismatch: {path}")
            runs[(method, seed)] = rows
            hashes[f"{method}_s{seed}"] = sha256_file(path)
    return scenarios, runs, hashes, completion


def comparison(runs, baseline, candidate):
    metrics = {
        "full_success": (8, 1.0),
        "agent_success": (6, float(AGENTS)),
        "collision": (7, float(AGENTS)),
        "unresolved": (10, float(AGENTS)),
        "timeout": (11, 1.0),
        "raw_steps": (3, 1.0),
        "interaction_share": (14, 1.0),
        "switches": (15, 1.0),
    }
    output = {}
    for name, (column, scale) in metrics.items():
        base = metric_matrix(runs, baseline, column, scale)
        value = metric_matrix(runs, candidate, column, scale)
        output[name] = effect_summary(
            (value - base).mean(axis=1), include_sign_flip=(name == "full_success")
        )

    base_full = metric_matrix(runs, baseline, 8)
    value_full = metric_matrix(runs, candidate, 8)
    pairs = {}
    for index, seed in enumerate(SEEDS):
        delta = value_full[:, index] - base_full[:, index]
        pairs[str(seed)] = {
            "improved": int(np.sum(delta > 0)),
            "degraded": int(np.sum(delta < 0)),
            "tied": int(np.sum(delta == 0)),
        }
    output["full_success"]["per_repeat_pairs"] = pairs

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


def main():
    scenarios, runs, hashes, completion = load_runs()
    pooled = {
        method: aggregate(np.concatenate([runs[(method, seed)] for seed in SEEDS]))
        for method in METHODS
    }
    comparisons = {
        "nf_switch_minus_5a": comparison(runs, "5a", "nf_switch"),
        "b2_minus_5a": comparison(runs, "5a", "b2"),
        "b2_minus_nf_switch": comparison(runs, "nf_switch", "b2"),
    }
    edge_bands = {}
    edge_counts = np.asarray([int(item["metrics"]["conflict_edge_count"]) for item in scenarios])
    for band, mask in (
        ("0", edge_counts == 0),
        ("1", edge_counts == 1),
        ("2", edge_counts == 2),
        ("3+", edge_counts >= 3),
    ):
        band_output = {"scenes": int(np.sum(mask))}
        for method in METHODS:
            rows = np.concatenate([runs[(method, seed)] for seed in SEEDS])
            repeated_mask = np.tile(mask, len(SEEDS))
            band_output[method] = aggregate(rows[repeated_mask]) if np.any(mask) else None
        edge_bands[band] = band_output

    output = {
        "protocol": {
            "experiment_id": "G26-E1-normalizing-flow-inspired-switch",
            "implementation_label": "normalizing-flow-inspired switching baseline",
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "scene_clusters": SCENES,
            "source_slice": "dense/test original order [256:384]",
            "total_episodes": SCENES * len(SEEDS) * len(METHODS),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "sign_flip_samples": SIGN_FLIP_SAMPLES,
            "random_seed": BOOTSTRAP_SEED,
            "inference_status": "exploratory supplemental; no G25 confirmatory statistic modified",
            "actor_or_router_updated": False,
            "sealed_test_read": True,
        },
        "manifest_sha256": sha256_file(MANIFEST),
        "completion_sha256": sha256_file(COMPLETION),
        "result_sha256": hashes,
        "completion": completion,
        "pooled_descriptive": pooled,
        "comparisons": comparisons,
        "by_conflict_edges_descriptive": edge_bands,
        "interpretation_guardrails": [
            "Compare methods within the same scene and repeat; do not compare this slice directly with G25 first256 absolute rates.",
            "Sign-flip p-values and BCa intervals are exploratory supplemental summaries, not G25 confirmatory tests.",
            "NF is a local literature-inspired baseline, not the original IROS 2024 authors' implementation.",
        ],
    }
    output_path = E1 / "e1_statistics.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"sha256 {sha256_file(output_path)}  {output_path}")


if __name__ == "__main__":
    main()
