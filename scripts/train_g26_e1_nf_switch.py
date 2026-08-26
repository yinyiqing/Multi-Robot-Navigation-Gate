#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from normalizing_flow import RealNVPFlow


BASE = PROJECT_ROOT / "experiments/03_保留专门化/02_论文主线"
G11_A1 = BASE / "11_可部署在线Gate研究/G11_A1_当前协议时序pilot"
G26 = BASE / "26_数量泛化与外部切换基线"

EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026"
)
EXPECTED_COLLECTION_LOG_SHA256 = (
    "8c58e0b15e69db1913a636742ee451d26774ed64293d2bb251de686144c39c51"
)
EXPECTED_A1_SHARD_DIGEST = (
    "7b50f36611629332f89515b8035ce5576c100217a80536b7fe13601ce839fa4e"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the frozen G26-E1 normalizing-flow-inspired switch."
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=BASE / "datasets/fixed_v1/views/g11_a1_gate_v1/train.json.gz",
    )
    parser.add_argument(
        "--a1-shard-dir",
        type=Path,
        default=G11_A1 / "local_data/shards/train",
    )
    parser.add_argument(
        "--a1-shard-audit",
        type=Path,
        default=G11_A1 / "local_data/shard_audit.json",
    )
    parser.add_argument(
        "--collection-log",
        type=Path,
        default=(
            PROJECT_ROOT
            / "logs/archive/diagnostic/g11_a1/collection"
            / "collect_g11_a1_gate_train_20260804_212736.log"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=G26 / "local_data/e1/nf_switch_seed20260821",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--input-dim", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--log-scale-limit", type=float, default=2.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--fit-fraction", type=float, default=0.80)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def shard_tree_digest(shard_dir):
    paths = sorted(Path(shard_dir).glob("*.npz"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(paths)


def load_manifest(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("train manifest must contain scenarios")
    return payload, {str(item["scenario_id"]): item for item in scenarios}


EPISODE_PATTERN = re.compile(
    r"Episode\s+(\d+)\s+complete\s+\|\s+case=([^|]+?)\s+\|.*?"
    r"\|\s+success=(\d+)/(\d+)\s+\|\s+collision=(\d+)/(\d+)\s+\|"
    r"\s+unresolved=(\d+)/(\d+)\s+\|\s+full_success=(\d+)\s+\|"
    r"\s+timeout=(\d+)"
)


def parse_collection_log(path):
    outcomes = {}
    episode_order = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPISODE_PATTERN.search(line)
        if not match:
            continue
        (
            episode,
            scenario_id,
            success,
            agent_count,
            collision,
            _collision_denominator,
            unresolved,
            _unresolved_denominator,
            full_success,
            timeout,
        ) = match.groups()
        scenario_id = scenario_id.strip()
        if scenario_id in outcomes:
            raise ValueError("duplicate scenario in collection log: %s" % scenario_id)
        outcomes[scenario_id] = {
            "episode": int(episode),
            "success": int(success),
            "agent_count": int(agent_count),
            "collision": int(collision),
            "unresolved": int(unresolved),
            "full_success": int(full_success),
            "timeout": int(timeout),
        }
        episode_order.append(scenario_id)
    if not outcomes:
        raise ValueError("collection log contains no episode summaries")
    expected_episodes = list(range(1, len(episode_order) + 1))
    observed_episodes = [outcomes[item]["episode"] for item in episode_order]
    if observed_episodes != expected_episodes:
        raise ValueError("collection log episode numbers are not contiguous")
    return outcomes, episode_order


def stratum_key(scenario):
    view = scenario.get("view", {})
    pool = str(view.get("gate_pool") or view.get("perception_pool") or scenario.get("preset"))
    band = str(view.get("interaction_band") or "unknown")
    return "%s_%s" % (pool, band)


def split_success_scenarios(
    scenarios_by_id,
    outcomes,
    split_seed="20260821",
    fit_fraction=0.80,
):
    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError("fit_fraction must lie in (0, 1)")
    missing = sorted(set(outcomes) - set(scenarios_by_id))
    if missing:
        raise ValueError("collection log contains scenarios outside manifest: %s" % missing[:5])
    success_ids = [
        scenario_id
        for scenario_id, outcome in outcomes.items()
        if int(outcome["full_success"]) == 1
    ]
    grouped = defaultdict(list)
    for scenario_id in success_ids:
        grouped[stratum_key(scenarios_by_id[scenario_id])].append(scenario_id)

    fit_ids = []
    calibration_ids = []
    strata = {}
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda value: hashlib.sha256(
                ("%s:%s" % (split_seed, value)).encode("utf-8")
            ).hexdigest(),
        )
        fit_count = int(len(ordered) * fit_fraction)
        fit_ids.extend(ordered[:fit_count])
        calibration_ids.extend(ordered[fit_count:])
        strata[key] = {
            "success_scenarios": len(ordered),
            "fit_scenarios": fit_count,
            "calibration_scenarios": len(ordered) - fit_count,
        }
    if set(fit_ids) & set(calibration_ids):
        raise ValueError("fit/calibration split overlaps")
    return fit_ids, calibration_ids, strata


def load_states_for_scenarios(shard_dir, scenario_ids, input_dim=24):
    states = []
    groups = []
    per_scenario_frames = {}
    for scenario_id in scenario_ids:
        path = Path(shard_dir) / ("%s.npz" % scenario_id)
        if not path.is_file():
            raise ValueError("missing shard for selected scenario: %s" % scenario_id)
        with np.load(path, allow_pickle=False) as shard:
            values = np.asarray(shard["frame_actor_states"], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != int(input_dim):
                raise ValueError("invalid actor states in %s" % path)
            if str(shard["scenario_id"]) != scenario_id:
                raise ValueError("shard scenario_id mismatch in %s" % path)
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite actor states in %s" % path)
            states.append(values)
            groups.extend([scenario_id] * len(values))
            per_scenario_frames[scenario_id] = int(len(values))
    if not states:
        raise ValueError("at least one selected scenario is required")
    return np.concatenate(states), np.asarray(groups), per_scenario_frames


def scenario_equal_frame_weights(groups):
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("groups must be a non-empty vector")
    counts = Counter(groups.tolist())
    weights = np.asarray([1.0 / counts[item] for item in groups], dtype=np.float64)
    weights *= 1.0 / float(len(counts))
    return (weights / np.mean(weights)).astype(np.float32)


def normalize_fit_calibration(fit_states, calibration_states, std_floor=1e-6):
    mean = np.mean(fit_states, axis=0).astype(np.float32)
    std = np.std(fit_states, axis=0).astype(np.float32)
    floored = std < float(std_floor)
    std[floored] = 1.0
    return (
        ((fit_states - mean) / std).astype(np.float32),
        ((calibration_states - mean) / std).astype(np.float32),
        mean,
        std,
        floored,
    )


def nll_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def model_nll(model, values, device):
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
    return model.negative_log_likelihood(tensor).cpu().numpy()


def train_flow(model, fit_values, weights, args, device):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    fit_tensor = torch.from_numpy(fit_values.astype(np.float32)).to(device)
    weight_tensor = torch.from_numpy(weights.astype(np.float32)).to(device)
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        nll = model.negative_log_likelihood(fit_tensor)
        loss = torch.sum(nll * weight_tensor) / torch.sum(weight_tensor)
        if not torch.isfinite(loss):
            raise ValueError("non-finite flow loss at epoch %i" % epoch)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "weighted_nll": float(loss.detach().cpu().item())})
        print("epoch=%03d weighted_nll=%.6f" % (epoch, history[-1]["weighted_nll"]))
    return history


def verify_invertibility(model, values, device):
    sample = torch.from_numpy(values[: min(len(values), 64)].astype(np.float32)).to(device)
    with torch.no_grad():
        latent, forward_log_det = model(sample)
        restored, inverse_log_det = model.inverse(latent)
    max_error = float(torch.max(torch.abs(restored - sample)).cpu().item())
    log_det_error = float(torch.max(torch.abs(forward_log_det + inverse_log_det)).cpu().item())
    if max_error > 1e-4 or log_det_error > 1e-4:
        raise ValueError(
            "RealNVP inverse check failed: max_error=%.8f log_det_error=%.8f"
            % (max_error, log_det_error)
        )
    return {"max_reconstruction_error": max_error, "max_log_det_error": log_det_error}


def verify_sources(args):
    train_manifest_sha256 = sha256_file(args.train_manifest)
    collection_log_sha256 = sha256_file(args.collection_log)
    shard_digest, shard_count = shard_tree_digest(args.a1_shard_dir)
    with args.a1_shard_audit.open("r", encoding="utf-8") as handle:
        shard_audit = json.load(handle)
    audit_digest = shard_audit.get("train", {}).get("dataset_sha256")
    if train_manifest_sha256 != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise ValueError("train manifest SHA-256 mismatch")
    if collection_log_sha256 != EXPECTED_COLLECTION_LOG_SHA256:
        raise ValueError("collection log SHA-256 mismatch")
    if shard_digest != EXPECTED_A1_SHARD_DIGEST or audit_digest != shard_digest:
        raise ValueError("A1 shard digest mismatch")
    if shard_count != 640:
        raise ValueError("expected 640 A1 train shards, got %i" % shard_count)
    return {
        "train_manifest": project_relative_path(args.train_manifest),
        "train_manifest_sha256": train_manifest_sha256,
        "collection_log": project_relative_path(args.collection_log),
        "collection_log_sha256": collection_log_sha256,
        "a1_shard_dir": project_relative_path(args.a1_shard_dir),
        "a1_shard_digest": shard_digest,
        "a1_shard_count": shard_count,
        "a1_shard_audit": project_relative_path(args.a1_shard_audit),
    }


def main():
    args = parse_args()
    if int(args.epochs) != 50:
        raise ValueError("G26-E1 protocol freezes training at exactly 50 epochs")
    if int(args.input_dim) != 24:
        raise ValueError("G26-E1 protocol freezes the flow input at 24 dimensions")
    if int(args.num_blocks) != 6 or int(args.hidden_dim) != 128:
        raise ValueError("G26-E1 protocol freezes a 6-block RealNVP with hidden_dim=128")
    if float(args.threshold_quantile) != 0.95:
        raise ValueError("G26-E1 protocol freezes the threshold at calibration p95")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))

    output_dir = args.output_dir
    checkpoint_path = output_dir / "checkpoint.pt"
    summary_path = output_dir / "summary.json"
    if not args.overwrite and (checkpoint_path.exists() or summary_path.exists()):
        raise SystemExit("E1 output already exists; use --overwrite only before test read")

    sources = verify_sources(args)
    manifest_payload, scenarios_by_id = load_manifest(args.train_manifest)
    outcomes, episode_order = parse_collection_log(args.collection_log)
    if len(outcomes) != 640:
        raise ValueError("expected 640 A1 collection outcomes, got %i" % len(outcomes))
    if set(outcomes) != set(scenarios_by_id):
        raise ValueError("collection log and train manifest scenario sets differ")
    if any(item["agent_count"] != 5 for item in outcomes.values()):
        raise ValueError("G26-E1 source must be the 5-robot A1 collection")

    fit_ids, calibration_ids, strata = split_success_scenarios(
        scenarios_by_id,
        outcomes,
        split_seed=str(args.seed),
        fit_fraction=float(args.fit_fraction),
    )
    if len(fit_ids) != 352 or len(calibration_ids) != 90:
        raise ValueError("unexpected E1 split size")

    fit_states, fit_groups, fit_frame_counts = load_states_for_scenarios(
        args.a1_shard_dir,
        fit_ids,
        input_dim=args.input_dim,
    )
    calibration_states, calibration_groups, calibration_frame_counts = (
        load_states_for_scenarios(
            args.a1_shard_dir,
            calibration_ids,
            input_dim=args.input_dim,
        )
    )
    if len(fit_states) != 15011 or len(calibration_states) != 4059:
        raise ValueError("unexpected E1 fit/calibration frame count")

    fit_values, calibration_values, feature_mean, feature_std, std_floored = (
        normalize_fit_calibration(fit_states, calibration_states)
    )
    fit_weights = scenario_equal_frame_weights(fit_groups)

    reset_random_seed(args.seed)
    device = torch.device("cpu")
    model_config = {
        "input_dim": int(args.input_dim),
        "num_blocks": int(args.num_blocks),
        "hidden_dim": int(args.hidden_dim),
        "log_scale_limit": float(args.log_scale_limit),
    }
    model = RealNVPFlow(**model_config).to(device)
    history = train_flow(model, fit_values, fit_weights, args, device)
    invertibility = verify_invertibility(model, fit_values, device)

    fit_nll = model_nll(model, fit_values, device)
    calibration_nll = model_nll(model, calibration_values, device)
    if not np.all(np.isfinite(fit_nll)) or not np.all(np.isfinite(calibration_nll)):
        raise ValueError("flow produced non-finite NLL values")
    threshold_nll = float(np.quantile(calibration_nll, args.threshold_quantile))
    calibration_nll_sorted = np.sort(calibration_nll.astype(np.float32))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "experiment_id": "G26-E1-normalizing-flow-inspired-switch",
        "model_id": "RealNVP-24D-6x128",
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "threshold_nll": threshold_nll,
        "threshold_quantile": float(args.threshold_quantile),
        "calibration_nll_sorted": calibration_nll_sorted,
        "strict_reproduction": False,
        "actor_or_b2_updated": False,
        "training_seed": int(args.seed),
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    summary = {
        "protocol": {
            "experiment_id": "G26-E1-normalizing-flow-inspired-switch",
            "implementation_label": "normalizing-flow-inspired switching baseline",
            "strict_reproduction": False,
            "seed": int(args.seed),
            "device": "cpu",
            "epochs": int(args.epochs),
            "optimizer": "Adam",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "fit_fraction_by_stratum": float(args.fit_fraction),
            "threshold_rule": "calibration NLL 95th percentile",
            "test_data_read": False,
            "actor_or_b2_updated": False,
        },
        "sources": sources,
        "manifest_dataset_id": manifest_payload.get("dataset_id"),
        "source_collection": {
            "episodes": len(episode_order),
            "full_success_scenarios": int(
                sum(item["full_success"] for item in outcomes.values())
            ),
            "collision_scenarios": int(
                sum(1 for item in outcomes.values() if item["collision"] > 0)
            ),
            "timeout_scenarios": int(sum(item["timeout"] for item in outcomes.values())),
        },
        "split": {
            "strata": strata,
            "fit_scenarios": len(fit_ids),
            "calibration_scenarios": len(calibration_ids),
            "fit_frames": int(len(fit_states)),
            "calibration_frames": int(len(calibration_states)),
            "fit_frame_count_range": [
                int(min(fit_frame_counts.values())),
                int(max(fit_frame_counts.values())),
            ],
            "calibration_frame_count_range": [
                int(min(calibration_frame_counts.values())),
                int(max(calibration_frame_counts.values())),
            ],
        },
        "normalization": {
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "std_floor": 1e-6,
            "floored_dimensions": np.flatnonzero(std_floored).astype(int).tolist(),
        },
        "model_config": model_config,
        "training_history": history,
        "fit_nll": nll_summary(fit_nll),
        "calibration_nll": nll_summary(calibration_nll),
        "threshold_nll": threshold_nll,
        "invertibility_check": invertibility,
        "checkpoint": project_relative_path(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "claim_boundary": (
            "This is a local normalizing-flow-inspired switch under the frozen "
            "G26-E1 protocol; it is not the original authors' implementation."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_sha256 = sha256_file(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("sha256 %s  %s" % (checkpoint_sha256, checkpoint_path))
    print("sha256 %s  %s" % (summary_sha256, summary_path))


if __name__ == "__main__":
    main()
