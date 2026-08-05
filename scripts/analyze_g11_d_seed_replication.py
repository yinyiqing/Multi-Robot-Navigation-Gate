#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_ROUTE = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
)
DEFAULT_TRAINING_DIR = GATE_ROUTE / "G11_B_student_rollout_v1/local_data/training"
DEFAULT_OUTPUT = (
    GATE_ROUTE
    / "G11_D_Gate复核与独立准入/local_data/seed_replication_summary.json"
)
MAIN_SEED = 20260804
REPLICATION_SEEDS = (20260805, 20260806, 20260807, 20260808)
MAX_F1_DROP = 0.03
MAX_INTERVAL_IOU_DROP = 0.04


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize G11-D1 B2 seed replication.")
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_seed(training_dir, seed):
    summary_path = training_dir / ("seed%d" % seed) / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("missing completed seed summary: %s" % summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    if protocol["experiment_id"] != "G11-B2" or int(protocol["seed"]) != seed:
        raise ValueError("seed summary protocol mismatch: %s" % summary_path)
    if protocol["device"] != "cpu" or bool(protocol["sealed_test_read"]):
        raise ValueError("seed summary violates D1 device/data contract: %s" % summary_path)
    checkpoint = training_dir / ("seed%d" % seed) / "any/T1/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("missing seed checkpoint: %s" % checkpoint)
    return payload, summary_path, checkpoint


def seed_record(payload, summary_path, checkpoint, thresholds):
    validation = payload["result"]["validation"]
    events = validation["events"]
    checks = {
        "overall_fpr_cap": validation["fpr"] <= thresholds["fpr_cap"] + 1e-12,
        "weak_fpr_cap": validation["weak_fpr"] <= thresholds["weak_fpr_cap"] + 1e-12,
        "f1_stability": validation["f1"] >= thresholds["min_f1"] - 1e-12,
        "interval_iou_stability": (
            events["positive_interval_iou"]
            >= thresholds["min_positive_interval_iou"] - 1e-12
        ),
    }
    return {
        "seed": int(payload["protocol"]["seed"]),
        "best_epoch": int(payload["result"]["best_epoch"]),
        "threshold": float(payload["result"]["threshold"]),
        "f1": float(validation["f1"]),
        "average_precision": float(validation["average_precision"]),
        "fpr": float(validation["fpr"]),
        "weak_fpr": float(validation["weak_fpr"]),
        "positive_interval_iou": float(events["positive_interval_iou"]),
        "event_recall": float(events["event_recall"]),
        "switches": int(events["total_switches"]),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def metric_stats(records, name):
    values = [float(record[name]) for record in records]
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def analyze(training_dir):
    main, main_summary, main_checkpoint = load_seed(training_dir, MAIN_SEED)
    reference = main["a1_reference_validation"]
    caps = main["protocol"]["fpr_caps"]
    thresholds = {
        "fpr_cap": float(caps["fpr"]),
        "weak_fpr_cap": float(caps["weak_fpr"]),
        "min_f1": float(reference["f1"]) - MAX_F1_DROP,
        "min_positive_interval_iou": (
            float(reference["events"]["positive_interval_iou"])
            - MAX_INTERVAL_IOU_DROP
        ),
    }
    main_record = seed_record(main, main_summary, main_checkpoint, thresholds)
    replication_records = []
    for seed in REPLICATION_SEEDS:
        payload, summary_path, checkpoint = load_seed(training_dir, seed)
        if payload["protocol"]["manifest_hashes"] != main["protocol"]["manifest_hashes"]:
            raise ValueError("manifest hash changed for seed %d" % seed)
        if payload["protocol"]["frozen_hashes"] != main["protocol"]["frozen_hashes"]:
            raise ValueError("frozen component hash changed for seed %d" % seed)
        replication_records.append(
            seed_record(payload, summary_path, checkpoint, thresholds)
        )
    passed = sum(record["pass"] for record in replication_records)
    return {
        "protocol": {
            "experiment_id": "G11-D1",
            "main_seed": MAIN_SEED,
            "replication_seeds": list(REPLICATION_SEEDS),
            "sealed_test_read": False,
            "main_seed_reselection_allowed": False,
            "required_replication_passes": 3,
            "thresholds": thresholds,
        },
        "main_seed": main_record,
        "replications": replication_records,
        "replication_metric_stats": {
            name: metric_stats(replication_records, name)
            for name in (
                "f1",
                "average_precision",
                "fpr",
                "weak_fpr",
                "positive_interval_iou",
                "event_recall",
                "switches",
                "threshold",
            )
        },
        "passed_replications": passed,
        "admission_pass": bool(passed >= 3),
    }


def main():
    args = parse_args()
    result = analyze(args.training_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
