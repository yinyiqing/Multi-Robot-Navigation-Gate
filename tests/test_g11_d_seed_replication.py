import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_g11_d_seed_replication import metric_stats, seed_record


def test_metric_stats_uses_population_std():
    result = metric_stats([{"f1": 1.0}, {"f1": 3.0}], "f1")
    assert result == {"mean": 2.0, "population_std": 1.0, "min": 1.0, "max": 3.0}


def test_seed_record_applies_all_registered_checks(tmp_path):
    summary = tmp_path / "summary.json"
    checkpoint = tmp_path / "best.pt"
    summary.write_text(json.dumps({"ok": True}), encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "protocol": {"seed": 7},
        "result": {
            "best_epoch": 2,
            "threshold": 0.4,
            "validation": {
                "f1": 0.82,
                "average_precision": 0.9,
                "fpr": 0.2,
                "weak_fpr": 0.3,
                "events": {
                    "positive_interval_iou": 0.71,
                    "event_recall": 0.95,
                    "total_switches": 12,
                },
            },
        },
    }
    thresholds = {
        "fpr_cap": 0.21,
        "weak_fpr_cap": 0.31,
        "min_f1": 0.81,
        "min_positive_interval_iou": 0.70,
    }
    record = seed_record(payload, summary, checkpoint, thresholds)
    assert record["pass"] is True
    assert all(record["checks"].values())

    payload["result"]["validation"]["weak_fpr"] = 0.32
    failed = seed_record(payload, summary, checkpoint, thresholds)
    assert failed["pass"] is False
    assert failed["checks"]["weak_fpr_cap"] is False
