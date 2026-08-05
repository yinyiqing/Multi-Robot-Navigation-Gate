#!/usr/bin/env python3
import gzip
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
    / "G11_C_50场闭环pilot"
)
MANIFEST = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
    / "g11_c_pilot_v1/validation.json.gz"
)
POLICIES = ("5a", "a1", "b2")
REPEATS = {1: 20260805, 2: 20260806}
ACTIVE_LOG_DIR = PROJECT_ROOT / "local/logs/gate-g11-c-pilot"
ARCHIVE_LOG_DIR = RUN_DIR / "local_data/logs"
LEGACY_RUNNER_LOG = RUN_DIR / "local_data/pilot_runner.log"


def load_rows(policy, repeat, seed):
    path = RUN_DIR / (
        "local_data/results/g11_c_%s_r%d_s%d.npy" % (policy, repeat, seed)
    )
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (50, 17):
        raise ValueError("invalid G11-C result shape: %s %s" % (path, rows.shape))
    return rows


def aggregate(rows):
    episodes = len(rows)
    agents = episodes * 5
    success = sum(int(row[6]) for row in rows)
    collision = sum(int(row[7]) for row in rows)
    unresolved = sum(int(row[10]) for row in rows)
    if success + collision + unresolved != agents:
        raise ValueError("G11-C outcome accounting mismatch")
    return {
        "episodes": episodes,
        "agent_success_rate": success / agents,
        "collision_rate": collision / agents,
        "unresolved_rate": unresolved / agents,
        "full_success_rate": sum(int(row[8]) for row in rows) / episodes,
        "timeout_episode_rate": sum(int(row[11]) for row in rows) / episodes,
        "mean_episode_steps": float(np.mean([int(row[3]) for row in rows])),
        "mean_dense_action_share": float(np.mean([float(row[14]) for row in rows])),
        "mean_gate_switches": float(np.mean([int(row[15]) for row in rows])),
        "mean_gate_probability": float(np.mean([float(row[16]) for row in rows])),
        "counts": {
            "success": success,
            "collision": collision,
            "unresolved": unresolved,
            "full_success": sum(int(row[8]) for row in rows),
            "timeout_episodes": sum(int(row[11]) for row in rows),
        },
    }


def mcnemar_exact(improved, degraded):
    discordant = improved + degraded
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(improved, degraded) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired(candidate, baseline):
    candidate_by_case = {str(row[12]): row for row in candidate}
    baseline_by_case = {str(row[12]): row for row in baseline}
    if set(candidate_by_case) != set(baseline_by_case):
        raise ValueError("paired G11-C runs do not contain the same scenarios")
    improved = sum(
        int(candidate_by_case[key][8]) > int(baseline_by_case[key][8])
        for key in candidate_by_case
    )
    degraded = sum(
        int(candidate_by_case[key][8]) < int(baseline_by_case[key][8])
        for key in candidate_by_case
    )
    return {
        "full_success_improved": improved,
        "full_success_degraded": degraded,
        "full_success_tied": len(candidate_by_case) - improved - degraded,
        "mcnemar_exact_p": mcnemar_exact(improved, degraded),
    }


def archive_completed_logs(
    active_log_dir=ACTIVE_LOG_DIR,
    archive_log_dir=ARCHIVE_LOG_DIR,
    legacy_runner_log=LEGACY_RUNNER_LOG,
):
    active_log_dir = Path(active_log_dir)
    archive_log_dir = Path(archive_log_dir)
    legacy_runner_log = Path(legacy_runner_log)
    if not active_log_dir.is_dir():
        return []

    if archive_log_dir.is_symlink():
        archive_log_dir.unlink()
    archive_log_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for source in sorted(active_log_dir.iterdir()):
        target = archive_log_dir / source.name
        if target.exists() or target.is_symlink():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            target = archive_log_dir / (
                "%s_archived_%s%s" % (source.stem, timestamp, source.suffix)
            )
        shutil.move(str(source), str(target))
        moved.append(target)
    active_log_dir.rmdir()

    legacy_runner_log.parent.mkdir(parents=True, exist_ok=True)
    if legacy_runner_log.exists() or legacy_runner_log.is_symlink():
        legacy_runner_log.unlink()
    runner_archive = archive_log_dir / "pilot_runner.log"
    if runner_archive.exists():
        legacy_runner_log.symlink_to(Path("logs") / runner_archive.name)
    return moved


def main():
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    strata = {
        item["scenario_id"]: item["view"]["g11_c_stratum"] for item in scenarios
    }
    expected_ids = set(strata)
    runs = {}
    for repeat, seed in REPEATS.items():
        runs[repeat] = {}
        for policy in POLICIES:
            rows = load_rows(policy, repeat, seed)
            if set(str(row[12]) for row in rows) != expected_ids:
                raise ValueError("G11-C result/manifest mismatch")
            runs[repeat][policy] = rows

    summary = {
        "protocol": {
            "experiment_id": "G11-C1",
            "manifest": str(MANIFEST.relative_to(PROJECT_ROOT)),
            "episodes_per_run": 50,
            "repeats": REPEATS,
            "sealed_test_read": False,
        },
        "per_repeat": {},
        "combined": {},
        "paired": {},
        "strata": {},
    }
    for repeat in REPEATS:
        summary["per_repeat"][str(repeat)] = {
            policy: aggregate(runs[repeat][policy]) for policy in POLICIES
        }
    combined = {
        policy: np.concatenate([runs[repeat][policy] for repeat in REPEATS])
        for policy in POLICIES
    }
    summary["combined"] = {
        policy: aggregate(rows) for policy, rows in combined.items()
    }
    for repeat in REPEATS:
        summary["paired"][str(repeat)] = {
            "a1_vs_5a": paired(runs[repeat]["a1"], runs[repeat]["5a"]),
            "b2_vs_5a": paired(runs[repeat]["b2"], runs[repeat]["5a"]),
            "b2_vs_a1": paired(runs[repeat]["b2"], runs[repeat]["a1"]),
        }
    for stratum in sorted(set(strata.values())):
        ids = {key for key, value in strata.items() if value == stratum}
        summary["strata"][stratum] = {
            policy: aggregate(
                np.asarray([row for row in rows if str(row[12]) in ids], dtype=object)
            )
            for policy, rows in combined.items()
        }

    output = RUN_DIR / "local_data/summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    moved_logs = archive_completed_logs()
    if moved_logs:
        print("Archived %d G11-C logs to %s" % (len(moved_logs), ARCHIVE_LOG_DIR))


if __name__ == "__main__":
    main()
