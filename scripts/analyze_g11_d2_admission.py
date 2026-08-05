#!/usr/bin/env python3
import gzip
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究"
    / "G11_D_Gate复核与独立准入"
)
MANIFEST = (
    PROJECT_ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
    / "g11_d2_admission_v1/validation.json.gz"
)
POLICIES = ("5a", "rule", "old_g2a", "a1", "b2", "oracle", "epoch16")
SEED = 20260809
EXPECTED_MANIFEST_SHA256 = (
    "6250b941f127d550641a621d4253e17ea0770ff3c0cb94e6254e1f26b9f4978a"
)
ACTIVE_LOG_DIR = PROJECT_ROOT / "logs/active/gate-g11-d2-admission"
ARCHIVE_LOG_DIR = PROJECT_ROOT / "logs/archive/validation/g11_d2"
LEGACY_LOG_DIR = RUN_DIR / "local_data/logs"
LEGACY_RUNNER_LOG = RUN_DIR / "local_data/admission_runner.log"


def aggregate(rows):
    episodes = len(rows)
    if episodes < 1:
        raise ValueError("cannot aggregate an empty D2 result")
    agents = episodes * 5
    success = sum(int(row[6]) for row in rows)
    collision = sum(int(row[7]) for row in rows)
    unresolved = sum(int(row[10]) for row in rows)
    if success + collision + unresolved != agents:
        raise ValueError("G11-D2 outcome accounting mismatch")
    return {
        "episodes": episodes,
        "agent_success_rate": success / agents,
        "collision_rate": collision / agents,
        "unresolved_rate": unresolved / agents,
        "full_success_rate": sum(int(row[8]) for row in rows) / episodes,
        "timeout_episode_rate": sum(int(row[11]) for row in rows) / episodes,
        "mean_episode_steps": float(np.mean([int(row[3]) for row in rows])),
        "mean_interaction_action_share": float(
            np.mean([float(row[14]) for row in rows])
        ),
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
        raise ValueError("paired G11-D2 runs do not contain the same scenarios")
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


def subset(rows, ids):
    return np.asarray([row for row in rows if str(row[12]) in ids], dtype=object)


def load_rows(policy):
    path = RUN_DIR / (
        "local_data/results/g11_d2_%s_r1_s%d.npy" % (policy, SEED)
    )
    rows = np.load(path, allow_pickle=True)
    if rows.shape != (200, 17):
        raise ValueError("invalid G11-D2 result shape: %s %s" % (path, rows.shape))
    return rows


def archive_completed_logs():
    if not ACTIVE_LOG_DIR.is_dir():
        return []
    ARCHIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    moved = []
    for source in sorted(ACTIVE_LOG_DIR.iterdir()):
        target = ARCHIVE_LOG_DIR / source.name
        if target.exists() or target.is_symlink():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            target = ARCHIVE_LOG_DIR / (
                "%s_archived_%s%s" % (source.stem, stamp, source.suffix)
            )
        shutil.move(str(source), str(target))
        moved.append(target)
    ACTIVE_LOG_DIR.rmdir()

    LEGACY_RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
    if LEGACY_RUNNER_LOG.exists() or LEGACY_RUNNER_LOG.is_symlink():
        LEGACY_RUNNER_LOG.unlink()
    runner_archive = ARCHIVE_LOG_DIR / "admission_runner.log"
    if runner_archive.exists():
        LEGACY_RUNNER_LOG.symlink_to(
            os.path.relpath(runner_archive, LEGACY_RUNNER_LOG.parent)
        )
    if LEGACY_LOG_DIR.exists() or LEGACY_LOG_DIR.is_symlink():
        if LEGACY_LOG_DIR.is_symlink():
            LEGACY_LOG_DIR.unlink()
        elif LEGACY_LOG_DIR.is_dir() and not any(LEGACY_LOG_DIR.iterdir()):
            LEGACY_LOG_DIR.rmdir()
    if not LEGACY_LOG_DIR.exists():
        LEGACY_LOG_DIR.symlink_to(
            os.path.relpath(ARCHIVE_LOG_DIR, LEGACY_LOG_DIR.parent),
            target_is_directory=True,
        )
    return moved


def main():
    import hashlib

    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("G11-D2 manifest hash mismatch")
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
        scenarios = json.load(handle)["scenarios"]
    strata = {
        item["scenario_id"]: item["view"]["g11_d2_stratum"] for item in scenarios
    }
    expected_ids = set(strata)
    runs = {policy: load_rows(policy) for policy in POLICIES}
    for policy, rows in runs.items():
        if set(str(row[12]) for row in rows) != expected_ids:
            raise ValueError("G11-D2 result/manifest mismatch for %s" % policy)

    stratum_ids = {
        name: {key for key, value in strata.items() if value == name}
        for name in sorted(set(strata.values()))
    }
    topology_ids = {
        "zero": {key for key, value in strata.items() if value.endswith("_zero")},
        "edge1": {key for key, value in strata.items() if value.endswith("_edge1")},
    }
    overall = {policy: aggregate(rows) for policy, rows in runs.items()}
    by_topology = {
        topology: {
            policy: aggregate(subset(rows, ids)) for policy, rows in runs.items()
        }
        for topology, ids in topology_ids.items()
    }
    by_stratum = {
        name: {
            policy: aggregate(subset(rows, ids)) for policy, rows in runs.items()
        }
        for name, ids in stratum_ids.items()
    }
    paired_results = {
        "b2_vs_5a": paired(runs["b2"], runs["5a"]),
        "b2_vs_a1": paired(runs["b2"], runs["a1"]),
        "b2_vs_rule": paired(runs["b2"], runs["rule"]),
        "b2_vs_old_g2a": paired(runs["b2"], runs["old_g2a"]),
        "oracle_vs_5a": paired(runs["oracle"], runs["5a"]),
        "b2_vs_5a_edge1": paired(
            subset(runs["b2"], topology_ids["edge1"]),
            subset(runs["5a"], topology_ids["edge1"]),
        ),
    }
    denominator = (
        overall["oracle"]["full_success_rate"]
        - overall["5a"]["full_success_rate"]
    )
    oracle_recovery = (
        (overall["b2"]["full_success_rate"] - overall["5a"]["full_success_rate"])
        / denominator
        if denominator > 0.0
        else None
    )
    edge_pair = paired_results["b2_vs_5a_edge1"]
    rule_pair = paired_results["b2_vs_rule"]
    old_gate_pair = paired_results["b2_vs_old_g2a"]
    navigation_checks = {
        "zero_full_success_drop_at_most_0.03": (
            by_topology["zero"]["b2"]["full_success_rate"]
            >= by_topology["zero"]["5a"]["full_success_rate"] - 0.03
        ),
        "edge1_improvements_exceed_degradations": (
            edge_pair["full_success_improved"] > edge_pair["full_success_degraded"]
        ),
        "edge1_mcnemar_p_below_0.05": edge_pair["mcnemar_exact_p"] < 0.05,
        "overall_timeout_increase_at_most_0.02": (
            overall["b2"]["timeout_episode_rate"]
            <= overall["5a"]["timeout_episode_rate"] + 0.02
        ),
        "b2_exceeds_rule_with_more_improvements": (
            overall["b2"]["full_success_rate"]
            > overall["rule"]["full_success_rate"]
            and rule_pair["full_success_improved"]
            > rule_pair["full_success_degraded"]
        ),
        "b2_vs_rule_mcnemar_p_below_0.05": rule_pair["mcnemar_exact_p"] < 0.05,
        "b2_exceeds_old_g2a_with_more_improvements": (
            overall["b2"]["full_success_rate"]
            > overall["old_g2a"]["full_success_rate"]
            and old_gate_pair["full_success_improved"]
            > old_gate_pair["full_success_degraded"]
        ),
        "absolute_or_oracle_recovery_target": (
            overall["b2"]["full_success_rate"] >= 0.45
            or (oracle_recovery is not None and oracle_recovery >= 0.60)
        ),
    }
    step_ratio = (
        overall["b2"]["mean_episode_steps"] / overall["5a"]["mean_episode_steps"]
    )
    efficiency_checks = {
        "mean_steps_at_most_2x_5a": step_ratio <= 2.0,
        "overall_interaction_share_at_most_0.75": (
            overall["b2"]["mean_interaction_action_share"] <= 0.75
        ),
        "zero_timeout_increase_at_most_0.02": (
            by_topology["zero"]["b2"]["timeout_episode_rate"]
            <= by_topology["zero"]["5a"]["timeout_episode_rate"] + 0.02
        ),
    }
    summary = {
        "protocol": {
            "experiment_id": "G11-D2",
            "manifest": str(MANIFEST.relative_to(PROJECT_ROOT)),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "episodes_per_run": 200,
            "seed": SEED,
            "policies": list(POLICIES),
            "sealed_test_read": False,
        },
        "overall": overall,
        "topology": by_topology,
        "strata": by_stratum,
        "paired": paired_results,
        "oracle_benefit_recovery": oracle_recovery,
        "b2_mean_steps_ratio_vs_5a": step_ratio,
        "navigation_checks": navigation_checks,
        "navigation_admission_pass": bool(all(navigation_checks.values())),
        "efficiency_checks": efficiency_checks,
        "efficiency_admission_pass": bool(all(efficiency_checks.values())),
        "admission_pass": bool(
            all(navigation_checks.values()) and all(efficiency_checks.values())
        ),
    }
    output = RUN_DIR / "local_data/d2_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    moved = archive_completed_logs()
    if moved:
        print("Archived %d G11-D2 logs to %s" % (len(moved), ARCHIVE_LOG_DIR))


if __name__ == "__main__":
    main()
