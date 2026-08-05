import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_g11_d2_admission import aggregate, mcnemar_exact, paired


def row(case, full, success=5, collision=0, unresolved=0):
    return [
        1,
        1,
        5,
        10,
        5,
        0.0,
        success,
        collision,
        full,
        0.0,
        unresolved,
        0,
        case,
        0,
        0.2,
        2,
        0.3,
    ]


def test_aggregate_accounts_for_all_agents():
    result = aggregate(np.asarray([row("a", 1), row("b", 0, 4, 1)], dtype=object))
    assert result["agent_success_rate"] == 0.9
    assert result["collision_rate"] == 0.1
    assert result["full_success_rate"] == 0.5
    assert result["mean_interaction_action_share"] == 0.2


def test_aggregate_rejects_missing_outcomes():
    try:
        aggregate(np.asarray([row("a", 0, 4, 0, 0)], dtype=object))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid agent accounting was accepted")


def test_paired_is_scenario_aligned():
    baseline = np.asarray([row("a", 0), row("b", 1)], dtype=object)
    candidate = np.asarray([row("b", 0), row("a", 1)], dtype=object)
    result = paired(candidate, baseline)
    assert result["full_success_improved"] == 1
    assert result["full_success_degraded"] == 1
    assert result["full_success_tied"] == 0


def test_mcnemar_exact_known_value():
    assert mcnemar_exact(6, 0) == 0.03125
