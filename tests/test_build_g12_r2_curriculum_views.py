import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_g12_r2_curriculum_views import derive_scenario, filtered_conflict_metrics


def source_scenario():
    return {
        "scenario_id": "example",
        "split": "validation",
        "agents": {
            "r1": {"start": [0, 0], "goal": [1, 0], "heading": 0},
            "r2": {"start": [1, 0], "goal": [0, 0], "heading": 0},
            "r3": {"start": [0, 1], "goal": [1, 1], "heading": 0},
            "r4": {"start": [1, 1], "goal": [0, 1], "heading": 0},
            "r5": {"start": [2, 2], "goal": [3, 3], "heading": 0},
        },
        "metrics": {
            "conflict_edge_count": 2,
            "conflict_edges": [
                {"agents": ["r1", "r2"], "time_s": 1.0},
                {"agents": ["r3", "r4"], "time_s": 1.2},
            ],
        },
        "view": {"source": "fixed-v1"},
    }


def test_filtered_conflict_metrics_uses_only_retained_agents():
    metrics = filtered_conflict_metrics(source_scenario()["metrics"], ["r1", "r2"])
    assert metrics["conflict_edge_count"] == 1
    assert metrics["max_conflict_degree"] == 1
    assert metrics["interaction_density"] == 1.0
    assert metrics["simultaneous_conflict_count"] == 1


def test_filtered_conflict_metrics_handles_one_agent():
    metrics = filtered_conflict_metrics(source_scenario()["metrics"], ["r1"])
    assert metrics["conflict_edge_count"] == 0
    assert metrics["interaction_density"] == 0.0
    assert metrics["earliest_conflict_time_s"] is None


def test_derive_scenario_does_not_mutate_source():
    source = source_scenario()
    result = derive_scenario(source, "train", 2)
    assert list(result["agents"]) == ["r1", "r2"]
    assert result["split"] == "train"
    assert result["navigation_split"] == "validation"
    assert result["view"]["capacity_stage_agents"] == 2
    assert len(source["agents"]) == 5
    assert source["metrics"]["conflict_edge_count"] == 2
