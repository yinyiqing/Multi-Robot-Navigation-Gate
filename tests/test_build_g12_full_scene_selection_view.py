import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_g12_full_scene_selection_view import (
    annotate,
    select_ranked,
    topology_name,
)


def test_topology_name_groups_all_multi_edge_cases():
    assert topology_name({"metrics": {"conflict_edge_count": 0}}) == "zero"
    assert topology_name({"metrics": {"conflict_edge_count": 1}}) == "edge1"
    assert topology_name({"metrics": {"conflict_edge_count": 2}}) == "multi"
    assert topology_name({"metrics": {"conflict_edge_count": 7}}) == "multi"


def test_topology_name_rejects_negative_edge_count():
    try:
        topology_name({"metrics": {"conflict_edge_count": -1}})
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("negative edge count was accepted")


def test_ranked_selection_is_deterministic():
    scenarios = [{"scenario_id": str(index)} for index in range(20)]
    first = select_ranked(scenarios, 8, 20260806, "dense", "multi")
    second = select_ranked(
        list(reversed(scenarios)), 8, 20260806, "dense", "multi"
    )
    assert first == second
    assert len({item["scenario_id"] for item in first}) == 8


def test_ranked_selection_rejects_short_stratum():
    try:
        select_ranked([{"scenario_id": "one"}], 2, 20260806, "dense", "zero")
    except ValueError as error:
        assert "need 2" in str(error)
    else:
        raise AssertionError("short stratum was accepted")


def test_annotation_preserves_source_and_navigation_split():
    source = {
        "scenario_id": "example",
        "split": "validation",
        "view": {"source": "fixed-v1"},
    }
    result = annotate(source, "standard", "edge1")
    assert result["navigation_split"] == "validation"
    assert result["view"]["capacity_protocol"] == "g12-full-scene-selection-v1"
    assert result["view"]["g12_stratum"] == "standard_edge1"
    assert source["view"] == {"source": "fixed-v1"}
