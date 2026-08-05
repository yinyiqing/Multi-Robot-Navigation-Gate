import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_g11_d2_admission_view import annotate, select_ranked


def test_ranked_selection_is_deterministic():
    scenarios = [{"scenario_id": str(index)} for index in range(20)]
    first = select_ranked(scenarios, 8, 123, "dense", "edge1")
    second = select_ranked(list(reversed(scenarios)), 8, 123, "dense", "edge1")
    assert first == second
    assert len({item["scenario_id"] for item in first}) == 8


def test_ranked_selection_rejects_short_stratum():
    try:
        select_ranked([{"scenario_id": "one"}], 2, 123, "dense", "edge1")
    except ValueError as error:
        assert "need 2" in str(error)
    else:
        raise AssertionError("short stratum was accepted")


def test_annotation_preserves_navigation_split():
    source = {
        "scenario_id": "example",
        "split": "validation",
        "view": {"source": "fixed-v1"},
    }
    result = annotate(source, "standard", "zero")
    assert result["navigation_split"] == "validation"
    assert result["view"]["gate_protocol"] == "g11-d2-v1"
    assert result["view"]["g11_d2_stratum"] == "standard_zero"
    assert source["view"] == {"source": "fixed-v1"}
