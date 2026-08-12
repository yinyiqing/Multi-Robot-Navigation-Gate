import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_ie2_multi_conflict_views import TRAIN_SCHEDULE, interleave, topology


def test_topology_preserves_multi_conflict_severity():
    assert topology({"metrics": {"conflict_edge_count": 0}}) == "zero"
    assert topology({"metrics": {"conflict_edge_count": 1}}) == "edge1"
    assert topology({"metrics": {"conflict_edge_count": 2}}) == "edge2"
    assert topology({"metrics": {"conflict_edge_count": 5}}) == "edge3plus"


def test_interleave_uses_registered_433_schedule():
    groups = {
        "edge1": [{"id": f"a{i}"} for i in range(8)],
        "edge2": [{"id": f"b{i}"} for i in range(6)],
        "edge3plus": [{"id": f"c{i}"} for i in range(6)],
    }
    result = interleave(groups)
    labels = [item["id"][0] for item in result[:10]]
    expected = {"edge1": "a", "edge2": "b", "edge3plus": "c"}
    assert labels == [expected[group] for group in TRAIN_SCHEDULE]
    assert Counter(item["id"][0] for item in result) == {"a": 8, "b": 6, "c": 6}
