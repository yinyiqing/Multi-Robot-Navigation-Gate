import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_g12_r3_mixed_view import annotate, balanced_strong_sequence


def test_balanced_strong_sequence_uses_frozen_40_40_20_schedule():
    scenarios = [
        {"scenario_id": "deep", "view": {"interaction_band": "deep"}},
        {"scenario_id": "close", "view": {"interaction_band": "close"}},
        {"scenario_id": "margin", "view": {"interaction_band": "margin"}},
    ]
    selected = balanced_strong_sequence(scenarios, 10)
    counts = Counter(item["view"]["interaction_band"] for item in selected)
    assert counts == {"deep": 4, "close": 4, "margin": 2}


def test_annotation_preserves_source_id_under_unique_schedule_alias():
    source = {
        "scenario_id": "source-id",
        "split": "train",
        "view": {"interaction_band": "deep"},
    }
    result = annotate(source, "strong", 7)
    assert result["scenario_id"] == "g12-r3-00007-source-id"
    assert result["view"]["g12_r3_source_scenario_id"] == "source-id"
    assert result["view"]["g12_r3_stream"] == "strong"
    assert source["scenario_id"] == "source-id"
