import gzip
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_g11_e_edge2_views import build_views


def make_scenario(index, edges=2):
    return {
        "scenario_id": "case-%03d" % index,
        "split": "validation_confirmation",
        "metrics": {"conflict_edge_count": edges},
    }


def load_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def test_build_views_preserves_frozen_order_and_partitions(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"dataset_id": "source", "scenarios": [make_scenario(i) for i in range(200)]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "views"
    result = build_views(source, output_dir, pilot_scenarios=50)

    pilot = load_gzip(output_dir / "pilot.json.gz")
    confirmation = load_gzip(output_dir / "confirmation.json.gz")
    assert result["pilot"]["scenarios"] == 50
    assert result["confirmation"]["scenarios"] == 150
    assert [item["scenario_id"] for item in pilot["scenarios"]] == [
        "case-%03d" % i for i in range(50)
    ]
    assert [item["scenario_id"] for item in confirmation["scenarios"]] == [
        "case-%03d" % i for i in range(50, 200)
    ]
    assert not (
        {item["scenario_id"] for item in pilot["scenarios"]}
        & {item["scenario_id"] for item in confirmation["scenarios"]}
    )
    assert all(
        item["view"]["gate_topology"] == "edge2"
        for item in pilot["scenarios"] + confirmation["scenarios"]
    )


def test_build_views_rejects_non_edge2_source(tmp_path):
    scenarios = [make_scenario(i) for i in range(200)]
    scenarios[-1] = make_scenario(199, edges=1)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"scenarios": scenarios}), encoding="utf-8")
    try:
        build_views(source, tmp_path / "views")
    except ValueError as exc:
        assert "non-edge-2" in str(exc)
    else:
        raise AssertionError("non-edge-2 source was accepted")
