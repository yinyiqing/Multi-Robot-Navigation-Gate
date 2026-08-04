import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_g11_a1_gate_views import annotate_scenario, select_stratum


class G11A1ViewTest(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        scenarios = [{"scenario_id": str(index)} for index in range(20)]
        first = select_stratum(scenarios, 8, 4, 123)
        second = select_stratum(scenarios, 8, 4, 123)
        self.assertEqual(first, second)
        self.assertFalse(
            {item["scenario_id"] for item in first[0]}
            & {item["scenario_id"] for item in first[1]}
        )

    def test_insufficient_stratum_is_rejected(self):
        with self.assertRaises(ValueError):
            select_stratum([{"scenario_id": "one"}], 1, 1, 1)

    def test_annotation_preserves_navigation_split(self):
        source = {
            "scenario_id": "example",
            "split": "train",
            "view": {"source": "fixed-v1"},
        }
        result = annotate_scenario(source, "validation", "dense", "edge1")
        self.assertEqual(result["navigation_split"], "train")
        self.assertEqual(result["split"], "validation")
        self.assertEqual(result["view"]["gate_pool"], "dense")
        self.assertEqual(result["view"]["gate_topology"], "edge1")
        self.assertEqual(result["view"]["perception_pool"], "dense")
        self.assertEqual(result["view"]["interaction_band"], "interaction")
        self.assertEqual(source["split"], "train")


if __name__ == "__main__":
    unittest.main()
