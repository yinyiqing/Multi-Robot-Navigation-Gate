import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_scenario_dataset_quality import (
    edge_bucket,
    severity_bucket,
    standardized_mean_difference,
    total_variation,
)


class ScenarioDatasetQualityTests(unittest.TestCase):
    def test_edge_buckets_preserve_compositional_boundary(self):
        self.assertEqual([edge_bucket(value) for value in range(5)], ["0", "1", "2", "3+", "3+"])

    def test_severity_distinguishes_collision_from_nominal_margin(self):
        self.assertEqual(
            severity_bucket(0.533, 0.534, 0.90),
            "physical_overlap_on_nominal_paths",
        )
        self.assertEqual(
            severity_bucket(0.534, 0.534, 0.90),
            "safety_margin_conflict",
        )
        self.assertEqual(
            severity_bucket(0.90, 0.534, 0.90),
            "no_nominal_conflict",
        )

    def test_distribution_distances_are_zero_for_identical_samples(self):
        self.assertEqual(total_variation({"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}), 0.0)
        self.assertEqual(standardized_mean_difference([1, 2, 3], [1, 2, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
