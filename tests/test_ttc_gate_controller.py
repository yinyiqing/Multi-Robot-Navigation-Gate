import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from ttc_gate_controller import RiskThresholds, candidate_matches_risk


class TtcRiskRuleTest(unittest.TestCase):
    def setUp(self):
        self.thresholds = RiskThresholds(0.5, 2, 3.0, 0.1, 3.0, 1.0)
        self.candidate = SimpleNamespace(
            shape_probability=0.4,
            smoothed_shape_probability=0.7,
            age=3,
            local_center=np.asarray([1.5, 0.0]),
            closing_speed=0.3,
            ttc=2.0,
            closest_approach_distance=0.6,
        )

    def test_candidate_must_satisfy_all_deployable_risk_terms(self):
        self.assertTrue(candidate_matches_risk(self.candidate, self.thresholds))
        fields = {
            "smoothed_shape_probability": 0.3,
            "age": 1,
            "local_center": np.asarray([3.5, 0.0]),
            "closing_speed": 0.05,
            "ttc": 3.5,
            "closest_approach_distance": 1.2,
        }
        for name, value in fields.items():
            changed = vars(self.candidate).copy()
            changed[name] = value
            if name == "smoothed_shape_probability":
                changed["shape_probability"] = 0.3
            self.assertFalse(
                candidate_matches_risk(SimpleNamespace(**changed), self.thresholds),
                msg=name,
            )

    def test_raw_or_smoothed_shape_probability_can_support_candidate(self):
        changed = vars(self.candidate).copy()
        changed["shape_probability"] = 0.8
        changed["smoothed_shape_probability"] = 0.2
        self.assertTrue(
            candidate_matches_risk(SimpleNamespace(**changed), self.thresholds)
        )


if __name__ == "__main__":
    unittest.main()
