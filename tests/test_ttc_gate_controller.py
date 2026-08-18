import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from ttc_gate_controller import (
    RiskThresholds,
    TtcCpaActorSwitcher,
    candidate_matches_risk,
)


class FixedPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)

    def get_action(self, state):
        del state
        return self.action.copy()


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


class TtcSwitcherStateTest(unittest.TestCase):
    def make_switcher(self, hold=3):
        switcher = TtcCpaActorSwitcher.__new__(TtcCpaActorSwitcher)
        switcher.standard_policy = FixedPolicy([0.1, 0.2])
        switcher.interaction_policy = FixedPolicy([0.3, 0.4])
        switcher.enter_thresholds = RiskThresholds(0.5, 2, 2.5, 0.1, 3.0, 0.75)
        switcher.stay_thresholds = RiskThresholds(0.3, 2, 3.0, 0.05, 3.5, 1.0)
        switcher.minimum_hold_steps = hold
        switcher.trackers = {"r1": object()}
        switcher.modes = {"r1": "standard"}
        switcher.mode_steps = {"r1": 0}
        switcher.evaluation_steps = {"r1": 0}
        switcher.last_risks = {"r1": False}
        switcher.last_track_counts = {"r1": 0}
        switcher.switch_count = 0
        switcher.risk_sum = 0.0
        switcher.risk_count = 0
        return switcher

    def candidate(self, **updates):
        values = {
            "shape_probability": 0.7,
            "smoothed_shape_probability": 0.7,
            "age": 3,
            "local_center": np.asarray([1.5, 0.0]),
            "closing_speed": 0.3,
            "ttc": 2.0,
            "closest_approach_distance": 0.6,
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def test_enter_and_hold_before_release(self):
        switcher = self.make_switcher(hold=3)
        mode, risk = switcher._update_mode("r1", [self.candidate()])
        self.assertEqual((mode, risk), ("dense", True))
        safe = self.candidate(local_center=np.asarray([4.0, 0.0]))
        self.assertEqual(switcher._update_mode("r1", [safe])[0], "dense")
        self.assertEqual(switcher._update_mode("r1", [safe])[0], "dense")
        self.assertEqual(switcher._update_mode("r1", [safe])[0], "standard")
        self.assertEqual(switcher.episode_stats()["switches"], 2)

    def test_selected_mode_controls_policy(self):
        switcher = self.make_switcher()
        switcher.modes["r1"] = "dense"
        switcher.evaluation_steps["r1"] = 1
        switcher.evaluation_stride = 2
        action, mode, risk, tracks = switcher.choose_action(
            None, "r1", np.zeros(3), logical_time=0.2
        )
        self.assertEqual(mode, "dense")
        self.assertFalse(risk)
        self.assertEqual(tracks, 0)
        self.assertTrue(np.allclose(action, [0.3, 0.4]))


if __name__ == "__main__":
    unittest.main()
