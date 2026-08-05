import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from rule_gate_controllers import MinLidarActorSwitcher


class FixedPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)

    def get_action(self, state):
        del state
        return self.action.copy()


def make_switcher(hold=3):
    switcher = MinLidarActorSwitcher(
        FixedPolicy([0.1, 0.2]),
        FixedPolicy([0.3, 0.4]),
        lidar_bins=2,
        switch_on_distance=2.0,
        switch_off_distance=2.2,
        minimum_hold_steps=hold,
    )
    switcher.reset(["r1"])
    return switcher


def test_min_lidar_switches_without_object_identity():
    switcher = make_switcher()
    action, mode, distance, _ = switcher.choose_action(None, "r1", [1.9, 4.0, 0.0])
    assert mode == "dense"
    assert distance == 1.9
    assert np.allclose(action, [0.3, 0.4])
    assert switcher.episode_stats()["switches"] == 1


def test_hysteresis_and_minimum_hold_are_enforced():
    switcher = make_switcher(hold=3)
    switcher.choose_action(None, "r1", [1.9, 4.0])
    assert switcher.choose_action(None, "r1", [2.3, 4.0])[1] == "dense"
    assert switcher.choose_action(None, "r1", [2.3, 4.0])[1] == "dense"
    assert switcher.choose_action(None, "r1", [2.3, 4.0])[1] == "standard"
    assert switcher.episode_stats()["switches"] == 2


def test_deadband_keeps_current_mode():
    switcher = make_switcher(hold=0)
    assert switcher.choose_action(None, "r1", [2.1, 4.0])[1] == "standard"
    switcher.choose_action(None, "r1", [1.9, 4.0])
    assert switcher.choose_action(None, "r1", [2.1, 4.0])[1] == "dense"


def test_invalid_lidar_state_is_rejected():
    switcher = make_switcher()
    for state in ([1.0], [float("nan"), 2.0]):
        try:
            switcher.choose_action(None, "r1", state)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid LiDAR state was accepted")
