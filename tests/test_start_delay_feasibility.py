import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_start_delay_feasibility import find_start_delay_schedule


class StartDelayFeasibilityTests(unittest.TestCase):
    def test_non_conflicting_paths_need_no_delay(self):
        paths = {
            "r1": [(0.0, 0.0), (2.0, 0.0)],
            "r2": [(0.0, 2.0), (2.0, 2.0)],
        }
        result = find_start_delay_schedule(paths, separation=0.5, max_delay=2.0)
        self.assertTrue(result["solved"])
        self.assertEqual(result["max_delay_s"], 0.0)

    def test_crossing_paths_are_solved_by_staggering(self):
        paths = {
            "r1": [(-1.0, 0.0), (1.0, 0.0)],
            "r2": [(0.0, -1.0), (0.0, 1.0)],
        }
        result = find_start_delay_schedule(
            paths, separation=0.5, delay_step=0.5, max_delay=3.0
        )
        self.assertTrue(result["solved"])
        self.assertGreater(result["max_delay_s"], 0.0)

    def test_exact_swap_is_not_mislabeled_as_delay_solvable(self):
        paths = {
            "r1": [(-1.0, 0.0), (1.0, 0.0)],
            "r2": [(1.0, 0.0), (-1.0, 0.0)],
        }
        result = find_start_delay_schedule(
            paths, separation=0.5, delay_step=0.5, max_delay=3.0
        )
        self.assertFalse(result["solved"])


if __name__ == "__main__":
    unittest.main()
