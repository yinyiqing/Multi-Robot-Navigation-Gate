import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from outcome_utils import resolve_terminal_outcome


class OutcomeUtilsTests(unittest.TestCase):
    def test_collision_wins_when_target_and_collision_are_simultaneous(self):
        self.assertEqual(
            resolve_terminal_outcome(False, False, True, True),
            (False, True),
        )

    def test_collision_cannot_later_become_success(self):
        self.assertEqual(
            resolve_terminal_outcome(False, True, True, False),
            (False, True),
        )

    def test_success_remains_success_without_a_collision(self):
        self.assertEqual(
            resolve_terminal_outcome(True, False, False, False),
            (True, False),
        )

    def test_unresolved_remains_unresolved(self):
        self.assertEqual(
            resolve_terminal_outcome(False, False, False, False),
            (False, False),
        )


if __name__ == "__main__":
    unittest.main()
