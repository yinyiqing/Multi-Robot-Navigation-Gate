import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from training_utils import episode_train_iterations, replay_done


class TrainingUtilsTests(unittest.TestCase):
    def test_timeout_is_terminal_for_replay(self):
        self.assertEqual(replay_done(True, False), 1)
        self.assertEqual(replay_done(False, True), 1)
        self.assertEqual(replay_done(False, False), 0)

    def test_updates_scale_with_collective_steps(self):
        self.assertEqual(episode_train_iterations(500, 5), 100)
        self.assertEqual(episode_train_iterations(501, 5), 101)
        self.assertEqual(episode_train_iterations(0, 5), 1)

    def test_update_scaling_rejects_invalid_agent_count(self):
        with self.assertRaises(ValueError):
            episode_train_iterations(10, 0)


if __name__ == "__main__":
    unittest.main()
