import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START_A = ROOT / "scripts" / "start_training_dense_simple_td3_hparam_a.sh"
START_B = ROOT / "scripts" / "start_training_dense_simple_td3_hparam_b.sh"
START_C = ROOT / "scripts" / "start_training_dense_simple_td3_hparam_c.sh"


class SimpleTd3ActionCoverageProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start_a = START_A.read_text(encoding="utf-8")
        cls.start_b = START_B.read_text(encoding="utf-8")
        cls.start_c = START_C.read_text(encoding="utf-8")

    def test_random_linear_exploration_defaults_off(self):
        self.assertIn(
            'RANDOM_LINEAR_STEPS="${DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS:-0}"',
            self.start_a,
        )

    def test_experiment_c_changes_action_coverage_only(self):
        self.assertIn("DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS", self.start_c)
        self.assertIn("10000", self.start_c)
        self.assertIn("DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE", self.start_c)
        self.assertIn("0.10", self.start_c)
        self.assertIn("start_training_dense_simple_td3_hparam_b.sh", self.start_c)

    def test_experiment_c_keeps_actor_frozen_for_its_budget(self):
        self.assertIn("DRL_MULTI_MAX_EPOCHS", self.start_c)
        self.assertIn("21000", self.start_b)
        self.assertIn("DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS", self.start_b)


if __name__ == "__main__":
    unittest.main()
