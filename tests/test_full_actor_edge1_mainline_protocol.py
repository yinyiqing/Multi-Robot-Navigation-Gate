import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "start_training_full_actor_edge1_from_5a.sh"


class FullActorEdge1MainlineProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = START.read_text(encoding="utf-8")

    def test_uses_corrected_edge1_scenarios_and_5a_launcher(self):
        self.assertIn("edge1_full_horizon_v1", self.script)
        self.assertIn("start_training_dense_simple_td3_hparam_a.sh", self.script)

    def test_trains_one_complete_shared_actor_with_original_critic(self):
        self.assertIn("DRL_MULTI_USE_LOCAL_CRITIC=0", self.script)
        self.assertIn("DRL_MULTI_USE_DYNAMIC_REWARD=0", self.script)
        self.assertIn("DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=20000", self.script)
        self.assertIn("DRL_MULTI_MAX_EPOCHS=6", self.script)

    def test_disables_controlled_sampling_and_extra_objectives(self):
        self.assertIn("DRL_MULTI_CONTROLLED_EGO_REPLAY_ONLY=0", self.script)
        self.assertIn("DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=0", self.script)
        self.assertIn("DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.0", self.script)
        self.assertNotIn("conflict_pair", self.script)
        self.assertNotIn("oracle", self.script.lower())
        self.assertNotIn("gate", self.script.lower())

    def test_keeps_5a_learning_rates_and_individual_reward(self):
        self.assertIn("DRL_MULTI_ACTOR_LR=0.000002", self.script)
        self.assertIn("DRL_MULTI_CRITIC_LR=0.00002", self.script)
        self.assertIn("DRL_MULTI_PROGRESS_REWARD_WEIGHT=20.0", self.script)
        self.assertIn("DRL_MULTI_TURN_PENALTY_WEIGHT=0.2", self.script)


if __name__ == "__main__":
    unittest.main()
