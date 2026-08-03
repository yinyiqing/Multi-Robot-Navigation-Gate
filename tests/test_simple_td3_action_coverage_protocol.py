import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START_A = ROOT / "scripts" / "start_training_dense_simple_td3_hparam_a.sh"
START_B = ROOT / "scripts" / "start_training_dense_simple_td3_hparam_b.sh"
START_C = ROOT / "scripts" / "start_training_dense_simple_td3_hparam_c.sh"
START_D = ROOT / "scripts" / "start_training_dense_simple_td3_hparam_d.sh"
START_D2 = ROOT / "scripts" / "start_training_dense_simple_td3_hparam_d2.sh"
START_D2B = ROOT / "scripts" / "start_training_dense_simple_td3_hparam_d2b.sh"
START_EDGE1_V2 = ROOT / "scripts" / "start_training_full_actor_edge1_simple_critic_v2.sh"


class SimpleTd3ActionCoverageProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start_a = START_A.read_text(encoding="utf-8")
        cls.start_b = START_B.read_text(encoding="utf-8")
        cls.start_c = START_C.read_text(encoding="utf-8")
        cls.start_d = START_D.read_text(encoding="utf-8")
        cls.start_d2 = START_D2.read_text(encoding="utf-8")
        cls.start_d2b = START_D2B.read_text(encoding="utf-8")
        cls.start_edge1_v2 = START_EDGE1_V2.read_text(encoding="utf-8")

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

    def test_experiment_d_covers_the_entire_critic_warmup(self):
        self.assertIn("DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS", self.start_d)
        self.assertIn("21000", self.start_d)
        self.assertIn("start_training_dense_simple_td3_hparam_b.sh", self.start_d)

    def test_experiment_d2_uses_controlled_single_ego_exploration(self):
        self.assertIn("DRL_MULTI_RANDOM_LINEAR_EXPLORATION_SCOPE=single_ego", self.start_d2)
        self.assertIn("DRL_MULTI_CRITIC_WARMUP_EXPL_NOISE=0.0", self.start_d2)
        self.assertIn("DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS", self.start_d2)
        self.assertIn("DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=1000000000", self.start_d2)

    def test_experiment_d2_uses_ego_motion_critic_without_extra_objectives(self):
        self.assertIn("DRL_MULTI_USE_LOCAL_CRITIC=1", self.start_d2)
        self.assertIn("DRL_MULTI_LOCAL_CRITIC_CONTEXT_MODE=ego_motion", self.start_d2)
        self.assertIn("DRL_MULTI_CRITIC_INTERACTION_FRACTION=0.0", self.start_d2)
        self.assertIn("start_training_dense_simple_td3_hparam_b.sh", self.start_d2)

    def test_experiment_d2b_stores_only_controlled_ego_transitions(self):
        self.assertIn("DRL_MULTI_CONTROLLED_EGO_REPLAY_ONLY=1", self.start_d2b)
        self.assertIn("DRL_MULTI_RANDOM_LINEAR_EXPLORATION_SCOPE=single_ego", self.start_d2b)
        self.assertIn("DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=1000000000", self.start_d2b)

    def test_experiment_d2b_changes_only_replay_protocol_and_budget(self):
        self.assertIn("DRL_MULTI_MIN_REPLAY_SIZE", self.start_d2b)
        self.assertIn("3000", self.start_d2b)
        self.assertIn("DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES", self.start_d2b)
        self.assertIn("12000", self.start_d2b)
        self.assertIn("DRL_MULTI_CRITIC_LR", self.start_d2b)
        self.assertIn("0.00002", self.start_d2b)
        self.assertIn("start_training_dense_simple_td3_hparam_b.sh", self.start_d2b)

    def test_edge1_v2_trains_one_full_actor_without_pair_routing(self):
        self.assertIn("edge1_full_horizon_v1", self.start_edge1_v2)
        self.assertIn(
            "DRL_MULTI_RANDOM_LINEAR_EXPLORATION_SCOPE=single_ego",
            self.start_edge1_v2,
        )
        self.assertIn("DRL_MULTI_CONTROLLED_EGO_REPLAY_ONLY=1", self.start_edge1_v2)
        self.assertNotIn("conflict_pair", self.start_edge1_v2)
        self.assertNotIn("GATE", self.start_edge1_v2.upper())

    def test_edge1_v2_keeps_actor_frozen_and_removes_reward_stack(self):
        self.assertIn(
            "DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=1000000000",
            self.start_edge1_v2,
        )
        self.assertIn("DRL_MULTI_USE_DYNAMIC_REWARD=0", self.start_edge1_v2)
        self.assertIn("DRL_MULTI_CRITIC_LR=0.00002", self.start_edge1_v2)


if __name__ == "__main__":
    unittest.main()
