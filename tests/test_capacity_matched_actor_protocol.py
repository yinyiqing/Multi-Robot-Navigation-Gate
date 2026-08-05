import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "start_training_capacity_matched_actor.sh"
QUEUE = ROOT / "scripts" / "queue_training_capacity_matched_actor.sh"


class CapacityMatchedActorProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = START.read_text(encoding="utf-8")
        cls.queue_script = QUEUE.read_text(encoding="utf-8")

    def test_waits_for_d2_archive(self):
        self.assertIn(".g11_d2_admission.pid", self.script)
        self.assertIn("d2_summary.json", self.script)
        self.assertIn("d2_summary.json", self.queue_script)
        self.assertIn("sleep 60", self.queue_script)

    def test_queue_avoids_an_occupied_gpu(self):
        self.assertIn("--query-compute-apps=pid", self.queue_script)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.queue_script)

    def test_uses_balanced_navigation_train_view(self):
        self.assertIn("g11_a1_gate_v1", self.script)
        self.assertIn(
            "a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026",
            self.script,
        )
        self.assertIn("DRL_MULTI_USE_DYNAMIC_REWARD=0", self.script)
        self.assertIn("DRL_MULTI_USE_LOCAL_CRITIC=0", self.script)

    def test_matches_two_actor_capacity_and_expands_5a(self):
        self.assertIn("DRL_MULTI_ACTOR_HIDDEN_DIM_1=1137", self.script)
        self.assertIn("DRL_MULTI_ACTOR_HIDDEN_DIM_2=855", self.script)
        self.assertIn("DRL_MULTI_ALLOW_ACTOR_WARMSTART_EXPANSION=1", self.script)
        self.assertIn("audit_capacity_matched_actor.py", self.script)

    def test_freezes_pilot_budget_and_degradation_stop(self):
        self.assertIn("DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=20000", self.script)
        self.assertIn("DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=20000", self.script)
        self.assertIn("DRL_MULTI_EVAL_EPISODES=120", self.script)
        self.assertIn("DRL_MULTI_MAX_EPOCHS=4", self.script)
        self.assertIn("DRL_MULTI_EARLY_STOP_PATIENCE=1", self.script)

    def test_keeps_logs_under_repository_active_logs(self):
        self.assertIn(
            "logs/active/capacity-matched-actor-g12-p1", self.script
        )


if __name__ == "__main__":
    unittest.main()
