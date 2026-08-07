import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "start_training_g12_r2_s4_n5.sh"
WORKER = ROOT / "scripts" / "run_training_g12_r2_s4_n5_worker.sh"
EXPERIMENT = ROOT / "scripts" / "experiment.sh"


class G12R2S4ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = START.read_text(encoding="utf-8")
        cls.worker = WORKER.read_text(encoding="utf-8")
        cls.experiment = EXPERIMENT.read_text(encoding="utf-8")

    def test_uses_frozen_n5_manifests(self):
        self.assertIn("g12_r2_curriculum_v1/n5", self.start)
        self.assertIn(
            "82f990dab54331ef55d3818fbe39b31fe00480dd99696987a5b85c5e2581ac1e",
            self.start,
        )
        self.assertIn(
            "e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7",
            self.start,
        )

    def test_full_warm_start_from_s3_best(self):
        model = "capacity_wide_r2_s3_broad_n3_seed20260815_best"
        self.assertIn(model, self.start)
        self.assertIn(model, self.worker)
        self.assertIn("DRL_MULTI_LOAD_ACTOR_ONLY=0", self.worker)
        self.assertIn("DRL_MULTI_REQUIRE_MODEL_LOAD=1", self.worker)

    def test_five_robot_pilot_budget_is_frozen(self):
        self.assertIn("DRL_MULTI_NUM_AGENTS=5", self.worker)
        self.assertIn("DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=10000", self.worker)
        self.assertIn("DRL_MULTI_EVAL_EPISODES=120", self.worker)
        self.assertIn("DRL_MULTI_MAX_EPOCHS=2", self.worker)

    def test_uses_gpu_and_repository_logs(self):
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", self.worker)
        self.assertIn("logs/active/capacity-wide-g12-r2/s4-n5", self.start)
        self.assertIn("actor-g12-r2-s4-n5", self.experiment)


if __name__ == "__main__":
    unittest.main()
