import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "start_training_g12_r2_s3_n3.sh"
WORKER = ROOT / "scripts" / "run_training_g12_r2_s3_n3_worker.sh"
EXPERIMENT = ROOT / "scripts" / "experiment.sh"


class G12R2S3ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = START.read_text(encoding="utf-8")
        cls.worker = WORKER.read_text(encoding="utf-8")
        cls.experiment = EXPERIMENT.read_text(encoding="utf-8")

    def test_uses_frozen_n3_manifests(self):
        self.assertIn("g12_r2_curriculum_v1/n3", self.start)
        self.assertIn(
            "b6ff22964a8b1795a783f8af9360c123fae44b4b44a86de63e76a57b4a0b4422",
            self.start,
        )
        self.assertIn(
            "f4b7d46fc488eb588007aa7ba72791545e750e691399da82c65d5cdf9f5938cc",
            self.start,
        )

    def test_full_warm_start_from_s2_best(self):
        model = "capacity_wide_r2_s2_broad_n2_seed20260814_best"
        self.assertIn(model, self.start)
        self.assertIn(model, self.worker)
        self.assertIn("DRL_MULTI_LOAD_ACTOR_ONLY=0", self.worker)
        self.assertIn("DRL_MULTI_REQUIRE_MODEL_LOAD=1", self.worker)

    def test_three_robot_pilot_budget_is_frozen(self):
        self.assertIn("DRL_MULTI_NUM_AGENTS=3", self.worker)
        self.assertIn("DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=10000", self.worker)
        self.assertIn("DRL_MULTI_EVAL_EPISODES=120", self.worker)
        self.assertIn("DRL_MULTI_MAX_EPOCHS=2", self.worker)

    def test_uses_gpu_and_repository_logs(self):
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", self.worker)
        self.assertIn("logs/active/capacity-wide-g12-r2/s3-n3", self.start)
        self.assertIn("actor-g12-r2-s3-n3", self.experiment)


if __name__ == "__main__":
    unittest.main()
