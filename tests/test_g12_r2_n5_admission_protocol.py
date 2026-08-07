import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "start_g12_r2_n5_admission.sh"
WORKER = ROOT / "scripts" / "run_g12_r2_n5_admission_worker.sh"
ANALYZER = ROOT / "scripts" / "analyze_g12_r2_n5_admission.py"


class G12R2N5AdmissionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = START.read_text(encoding="utf-8")
        cls.worker = WORKER.read_text(encoding="utf-8")
        spec = importlib.util.spec_from_file_location("n5_admission", ANALYZER)
        cls.analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.analyzer)

    def test_freezes_manifest_models_and_seed(self):
        self.assertIn(
            "e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7",
            self.start,
        )
        self.assertIn(
            "fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5",
            self.start,
        )
        self.assertIn(
            "67290450484c1fedd493fb029804b914438c5fb46cdb189ba8c642c3d98b2715",
            self.start,
        )
        self.assertIn("SEED=20260817", self.worker)

    def test_runs_policies_serially_with_matching_protocol(self):
        self.assertLess(
            self.worker.index("run_one 5a"),
            self.worker.index('run_one "$R2_POLICY"'),
        )
        self.assertIn("DRL_MULTI_MANIFEST_SAMPLING=cycle", self.worker)
        self.assertIn("DRL_MULTI_FIXED_PHYSICS_STEP_SIZE=0.001", self.worker)
        self.assertIn("DRL_MULTI_TEST_TARGET_EPISODES=\"$EPISODES\"", self.worker)

    def test_validates_order_and_supports_checked_resume(self):
        self.assertIn("observed != expected", self.worker)
        self.assertIn("verify_partial_result", self.worker)
        self.assertIn("curriculum_case_index", self.worker)
        self.assertIn("seq 1 5", self.worker)

    def test_registers_only_the_saved_10k_fallback(self):
        self.assertIn("10k)", self.start)
        self.assertIn("epoch_001_actor.pth", self.start)
        self.assertIn(
            "ace910553931873a275d66e3a964fd2b4716d30b6c68c8dcb3e7af96e56783ee",
            self.start,
        )
        self.assertIn("--candidate-policy", self.worker)

    def test_mcnemar_exact(self):
        self.assertEqual(self.analyzer.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(self.analyzer.mcnemar_exact(5, 0), 0.0625)


if __name__ == "__main__":
    unittest.main()
