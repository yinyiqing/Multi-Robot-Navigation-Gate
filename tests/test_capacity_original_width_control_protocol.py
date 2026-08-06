import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P1_START = ROOT / "scripts" / "start_training_capacity_matched_actor.sh"
R1_START = ROOT / "scripts" / "start_training_capacity_original_width_control.sh"
R1_STOP = ROOT / "scripts" / "stop_training_capacity_original_width_control.sh"


def exported_values(script: str) -> dict[str, str]:
    return dict(
        re.findall(r"^export (DRL_MULTI_[A-Z0-9_]+)=(.+)$", script, re.MULTILINE)
    )


class CapacityOriginalWidthControlProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p1_script = P1_START.read_text(encoding="utf-8")
        cls.r1_script = R1_START.read_text(encoding="utf-8")
        cls.stop_script = R1_STOP.read_text(encoding="utf-8")
        cls.p1_exports = exported_values(cls.p1_script)
        cls.r1_exports = exported_values(cls.r1_script)

    def test_changes_only_registered_identity_width_budget_and_ports(self):
        allowed_differences = {
            "DRL_MULTI_EXPERIMENT_LABEL",
            "DRL_MULTI_TRAIN_FILE_NAME",
            "DRL_MULTI_TRAINING_VERSION",
            "DRL_MULTI_PID_FILE",
            "DRL_MULTI_LOG_DIR",
            "DRL_MULTI_ACTOR_HIDDEN_DIM_1",
            "DRL_MULTI_ACTOR_HIDDEN_DIM_2",
            "DRL_MULTI_ALLOW_ACTOR_WARMSTART_EXPANSION",
            "DRL_MULTI_MAX_EPOCHS",
            "DRL_MULTI_ROS_PORT",
            "DRL_MULTI_GAZEBO_PORT",
        }
        shared_keys = set(self.p1_exports) & set(self.r1_exports)
        unexpected = {
            key
            for key in shared_keys
            if self.p1_exports[key] != self.r1_exports[key]
            and key not in allowed_differences
        }
        self.assertEqual(unexpected, set())
        self.assertEqual(set(self.p1_exports) - set(self.r1_exports), set())
        self.assertEqual(set(self.r1_exports) - set(self.p1_exports), set())

    def test_uses_original_actor_without_expansion(self):
        self.assertEqual(self.r1_exports["DRL_MULTI_ACTOR_HIDDEN_DIM_1"], "800")
        self.assertEqual(self.r1_exports["DRL_MULTI_ACTOR_HIDDEN_DIM_2"], "600")
        self.assertEqual(
            self.r1_exports["DRL_MULTI_ALLOW_ACTOR_WARMSTART_EXPANSION"], "0"
        )

    def test_hard_caps_control_at_40k_samples(self):
        self.assertEqual(
            self.r1_exports["DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES"], "20000"
        )
        self.assertEqual(self.r1_exports["DRL_MULTI_MAX_EPOCHS"], "2")
        self.assertEqual(
            self.r1_exports["DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS"], "20000"
        )

    def test_freezes_inputs_and_uses_repository_log_directory(self):
        self.assertIn("d2_summary.json", self.r1_script)
        self.assertIn("g11_a1_gate_v1", self.r1_script)
        self.assertIn(
            "a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026",
            self.r1_script,
        )
        self.assertIn(
            "logs/active/capacity-original-width-r1", self.r1_script
        )
        self.assertNotIn("audit_capacity_matched_actor.py", self.r1_script)

    def test_requires_idle_cuda_capacity(self):
        self.assertIn(
            "--query-gpu=memory.free,utilization.gpu", self.r1_script
        )
        self.assertIn("free_mib < 8192", self.r1_script)
        self.assertIn("utilization > 20", self.r1_script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", self.r1_script)

    def test_stop_script_targets_only_r1(self):
        self.assertIn("capacity_original_width_r1_n5_seed20260810", self.stop_script)
        self.assertIn(".g12_capacity_r1.pid", self.stop_script)
        self.assertNotIn(".g12_capacity_actor.pid", self.stop_script)


if __name__ == "__main__":
    unittest.main()
