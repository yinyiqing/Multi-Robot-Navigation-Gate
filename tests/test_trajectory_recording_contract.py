import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "TD3/test_velodyne_td3_multi.py"


class TrajectoryRecordingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = EVALUATOR.read_text(encoding="utf-8")

    def test_perception_router_raw_lidar_is_not_guarded_by_serialization_flag(self):
        required_block = (
            'if actor_selection_mode in ("learned_gate", "ttc_cpa_gate"):',
            'os.environ["DRL_MULTI_RECORD_RAW_LIDAR"] = "1"',
        )
        start = self.source.index(required_block[0])
        end = self.source.index(
            'if actor_selection_mode in ("normalizing_flow_switch", "nf_switch"):',
            start,
        )
        block = self.source[start:end]
        self.assertIn(required_block[1], block)
        self.assertNotIn("DRL_MULTI_TRAJECTORY_INCLUDE_RAW_LIDAR", block)

    def test_raw_lidar_jsonl_serialization_has_its_own_flag(self):
        self.assertIn(
            "if env.record_raw_lidar and trajectory_path and trajectory_include_raw_lidar",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
