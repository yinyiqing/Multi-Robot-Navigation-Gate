import gzip
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts/start_training_g12_r3_40k.sh"
WORKER = ROOT / "scripts/run_training_g12_r3_40k_worker.sh"
EXPERIMENT = ROOT / "scripts/experiment.sh"
MANIFEST = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views"
    / "g12_r3_mixed_v1/train.json.gz"
)


class G12R3ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = START.read_text(encoding="utf-8")
        cls.worker = WORKER.read_text(encoding="utf-8")
        cls.experiment = EXPERIMENT.read_text(encoding="utf-8")
        with gzip.open(MANIFEST, "rt", encoding="utf-8") as handle:
            cls.manifest = json.load(handle)

    def test_freezes_inputs_and_actor_only_warm_start(self):
        self.assertIn(
            "c2ce37e51e8e98423d6ed6d295a7f5cf54d02e76c42f6459ce35003c899e0841",
            self.start,
        )
        self.assertIn(
            "52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635",
            self.start,
        )
        self.assertIn("epoch_001", self.worker)
        self.assertIn("DRL_MULTI_LOAD_ACTOR_ONLY=1", self.worker)

    def test_manifest_has_exact_four_slot_schedule(self):
        scenarios = self.manifest["scenarios"]
        self.assertEqual(len(scenarios), 24000)
        streams = [item["view"]["g12_r3_stream"] for item in scenarios]
        self.assertEqual(streams[:8], ["standard", "strong", "dense", "strong"] * 2)
        self.assertEqual(
            Counter(streams), {"standard": 6000, "dense": 6000, "strong": 12000}
        )
        self.assertEqual(len({item["scenario_id"] for item in scenarios}), 24000)

    def test_freezes_conservative_actor_update(self):
        self.assertIn("DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS=21000", self.worker)
        self.assertIn("DRL_MULTI_ACTOR_Q_NORMALIZATION_ALPHA=1.0", self.worker)
        self.assertIn("DRL_MULTI_ACTOR_ANCHOR_WEIGHT=1.0", self.worker)
        self.assertIn("DRL_MULTI_ACTOR_ANCHOR_SAFE_ONLY=1", self.worker)
        self.assertIn("DRL_MULTI_ACTOR_GRAD_NORM_CLIP=1.0", self.worker)

    def test_full_actor_controls_all_states_with_local_critic(self):
        self.assertIn("DRL_MULTI_USE_LOCAL_CRITIC=1", self.worker)
        self.assertIn("DRL_MULTI_LOCAL_CRITIC_GEOMETRY_ONLY=1", self.worker)
        self.assertIn("DRL_MULTI_USE_ORACLE_INTERACTION_ROLLOUT=0", self.worker)
        self.assertIn("DRL_MULTI_USE_ORACLE_TARGET_POLICY=0", self.worker)
        self.assertIn("DRL_MULTI_ACTOR_INTERACTION_ONLY=0", self.worker)

    def test_budget_reward_and_entrypoint_are_frozen(self):
        self.assertIn("DRL_MULTI_REWARD_SELF_WEIGHT=0.8", self.worker)
        self.assertIn("DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES=20000", self.worker)
        self.assertIn("DRL_MULTI_EVAL_EPISODES=120", self.worker)
        self.assertIn("DRL_MULTI_MAX_EPOCHS=2", self.worker)
        self.assertIn("actor-g12-r3-40k", self.experiment)


if __name__ == "__main__":
    unittest.main()
