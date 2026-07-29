import gzip
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "start_training_independent_dense_actor_from_5a.sh"
FAST_MONITOR = (
    ROOT
    / "experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/"
    "dense_validation_monitor_ultrafast_v3/validation.json.gz"
)


class IndependentDenseActorProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = LAUNCHER.read_text(encoding="utf-8")

    def export_value(self, name):
        match = re.search(
            rf"^\s*export\s+{re.escape(name)}=([^\n]+)$",
            self.script,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(match, f"Missing protocol setting: {name}")
        return match.group(1).strip().strip("'\"")

    def test_one_actor_controls_the_full_episode(self):
        self.assertEqual(
            self.export_value("DRL_MULTI_USE_ORACLE_INTERACTION_ROLLOUT"), "0"
        )
        self.assertEqual(self.export_value("DRL_MULTI_ACTOR_INTERACTION_ONLY"), "0")
        self.assertEqual(self.export_value("DRL_MULTI_ACTOR_SAFETY_FOCUSED"), "0")

    def test_5a_is_only_a_warm_start_and_safe_state_reference(self):
        self.assertEqual(self.export_value("DRL_MULTI_LOAD_ACTOR_ONLY"), "1")
        self.assertEqual(
            self.export_value("DRL_MULTI_ACTOR_ANGULAR_ANCHOR_WEIGHT"), "0.0"
        )
        self.assertEqual(self.export_value("DRL_MULTI_ACTOR_ANCHOR_SAFE_ONLY"), "1")

    def test_interactions_are_oversampled_but_not_exclusive(self):
        self.assertEqual(
            self.export_value("DRL_MULTI_CRITIC_INTERACTION_FRACTION"), "0.75"
        )
        self.assertEqual(self.export_value("DRL_MULTI_ACTOR_INTERACTION_ONLY"), "0")

    def test_short_epochs_preserve_the_training_budget(self):
        self.assertEqual(
            self.export_value("DRL_MULTI_EVAL_FREQ_AGENT_SAMPLES"), "10000"
        )
        self.assertEqual(self.export_value("DRL_MULTI_EVAL_EPISODES"), "50")
        self.assertEqual(
            self.export_value("DRL_MULTI_ACTOR_UPDATE_DELAY_STEPS"), "21000"
        )
        self.assertIn('MAX_EPOCHS="${DRL_MULTI_MAX_EPOCHS:-96}"', self.script)

    def test_fast_monitor_is_fixed_and_representative(self):
        with gzip.open(FAST_MONITOR, "rt", encoding="utf-8") as handle:
            manifest = json.load(handle)
        scenarios = manifest["scenarios"]
        summary = manifest["view_config"]["monitor_summary"]

        self.assertEqual(
            manifest["dataset_id"], "dense-validation-monitor-ultrafast-v3"
        )
        self.assertTrue(manifest["view_config"]["policy_independent"])
        self.assertEqual(len(scenarios), 50)
        self.assertEqual(len({item["scenario_id"] for item in scenarios}), 50)
        self.assertAlmostEqual(summary["mean_conflict_edges"], 2.48)


if __name__ == "__main__":
    unittest.main()
