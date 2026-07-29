import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "start_training_independent_dense_actor_from_5a.sh"


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


if __name__ == "__main__":
    unittest.main()
