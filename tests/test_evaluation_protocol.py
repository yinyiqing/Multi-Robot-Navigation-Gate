import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from evaluation_protocol import build_eval_protocol_id, reconcile_evaluation_state


class EvaluationProtocolTests(unittest.TestCase):
    def test_protocol_tracks_manifest_content_and_episode_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json.gz"
            path.write_bytes(b"first")
            first = build_eval_protocol_id("manifest", path, 40, 300)
            same = build_eval_protocol_id("manifest", path, 40, 300)
            more_episodes = build_eval_protocol_id("manifest", path, 100, 300)
            path.write_bytes(b"second")
            changed_content = build_eval_protocol_id("manifest", path, 40, 300)

        self.assertEqual(first, same)
        self.assertNotEqual(first, more_episodes)
        self.assertNotEqual(first, changed_content)

    def test_matching_protocol_keeps_best_state(self):
        checkpoint = {
            "eval_protocol_id": "fixed-40",
            "evaluations": [[1.0]],
            "best_eval_summary": {"full_success_rate": 0.7},
            "best_epoch": 5,
            "evaluation_history": [],
        }
        state = reconcile_evaluation_state(checkpoint, "fixed-40")
        self.assertEqual(state[:3], ([[1.0]], {"full_success_rate": 0.7}, 5))
        self.assertFalse(state[4])

    def test_changed_protocol_archives_and_resets_best_state(self):
        checkpoint = {
            "evaluations": [[1.0]],
            "best_eval_summary": {"full_success_rate": 0.8},
            "best_epoch": 1,
        }
        evaluations, best, best_epoch, history, reset, previous = (
            reconcile_evaluation_state(checkpoint, "fixed-40")
        )
        self.assertEqual(evaluations, [])
        self.assertIsNone(best)
        self.assertIsNone(best_epoch)
        self.assertTrue(reset)
        self.assertIsNone(previous)
        self.assertEqual(history[0]["eval_protocol_id"], "legacy-unversioned")
        self.assertEqual(history[0]["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
