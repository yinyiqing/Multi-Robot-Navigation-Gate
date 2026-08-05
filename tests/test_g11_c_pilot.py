import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_g11_c_pilot import aggregate, archive_completed_logs, mcnemar_exact, paired
from build_g11_c_pilot_view import select_scenarios


def scenario(pool, topology, index):
    return {
        "scenario_id": "%s-%s-%d" % (pool, topology, index),
        "view": {"gate_pool": pool, "gate_topology": topology},
    }


def result_row(case, success=5, collision=0, unresolved=0, full=1, timeout=0):
    row = np.zeros(17, dtype=object)
    row[3] = 20
    row[6] = success
    row[7] = collision
    row[8] = full
    row[10] = unresolved
    row[11] = timeout
    row[12] = case
    return row


class PilotViewTest(unittest.TestCase):
    def test_selection_satisfies_cross_balanced_quotas(self):
        source = []
        for index in range(20):
            for pool in ("standard", "dense"):
                for topology in ("zero", "edge1"):
                    source.append(scenario(pool, topology, index))
        selected = select_scenarios(source)
        counts = Counter(item["view"]["g11_c_stratum"] for item in selected)
        self.assertEqual(len(selected), 50)
        self.assertEqual(counts["standard_zero"], 13)
        self.assertEqual(counts["standard_edge1"], 12)
        self.assertEqual(counts["dense_zero"], 12)
        self.assertEqual(counts["dense_edge1"], 13)

    def test_aggregate_checks_agent_outcome_accounting(self):
        metrics = aggregate(np.asarray([result_row("a")], dtype=object))
        self.assertEqual(metrics["full_success_rate"], 1.0)
        with self.assertRaises(ValueError):
            aggregate(
                np.asarray(
                    [result_row("bad", success=4, collision=0, unresolved=0)],
                    dtype=object,
                )
            )

    def test_paired_counts_improvements_and_degradations(self):
        baseline = np.asarray(
            [result_row("a", full=0), result_row("b", full=1)], dtype=object
        )
        candidate = np.asarray(
            [result_row("a", full=1), result_row("b", full=0)], dtype=object
        )
        metrics = paired(candidate, baseline)
        self.assertEqual(metrics["full_success_improved"], 1)
        self.assertEqual(metrics["full_success_degraded"], 1)
        self.assertEqual(metrics["mcnemar_exact_p"], 1.0)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_completed_logs_move_from_active_to_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "logs/active/gate-g11-c-pilot"
            archive = root / "logs/archive/validation/g11_c"
            legacy_logs = root / "experiment/local_data/logs"
            legacy_runner = root / "experiment/local_data/pilot_runner.log"
            active.mkdir(parents=True)
            (active / "pilot_runner.log").write_text("runner\n", encoding="utf-8")
            (active / "policy.log").write_text("policy\n", encoding="utf-8")
            legacy_logs.parent.mkdir(parents=True)
            legacy_logs.symlink_to(active, target_is_directory=True)
            legacy_runner.symlink_to(active / "pilot_runner.log")

            moved = archive_completed_logs(
                active, archive, legacy_runner, legacy_logs
            )

            self.assertEqual(len(moved), 2)
            self.assertFalse(active.exists())
            self.assertEqual((archive / "policy.log").read_text(), "policy\n")
            self.assertEqual(legacy_runner.resolve(), archive / "pilot_runner.log")
            self.assertEqual(legacy_logs.resolve(), archive)


if __name__ == "__main__":
    unittest.main()
