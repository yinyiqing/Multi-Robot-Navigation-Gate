import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_manifest_conflict_horizon import full_path_horizon


class ManifestConflictHorizonTests(unittest.TestCase):
    def test_horizon_covers_longest_path(self):
        paths = {
            "r1": [(0.0, 0.0), (1.0, 0.0)],
            "r2": [(0.0, 0.0), (0.0, 3.0)],
        }
        self.assertAlmostEqual(full_path_horizon(paths, 0.5), 6.2)


if __name__ == "__main__":
    unittest.main()
