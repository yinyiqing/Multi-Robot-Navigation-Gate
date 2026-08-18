import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_g25_sealed_results import bca_mean_interval, sign_flip_p


def test_bca_interval_contains_constant_mean():
    interval = bca_mean_interval(np.ones(20), samples=1000, seed=7)
    assert interval == [1.0, 1.0]


def test_sign_flip_is_two_sided_and_deterministic():
    values = np.asarray([1.0, 1.0, 1.0, 1.0])
    first = sign_flip_p(values, samples=5000, seed=11)
    second = sign_flip_p(values, samples=5000, seed=11)
    assert first == second
    assert 0.05 < first < 0.2
