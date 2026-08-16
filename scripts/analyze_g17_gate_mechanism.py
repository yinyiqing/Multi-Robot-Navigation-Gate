#!/usr/bin/env python3
import gzip, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
RUN = BASE / "22_G17_Gate机制对照/local_data"
MANIFEST = BASE / "datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz"
POLICIES = ("epoch17_always_on", "rule_2m_privileged")
SEEDS = (20260824, 20260825)

def stats(rows):
    n = len(rows)
    agents = n * 5
    return {
        "episodes": n,
        "agent_success_rate": float(rows[:, 6].astype(int).sum() / agents),
        "full_success_rate": float(rows[:, 8].astype(int).sum() / n),
        "collision_rate": float(rows[:, 7].astype(int).sum() / agents),
        "unresolved_rate": float(rows[:, 10].astype(int).sum() / agents),
        "timeout_episode_rate": float(rows[:, 11].astype(int).sum() / n),
        "mean_episode_steps": float(np.mean(rows[:, 3].astype(float))),
        "mean_interaction_share": float(np.mean(rows[:, 14].astype(float))),
        "mean_gate_switches": float(np.mean(rows[:, 15].astype(float))),
    }

def main():
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as f:
        ids = [str(x["scenario_id"]) for x in json.load(f)["scenarios"]]
    data = {}
    for seed in SEEDS:
        data[str(seed)] = {}
        for policy in POLICIES:
            p = RUN / "results" / f"g17_{policy}_s{seed}.npy"
            rows = np.load(p, allow_pickle=True)
            if rows.shape != (120, 17) or [str(x[12]) for x in rows] != ids:
                raise SystemExit(f"invalid result: {p}")
            if sum(int(x[6]) + int(x[7]) + int(x[10]) for x in rows) != 600:
                raise SystemExit(f"terminal accounting mismatch: {p}")
            data[str(seed)][policy] = stats(rows)
    summary = {"protocol": {"manifest": str(MANIFEST.relative_to(ROOT)), "seeds": SEEDS, "episodes_per_seed": 120, "sealed_test_read": False}, "per_seed": data}
    for policy in POLICIES:
        rows = [np.load(RUN / "results" / f"g17_{policy}_s{s}.npy", allow_pickle=True) for s in SEEDS]
        summary.setdefault("combined", {})[policy] = stats(np.concatenate(rows))
    out = RUN / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
