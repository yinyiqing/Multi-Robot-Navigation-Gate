#!/usr/bin/env python3
"""Reconstruct a readable B4 episode log from the preserved result array.

The output is intentionally labelled reconstructed: wall-clock timing and
Gazebo startup messages are not present in the .npy result artifact.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "experiments/03_保留专门化/02_论文主线/28_RewardAwareGate稳健修正版/local_data/validation/results/g28_dense256_b4_s20260911.npy"
DEFAULT_OUTPUT = ROOT / "experiments/03_保留专门化/02_论文主线/28_RewardAwareGate稳健修正版/logs/b4_result_reconstructed.log"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = np.load(args.result, allow_pickle=True)
    if rows.shape != (256, 17):
        raise ValueError("expected complete B4 result with shape (256,17), got %s" % (rows.shape,))
    source_hash = hashlib.sha256(args.result.read_bytes()).hexdigest()
    lines = [
        "G28 B4 result log reconstructed from preserved .npy result array",
        "WARNING: this is not the original Gazebo stdout; startup messages and wall-clock samples/sec were overwritten and cannot be recovered.",
        "source=%s" % args.result,
        "source_sha256=%s" % source_hash,
        "shape=%s" % (rows.shape,),
        "columns=episode,total_env_steps,total_agent_samples,episode_env_steps,episode_agent_samples,mean_reward,success,collision,full_success,mean_final_distance,unresolved,timeout,case,rule_enabled,dense_action_share,gate_switches,gate_mean_probability",
        "",
    ]
    success_total = collision_total = unresolved_total = full_total = timeout_total = 0
    for row in rows:
        episode = int(row[0])
        success = int(row[6])
        collision = int(row[7])
        unresolved = int(row[10])
        full = int(row[8])
        timeout = int(row[11])
        success_total += success
        collision_total += collision
        unresolved_total += unresolved
        full_total += full
        timeout_total += timeout
        lines.append(
            "Episode %i complete | case=%s | env_steps=%i | agent_samples=%i | episode_env_steps=%i | "
            "episode_agent_samples=%i | mean_reward=%.3f | success=%i/5 | collision=%i/5 | "
            "unresolved=%i/5 | full_success=%i | timeout=%i | mean_final_distance=%.3f | "
            "dense_action_share=%.3f | gate_switches=%i | gate_mean_probability=%.3f | samples/sec=NA"
            % (
                episode,
                str(row[12]),
                int(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                float(row[5]),
                success,
                collision,
                unresolved,
                full,
                timeout,
                float(row[9]),
                float(row[14]),
                int(row[15]),
                float(row[16]),
            )
        )
    lines.extend(
        [
            "",
            "Reconstructed aggregate | episodes=256 | total_success=%i | total_collision=%i | total_unresolved=%i | total_full_success=%i | timeout_episodes=%i | full_success_rate=%.4f | agent_success_rate=%.4f | robot_collision_rate=%.4f | robot_unresolved_rate=%.4f | timeout_episode_rate=%.4f | raw_steps=%.4f | interaction_selection_share=%.4f | mean_switches=%.4f"
            % (
                success_total,
                collision_total,
                unresolved_total,
                full_total,
                timeout_total,
                float(np.mean(rows[:, 8].astype(float))),
                success_total / (256 * 5),
                collision_total / (256 * 5),
                unresolved_total / (256 * 5),
                timeout_total / 256,
                float(np.mean(rows[:, 3].astype(float))),
                float(np.mean(rows[:, 14].astype(float))),
                float(np.mean(rows[:, 15].astype(float))),
            ),
            "end_of_reconstructed_log",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
