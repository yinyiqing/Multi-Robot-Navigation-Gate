# Qualitative trajectory capture

This directory contains the first post-sealed qualitative rollout capture. The 5A trajectory is
valid, but the B2 trajectory is invalid: raw LiDAR collection needed by the Router was accidentally
disabled while omitting point clouds from JSONL serialization. Do not use the B2 trajectory or the
derived Gate timeline. Corrected B2 data are stored in `../trajectory_capture_corrected/`.

This capture does not modify or extend the G25 sealed statistics.

- Source scenes: 64 scenes selected from the frozen G25 Dense test manifest, stratified into four
  outcome groups (16 each) using the frozen `5A` and `router-b2` results.
- Rollouts: one fixed-seed (`20260910`) run for each of `5A` and `router-b2`, 64 episodes per method.
- Deployment inputs: the same local-observation protocol as the mainline evaluation. No robot truth,
  communication, or privileged 2 m switch is used.
- Recorded fields: per-step robot positions, actions, terminal state, actor mode, and Gate probability.
  Raw LiDAR arrays are intentionally disabled to keep the capture lightweight.

The trajectory files are for qualitative visualization only. They must not be used to recompute
success, collision, confidence intervals, or the G25 confirmation test.

## Recreate figures

```bash
source env.python.sh
python scripts/plot_trajectory_capture.py \
  --manifest paper/generated/captures/trajectory_capture/trajectory_subset_64.json.gz \
  --five-a experiments/03_保留专门化/02_论文主线/25_最终消融与Sealed评测/local_data/sealed/results/g25_sealed_5a_s20260901.npy \
  --b2 experiments/03_保留专门化/02_论文主线/25_最终消融与Sealed评测/local_data/sealed/results/g25_sealed_b2_s20260901.npy \
  --five-a-trajectory paper/generated/captures/trajectory_capture/trajectories/trajectory_capture_5a_s20260910.jsonl \
  --b2-trajectory paper/generated/captures/trajectory_capture/trajectories/trajectory_capture_b2_s20260910.jsonl \
  --output-dir paper/generated/captures/trajectory_capture/figures_3 \
  --limit 3

python scripts/plot_gate_timeline.py \
  --selection paper/generated/captures/trajectory_capture/figures_3/trajectory_overview_selection.json \
  --trajectory paper/generated/captures/trajectory_capture/trajectories/trajectory_capture_b2_s20260910.jsonl \
  --output-dir paper/generated/captures/trajectory_capture/figures_3
```

The current three-scene figure intentionally shows a rescue case, a paired-success case, and a
paired-failure case. It is evidence of mechanism and behavior, not a representative estimate of
the aggregate test distribution.
