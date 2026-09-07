# Corrected qualitative trajectory capture

This directory contains the corrected post-sealed B2 trajectory capture for paper figures. The
learned Router used its required raw LiDAR buffer at runtime, while raw point clouds were omitted
from JSONL serialization. The capture does not modify or extend G25 sealed statistics.

- Scenes: the same 64 outcome-stratified scenes as the first qualitative capture.
- Seed: `20260910`.
- Frozen method: `generalist-5a + interaction-epoch16 + router-b2`.
- Audited data: 64 episodes, 320 per-robot sequences, no episode-switch reconstruction mismatch.
- Complete executed `standard -> interaction -> standard` cycles: 4 across 4 robots.
- Full-success episodes containing such a cycle: 2.

The old 5A trajectory in `../trajectory_capture/` remains valid and is reused only for matched
trajectory comparison. The old B2 trajectory is invalid and must not be used.

Key outputs:

- `analysis/mode_sequence_summary.json`: complete sequence audit and ranked real-cycle candidates.
- `analysis/mode_sequences.csv`: all 320 compressed per-robot mode sequences.
- `analysis/complete_cycles.csv`: the four complete cycles.
- `../../fig4_cycle/gate_cycle_example.svg`: selected full-success real-cycle figure.
- `figures_3/`: corrected supplemental trajectory overview and Router timeline.

Recreate the sequence audit and real-cycle figure:

```bash
source env.python.sh
python scripts/analyze_gate_trajectory.py \
  --trajectory paper/generated/captures/trajectory_capture_corrected/trajectories/trajectory_capture_b2_s20260910.jsonl \
  --results paper/generated/captures/trajectory_capture_corrected/results/trajectory_capture_b2_s20260910.npy \
  --output-dir paper/generated/captures/trajectory_capture_corrected/analysis \
  --seed 20260910

python scripts/plot_gate_cycle_example.py \
  --manifest paper/generated/captures/trajectory_capture_corrected/trajectory_subset_64.json.gz \
  --trajectory paper/generated/captures/trajectory_capture_corrected/trajectories/trajectory_capture_b2_s20260910.jsonl \
  --analysis paper/generated/captures/trajectory_capture_corrected/analysis/mode_sequence_summary.json \
  --output-dir paper/generated/fig4_cycle
```
