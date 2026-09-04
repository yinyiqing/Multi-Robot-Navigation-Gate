# Generated Paper Artifacts

These tables and SVG figures are generated from the frozen G25, Q1, and G26-E1 statistics. They are
not additional experiment results and do not read or modify model checkpoints.

```bash
source env.python.sh
python scripts/generate_paper_artifacts.py
```

The generator first rebuilds the tables and then calls
`scripts/redraw_paper_figures.py`, which applies the compact ICRA publication
layout to the four SVG/PNG figures. The redraw script can also be run directly
when only the figure style needs to be regenerated.

Outputs are grouped by figure rather than mixed in this directory:

- `fig1_overview/`: training/deployment information-boundary schematic;
- `fig2_tradeoff/`: G25 full-success, collision, and raw-step trade-off;
- `fig3_effects/`: G25 PIRoute effects relative to 5A with scene-cluster BCa intervals;
- `fig4_cycle/`: one real post-sealed `standard -> interaction -> standard` cycle;
- `supplement/`: exploratory G26-E1 effect figure;
- `tables/`: confirmatory G25 and exploratory G26 tables;
- `captures/`: frame-level logs and intermediate qualitative data;
- `generation_record.json`: source and output SHA-256 values.

Episode-level statistics do not contain frame-level trajectories. Router probability/mode timelines
and representative videos therefore require a separate frame-recording capture and are not fabricated
from these artifacts.

The corrected post-sealed qualitative capture is documented in
`captures/trajectory_capture_corrected/README.md`. Its exported Fig. 4 files are in
`fig4_cycle/`. The earlier B2 trajectory under `captures/trajectory_capture/` is invalid because its
Router input buffer was accidentally disabled; only the old 5A trajectory remains reusable.
