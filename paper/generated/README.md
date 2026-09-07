# Generated Paper Artifacts

These tables and SVG figures are generated from the frozen G25, Q1, and G26-E1 statistics. They are
not additional experiment results and do not read or modify model checkpoints.

```bash
source env.python.sh
python scripts/generate_paper_artifacts.py
```

The generator first rebuilds the tables and then calls
`scripts/redraw_paper_figures.py`, which rebuilds the data-driven main and
supplementary figures as editable SVG files. Fig. 1 is manually maintained and
is intentionally not overwritten by either script.

Outputs are grouped by figure rather than mixed in this directory:

- `fig1_overview/`: manually maintained training/deployment overview;
- `fig2_tradeoff/`: G25 full-success, collision, and raw-step trade-off;
- `fig3_effects/`: G25 PIRoute effects relative to 5A with scene-cluster BCa intervals;
- `fig4_cycle/`: one real post-sealed `standard -> interaction -> standard` cycle;
- `figS1_generalization/`: exploratory robot-count generalization;
- `figS2_trajectories/`: representative closed-loop trajectories;
- `figS3_gate_timeline/`: Router probability and executed-policy timelines;
- `figS4_external_router/`: exploratory G26-E1 external Router comparison;
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
