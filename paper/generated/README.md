# Generated Paper Artifacts

These tables and SVG figures are generated from the frozen G25, Q1, and G26-E1 statistics. They are
not additional experiment results and do not read or modify model checkpoints.

```bash
source env.python.sh
python scripts/generate_paper_artifacts.py
```

Outputs:

- `piroute_overview.svg`: conceptual training/deployment information-boundary schematic;
- `g25_main_table.csv` and `g25_main_table.md`: confirmatory sealed main table;
- `g25_pareto.svg`: full-success, collision, and raw-step trade-off;
- `g25_primary_effects.svg`: PIRoute effects relative to 5A with scene-cluster BCa intervals;
- `g26_supplement_table.csv` and `g26_supplement_table.md`: Q1 and E1 supplemental table;
- `g26_e1_effects.svg`: exploratory E1 full-success effects;
- `generation_record.json`: source and output SHA-256 values.

Episode-level statistics do not contain frame-level trajectories. Router probability/mode timelines
and representative videos therefore require a separate frame-recording capture and are not fabricated
from these artifacts.
