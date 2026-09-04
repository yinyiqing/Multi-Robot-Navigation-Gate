# Figure audit record

更新时间：2026-09-01

## Figure contract

| Figure | Conclusion | Evidence | Independent unit | Statistics |
|---|---|---|---|---|
| Fig. 1 | Privileged simulator state is removed before local-observation deployment | Frozen training/deployment protocol | Conceptual schematic | Not applicable |
| Fig. 2 | PIRoute occupies a higher-success/lower-collision point with a step cost | G25 sealed pooled descriptives | 256 scenes x 3 repeats; 5 robots | Descriptive; no error bars |
| Fig. 3 | PIRoute improves full success and collision while increasing paired-success steps | G25 scene-cluster paired effects | Scene clusters over 3 repeats | BCa 95% CI, preregistered |
| Fig. 4 | Router enters and exits the interaction Actor in a real closed-loop rollout | Corrected post-sealed qualitative capture | One selected full-success episode | Qualitative; not inferential |
| Fig. S1 | External NF-inspired switching baseline does not explain the PIRoute effect | G26-E1 exploratory statistics | 128 scenes x 2 repeats | Exploratory BCa 95% CI |

## Data and export contract

- Source statistics are the frozen G25, Q1 and E1 JSON files recorded in `generation_record.json`.
- The generated tables and figures contain no post-hoc exclusions, smoothing, or result-dependent sorting.
- Quantitative plots are editable SVG with embedded text and explicit Arial fallback; 600-DPI PNG
  exports are provided for review and raster-only submission systems.
- Figure 2 uses raw termination steps only as bubble size; Figure 3 uses paired-success steps as defined in the manuscript.
- Episode-level inputs do not contain frame trajectories; no router timeline or video is inferred from them.

## Preflight results

- `qinyan-nature-figures/scripts/figure_preflight.py`: 7 PASS, 3 WARN, 0 FAIL.
- `scipilot-figure-skill/scripts/check_figure.py`: PASS for the four main SVG artifacts.
- Warnings concern the absence of raster exports and automatic font-strategy detection; these are non-blocking because the deliverables are vector SVGs with explicit `font-family="Arial, sans-serif"`.
- The schematic and quantitative figures should still be checked at final ICRA column width during typesetting.

## Reproduction

```bash
source env.python.sh
python scripts/generate_paper_artifacts.py
```
