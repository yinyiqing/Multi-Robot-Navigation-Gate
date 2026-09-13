# Final dual-supervision ablation

Status: running since 2026-09-13 09:45 CST; 16-scene infrastructure pilot followed by 256 scenes x 3 repeats.

## Question

Does the predicted one-step reward difference change closed-loop behavior beyond the interaction-phase
branch in the final dual-encoder Router?

## Controlled comparison

- Full method: archived G34 results from `36_G34_G25slice_matched`.
- Phase-only decision: the same G34 checkpoint and identical model weights, with only
  `router_decision.reward_weight` changed from `0.25` to `0.0`.
- Frozen Navigation Actor, Interaction Actor, G0/G1, Router weights, normalization, 8-frame history,
  thresholds `0.43/0.33`, minimum hold 3, and stride 2.
- Same G25 Dense manifest `[0:256]` and seeds `20260901/20260902/20260903`.

No Actor or Router training is authorized. The 16-scene pilot checks only infrastructure, result shape,
scene order, and terminal accounting; performance is not a stop condition. After it passes, the first
repeat resumes from scene 16 and the other two repeats run automatically.

This is a post-main-evaluation component ablation on a reused matched slice, not a new independent or
confirmatory test. The old proximity-supervised Router structure experiments are not relabeled as
ablations of the final reward-aware method.

Run with:

```bash
bash scripts/start_g34_phase_only_ablation.sh
```

Live log:

```text
experiments/03_保留专门化/02_论文主线/37_最终双监督消融/logs/matched/runner.log
```
