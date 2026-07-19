# D4 V3 Full Validation Comparison

状态：`rejected candidate`。原始 5D 与 v3 epoch 2 在完整 500 场 `standard/validation` 上按相同 scenario ID 和顺序评估；未读取 standard test。

## 总结果

| Model | Agent success | Collision | Unresolved | Full success | Timeout | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5D | 0.8776 | 0.1192 | 0.0032 | 0.6020 | 0.0160 | 42.446 |
| v3 epoch 2 | 0.8712 | 0.1232 | 0.0056 | 0.5920 | 0.0280 | 50.824 |
| v3 - 5D | -0.0064 | +0.0040 | +0.0024 | -0.0100 | +0.0120 | +8.378 |

Episode paired bootstrap 95% CI（20,000 次，seed `20260719`）：

- agent success difference: `[-0.0168, 0.0044]`
- full success difference: `[-0.0440, 0.0240]`
- timeout difference: `[0.0000, 0.0240]`
- mean-step difference: `[4.4800, 12.3981]`

Full-success 配对：both `261`，5D-only `40`，v3-only `35`，neither `164`；McNemar exact `p=0.6445`。

## 5D 按交互强度分层

使用 manifest 中与策略无关的 `conflict_edge_count` 对同一批 500 场结果分层，不重新运行仿真，也不读取 test：

| Stratum | Episodes | Agent success | Collision | Unresolved | Full success | Timeout | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low interaction (`edges = 0`) | 206 | 0.9680 | 0.0291 | 0.0029 | 0.8544 | 0.0146 | 40.277 |
| interaction (`edges > 0`) | 294 | 0.8143 | 0.1823 | 0.0034 | 0.4252 | 0.0170 | 43.966 |

Episode bootstrap 95% CI（20,000 次，seed `20260719`）：低交互组 agent/full success 为 `[0.9563, 0.9786]` / `[0.8058, 0.8981]`，交互组为 `[0.7932, 0.8354]` / `[0.3707, 0.4830]`。逐冲突边结果及计数保存在 `interaction_stratified_5d.json`。

该结果表明 standard 总体低成功率主要由交互子集驱动。它支持将 5D 暂时冻结为低交互普通专家，并单独研究交互专家；分层标签只用于训练采样和结果分析，不能作为运行时 gate 输入。

## 结论

100 场诊断中的 v3 `+4` full-success points 没有在完整 validation 上复现。v3 不仅没有成功率收益，而且显著增加完成步数并提高 timeout，因此拒绝作为 standard expert。timeout terminal 修复仍是正确性修复，但不能单独使当前 full-Actor TD3 微调产生可靠收益。

归档保留两份完整日志、逐 episode 数组、最终 state 和顺序 runner 日志；文件哈希记录在 `summary.json`。
