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

## 结论

100 场诊断中的 v3 `+4` full-success points 没有在完整 validation 上复现。v3 不仅没有成功率收益，而且显著增加完成步数并提高 timeout，因此拒绝作为 standard expert。timeout terminal 修复仍是正确性修复，但不能单独使当前 full-Actor TD3 微调产生可靠收益。

归档保留两份完整日志、逐 episode 数组、最终 state 和顺序 runner 日志；文件哈希记录在 `summary.json`。
