# D3 Fixed-v1 Generalist Validation

状态：`complete`。冻结的 5D Actor 已按顺序运行全部 1000 场 `dense/validation`；未读取或修改 dense test。

## Dense 总结果

| Episodes | Agent success | Collision | Unresolved | Full success | Timeout | Mean steps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | `3561/5000 = 0.7122` | `1436/5000 = 0.2872` | `3/5000 = 0.0006` | `314/1000 = 0.3140` | `3/1000 = 0.0030` | 27.988 |

Episode bootstrap 95% CI（20,000 次，seed `20260719`）：agent success `[0.6970, 0.7272]`，full success `[0.2850, 0.3430]`。

## 交互分层

| Stratum | N | Agent success | Collision | Full success | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `edges = 0` | 42 | 0.9905 | 0.0048 | 0.9524 | 30.262 |
| `edges > 0` | 958 | 0.7000 | 0.2996 | 0.2860 | 27.888 |

逐冲突边结果：

| Edges | N | Agent success | Collision | Full success |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 42 | 0.9905 | 0.0048 | 0.9524 |
| 1 | 211 | 0.8531 | 0.1469 | 0.5592 |
| 2 | 294 | 0.7435 | 0.2565 | 0.3367 |
| 3 | 254 | 0.6606 | 0.3394 | 0.1535 |
| 4 | 127 | 0.5780 | 0.4205 | 0.1181 |
| 5+ | 72 | 0.4278 | 0.5708 | 0.0417 |

## 三组基线

同一 5D Actor 的 validation 结果：

| Group | Episodes | Agent success | Full success | Collision |
| --- | ---: | ---: | ---: | ---: |
| standard low-interaction | 206 | 0.9680 | 0.8544 | 0.0291 |
| standard interaction | 294 | 0.8143 | 0.4252 | 0.1823 |
| dense overall | 1000 | 0.7122 | 0.3140 | 0.2872 |

dense 的 0-edge 场景成功率很高，而有冲突时明显下降，说明主要难度来自多机器人交互，不是空间缩小本身。该结果支持保留 5D 作为低交互普通专家候选，并将 specialist 训练集中在 standard-interaction 与 dense-interaction；运行时 gate 不得读取离线冲突标签。

完整结构化统计与 SHA-256 保存在 `summary.json`。
