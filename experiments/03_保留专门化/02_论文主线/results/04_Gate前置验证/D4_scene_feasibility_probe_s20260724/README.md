# 固定强交互场景可解性检查

## 场景合法性

- `strong_interaction validation`：`140/140` 通过 Gazebo reset 检查。
- `close/deep/margin`：分别 `68/68`、`156/156`、`199/199` 通过。
- 没有初始碰撞、复位误差或障碍物模型缺失。

## 20 场 privileged yield pilot

同一 validation 顺序、同一 seed，比较纯 5D 与基于 manifest 冲突边的固定让行规则：

| 控制器 | full success | agent success | collision |
|---|---:|---:|---:|
| 5D | 7/20 | 77/100 | 23/100 |
| 5D + conflict-pair yield | 6/20 | 76/100 | 24/100 |

结论：当前简单让行规则没有超过 5D，不能作为场景可解性的上界，也不值得扩大到 140 场。下一步若继续做经典控制基线，应接入成熟 ORCA/RVO，而不是继续调手写停车规则。

原始日志在 `logs/`。
