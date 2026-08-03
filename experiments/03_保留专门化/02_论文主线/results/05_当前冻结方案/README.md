# 冻结Actor证据

状态：`frozen evidence / historical naming`。目录名保留以避免破坏历史链接，但这里不再
代表当前训练方案。

| 实验 | 状态 | 当前用途 |
| --- | --- | --- |
| `D4_interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726` | frozen teacher | epoch-16单冲突局部技能教师 |
| `D4_interaction_actor_matched_validation_s20260727` | complete | 证明局部调用可将full success从`0.421`提高到`0.700` |
| `D4_weak_actor_5a_vs_5d_s20260727` | complete | 0-edge上5A/5D等价，冻结5A为Actor A和Residual base |
| `D4_dense_validation_actor_comparison_s20260728` | complete | epoch-16独立运行timeout严重，不能直接作为Actor B |
| `D5_independent_dense_actor_from_5a_full_v1_s20260728` | rejected | 完整dense独立Actor没有超过5A |
| `D5_independent_dense_actor_from_5a_full_v2_s20260729` | cancelled | 未启动；其完整dense训练目标已被当前主线否定 |

当前系统定义见[论文主线](../../README.md)：

```text
Actor A = frozen 5A
Actor B = frozen 5A + epoch-16-guided single-conflict Residual
Gate    = frozen A/B + deployable local perception
```

本目录中的独立Dense Actor方案不得恢复。Actor B和Gate训练都只能看0-edge与单冲突
数据，多冲突保留为零样本组合测试。
