# 冻结Actor证据

状态：`frozen evidence / historical naming`。目录名保留以避免破坏历史链接，但这里不再
代表当前训练方案。

| 实验 | 状态 | 当前用途 |
| --- | --- | --- |
| `D4_interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726` | frozen current | epoch-16条件避障Actor |
| `D4_interaction_actor_matched_validation_s20260727` | complete | 证明局部调用可将full success从`0.421`提高到`0.700` |
| `D4_weak_actor_5a_vs_5d_s20260727` | complete | 0-edge上5A/5D等价，冻结5A为普通导航Actor |
| `D4_dense_validation_actor_comparison_s20260728` | complete | epoch-16全程运行timeout严重，只能局部调用 |
| `D5_independent_dense_actor_from_5a_full_v1_s20260728` | rejected | 完整dense独立Actor没有超过5A |
| `D5_independent_dense_actor_from_5a_full_v2_s20260729` | cancelled | 未启动；其完整dense训练目标已被当前主线否定 |

当前系统定义见[论文主线](../../README.md)：

```text
Actor N = frozen 5A
Actor I = frozen epoch-16
Gate    = deployable local perception and temporal state
```

本目录中的独立Dense Actor和Residual方案不得恢复。若保留严格single-to-multi主张，
必须先处理epoch-16原训练集中的11场full-horizon edge-2污染。
