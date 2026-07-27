# 当前冻结方案

这是当前方法唯一有效的 Actor 证据目录。两个 Actor 均已冻结；后续不得继续更新它们。

| 实验 | 状态 | 结论 |
| --- | --- | --- |
| `D4_interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726` | frozen candidate | 2560 场均衡全池训练完成 16 epoch，选择 epoch 16 |
| `D4_interaction_actor_matched_validation_s20260727` | complete | 5A + 条件交互 Actor 的 full success 从 `0.421` 提高到 `0.700` |
| `D4_weak_actor_5a_vs_5d_s20260727` | complete | 0-edge 场景上 5A/5D 等价；选择与交互 Actor 训练分布一致的 5A |

当前系统定义：

```text
普通状态       -> frozen generalist-5a
紧迫机器人交互 -> frozen strong-interaction-5a-balanced epoch 16
状态选择       -> 待训练的 deployable interaction-gate
```

训练期 `2.0 m` 真值距离切换只用于验证组合上限，不是最终 Gate。
