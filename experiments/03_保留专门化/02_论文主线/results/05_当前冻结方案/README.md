# 当前Actor方案

导师沟通后，当前目标改为“两个可独立完成导航的Actor + Gate”。5A作为普通Actor冻结；epoch-16条件Actor保留为有效的局部避险证据，但不再作为最终Dense Actor；新的独立Dense Actor从5A重训。

| 实验 | 状态 | 结论 |
| --- | --- | --- |
| `D4_interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726` | frozen candidate | 2560 场均衡全池训练完成 16 epoch，选择 epoch 16 |
| `D4_interaction_actor_matched_validation_s20260727` | complete | 5A + 条件交互 Actor 的 full success 从 `0.421` 提高到 `0.700` |
| `D4_weak_actor_5a_vs_5d_s20260727` | complete | 0-edge 场景上 5A/5D 等价；选择与交互 Actor 训练分布一致的 5A |
| `D4_dense_validation_actor_comparison_s20260728` | complete | 条件Actor独立运行超时严重；必须重训独立Dense Actor |
| `D5_independent_dense_actor_from_5a_full_v1_s20260728` | running | 5A warm-start，整场独立控制，使用完整dense/train |

当前目标系统：

```text
普通状态       -> frozen generalist-5a
Dense/多冲突状态 -> independent dense Actor（训练中）
状态选择          -> 两个独立Actor通过验证后再训Gate
```

训练期 `2.0 m` 真值距离切换只保留为历史条件策略证据，不用于新Dense Actor的rollout。
