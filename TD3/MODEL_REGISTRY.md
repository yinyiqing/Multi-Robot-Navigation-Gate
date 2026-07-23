# 模型注册表

代码仍使用历史 artifact 文件名以保持兼容；论文、图表和新命令只使用本表的短模型 ID。

## 当前模型

| 模型 ID | 角色 | 实际 artifact 前缀 | 状态 |
| --- | --- | --- | --- |
| `generalist-5a` | 普通五车共享 Actor 候选 | `TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best` | historical baseline |
| `weak-interaction-5d` | 冻结的弱交互 Actor / 论文 baseline | `TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best` | current |
| `bridge-full-ft` | 5D 上完整 Actor dense 微调 | `TD3_multi_dense5_bridge_geo_critic_from_5d_best` | failed |
| `bridge-head-only` | 5D 上只训练动作头 | `TD3_multi_dense5_bridge_from_5d_head_only_best` | failed |
| `moderate-full-ft` | moderate cases 上完整 Actor 微调 | `TD3_multi_dense5_moderate_geo_critic_from_5d_best` | failed |
| `strong-interaction-gru` | 冻结 5D + 8 帧 GRU 动作修正 | `interaction_expert_temporal_gru_pilot_s20260723` | pilot |
| `temporal-gate` | 本地观测历史门控 | 待 D5 后命名 | planned |

## 文件含义

```text
TD3/pytorch_models/<prefix>_actor.pth   发布/评测 Actor 权重
TD3/pytorch_models/<prefix>_critic.pth  对应 Critic 权重
TD3/checkpoints/*.pt                    训练或测试恢复状态
TD3/results/*.npy                       本地统计快照
TD3/runs/*                              TensorBoard 事件
```

只有 `pytorch_models` 中通过正式验证的 best 权重才能称为模型。`latest` checkpoint、测试 state 和 TensorBoard 目录都不能作为论文模型引用。

## 新命名规则

新 artifact 使用以下顺序，避免继续把完整训练历史编码进文件名：

```text
<method>_<scenario>_n<agents>_seed<seed>_<selection>
```

例如：

```text
residual_interaction-medium-high_n5_seed0_best
```

训练来源、commit、超参数和数据 split 写入 manifest，不再追加到文件名。历史模型不批量重命名，因为训练脚本和归档日志仍引用原名。

## 使用限制

- `weak-interaction-5d` 对应原 `generalist-5d` 权重；历史脚本仍可使用旧 ID。
- 三个 `failed` 模型只作为 full fine-tune/head-only 失败证据，不作为专家。
- `strong-interaction-gru` 必须通过 deep/close/margin 准入条件和 paired evaluation 后才能登记为 current。
- `temporal-gate` 只有在 specialist 达到论文协议 D5 准入条件后才允许创建。
