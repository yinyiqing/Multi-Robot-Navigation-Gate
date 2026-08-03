# 模型注册表

代码仍使用历史 artifact 文件名以保持兼容；论文、图表和新命令只使用本表的短模型 ID。

## 当前模型

| 模型 ID | 角色 | 实际 artifact 前缀 | 状态 |
| --- | --- | --- | --- |
| `generalist-5a` | 冻结Actor A / Actor B的基础导航策略 | `TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best` | frozen current |
| `interaction-teacher-epoch16` | 冻结的单冲突局部避让教师 | `interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016` | frozen teacher |
| `actor-b-single-edge-residual` | 冻结5A加epoch-16指导的Residual | 待R0/R1后命名 | current planned |
| `single-to-multi-conflict-gate` | 冻结A/B之间的可部署Gate | 待Actor B准入后命名 | pending |
| `learned-gate-g2a` | 历史可部署Gate基线 | G2-A checkpoint | frozen baseline; admission failed |
| `weak-interaction-5d` | 历史generalist / 论文对照 | `TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best` | baseline only |
| `bridge-full-ft` | 5D 上完整 Actor dense 微调 | `TD3_multi_dense5_bridge_geo_critic_from_5d_best` | failed |
| `bridge-head-only` | 5D 上只训练动作头 | `TD3_multi_dense5_bridge_from_5d_head_only_best` | failed |
| `moderate-full-ft` | moderate cases 上完整 Actor 微调 | `TD3_multi_dense5_moderate_geo_critic_from_5d_best` | failed |
| `strong-interaction-s1` | 5D Actor/Critic 完整 warm-start的课程Stage 1 | `strong_interaction_curriculum_stage1_s20260723` | failed |
| `gate-robot-perception` | 本机激光候选形状与相对运动前端 | G0/G1 pilot protocol | frozen frontend candidate |

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

- `generalist-5a`与`weak-interaction-5d`在固定0-edge validation上无显著差异；当前选择5A作为Actor A和Actor B的冻结base。
- 三个 `failed` 模型只作为 full fine-tune/head-only 失败证据，不作为专家。
- `strong-interaction-s1` 已验证退化，不作为后续warm-start来源。
- `interaction-teacher-epoch16`已通过匹配局部调用验证，但不是全程独立策略；它只能监督Residual，不能直接作为Actor B或整网warm-start续训。
- `actor-b-single-edge-residual`的5A base必须永久冻结，只允许Residual和对应Critic更新。
- Actor B和最终Gate训练只允许读取0-edge与single-edge数据；多冲突只用于零样本评测。
- `learned-gate-g2a`在exact-edge-2上有方向性收益但未通过准入，只作为新Gate基线。
- `gate-robot-perception`推理时只能使用本机传感器；仿真邻车位置只能生成训练标签和validation真值。
- `single-to-multi-conflict-gate`读取G0/G1连续证据，不要求感知前端先输出机器人硬标签；仿真真值只用于训练监督和validation。
