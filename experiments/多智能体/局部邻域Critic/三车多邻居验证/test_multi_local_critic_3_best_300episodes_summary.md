# 三车局部邻域 Critic best checkpoint 测试摘要

## 测试对象

- 模型：`TD3_velodyne_multi_v4_local_critic_3_best`
- 场景：三车，`r1, r2, r3`
- 测试规模：300 episodes
- 测试阶段 reward：individual reward，仅用于统计策略表现
- Actor 输入：本车 24 维 observation
- Critic 训练输入：本车 observation + 局部邻居 context + 本车 action

## 最终指标

| 指标 | 数值 |
| --- | ---: |
| `success_rate` | `0.913` |
| `collision_rate` | `0.052` |
| `full_success_rate` | `0.747` |
| `timeout_rate` | `0.100` |
| `success_80_rate` | `0.747` |
| `avg_reward` | `117.709` |
| `avg_final_distance` | `0.352` |
| `avg_episode_env_steps` | `72.3` |

累计计数：

- 单机器人成功：`822/900`
- 单机器人碰撞：`47/900`
- 三车全部成功 episode：`224/300`
- 至少 80% 机器人成功 episode：`224/300`
- timeout episode：`30/300`
- 发生碰撞 episode：`46/300`

## 结果解读

三车局部邻域 Critic 在 best checkpoint 下完成了 300 episodes 测试，单机器人成功率达到 `0.913`，三车全成功率达到 `0.747`。这说明在 actor 执行阶段仍只使用自身观测的前提下，训练阶段向 critic 开放局部邻居信息可以得到可用且较稳定的三车导航策略。

后续需要补齐三车公平对照实验，包括共享 policy baseline、reward-only/weighted08 和局部邻域 Critic，以判断提升是否来自 critic context，而不是 warm-start、训练预算或 reward shaping。
