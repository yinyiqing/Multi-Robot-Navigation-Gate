# 三车共享 Policy Baseline best checkpoint 测试摘要

## 测试对象

- 模型：`TD3_velodyne_multi_v4_shared_policy_3_best`
- 场景：三车，`r1, r2, r3`
- 测试规模：300 episodes
- 测试阶段 reward：individual reward，仅用于统计策略表现
- Actor 输入：本车 24 维 observation
- 训练方式：多机器人共享同一个 actor/critic，不使用局部邻域 critic
- 初始化方式：从统一 warm-start 基准 `TD3_velodyne_multi_v4` 初始化

## 最终指标

| 指标 | 数值 |
| --- | ---: |
| `success_rate` | `0.926` |
| `collision_rate` | `0.056` |
| `full_success_rate` | `0.797` |
| `timeout_rate` | `0.053` |
| `success_80_rate` | `0.797` |
| `avg_reward` | `119.580` |
| `avg_final_distance` | `0.356` |
| `avg_episode_env_steps` | `48.4` |

累计计数：

- 单机器人成功：`833/900`
- 单机器人碰撞：`50/900`
- 三车全部成功 episode：`239/300`
- 至少 80% 机器人成功 episode：`239/300`
- timeout episode：`16/300`
- 发生碰撞 episode：`47/300`

## 结果解读

三车共享 Policy Baseline 在 best checkpoint 下完成了 300 episodes 测试，单机器人成功率达到 `0.926`，三车全成功率达到 `0.797`。该结果说明，在当前三车弱耦合场景中，仅使用共享 policy 并从统一基础导航模型 warm-start，就已经可以得到较强的多机器人导航能力。

这一结果是后续三车对照实验的重要基准。由于三车局部邻域 Critic 版本同时引入了距离加权协作 reward 和局部邻域 critic，后续仍需要补齐 reward-only 与 weighted08 对照，才能判断性能变化主要来自 reward 设计还是 critic 结构。

与当前三车局部邻域 Critic 的 300 episodes 结果相比，本 baseline 在 `success_rate` 和 `full_success_rate` 上略高，但 `collision_rate` 也略高：

| 方法 | `success_rate` | `collision_rate` | `full_success_rate` |
| --- | ---: | ---: | ---: |
| 三车共享 Policy Baseline | `0.926` | `0.056` | `0.797` |
| 三车局部邻域 Critic + weighted08 | `0.913` | `0.052` | `0.747` |

因此，当前三车场景下共享 policy baseline 是一个强对照组，不能被视为弱 baseline。局部邻域 Critic 的价值需要在 reward-only/weighted08 单变量对照，以及更高交互密度的 5 车或 10 车场景中继续验证。
