# 20260730 v5：协调/重启奖励下的独立 Dense Actor

## 目的

在 v4 的让行/停车奖励基础上，继续解决强交互 actor 的核心问题：

> 车会慢下来之后，还需要知道谁让、谁走、让行时如何拉开距离、安全后什么时候重新启动。

本实验仍然坚持“独立 actor”约束：

- 从 5A actor warm-start；
- actor 独立控制完整 episode；
- 不使用 5A/5D runtime takeover；
- 不使用 oracle rollout；
- 不使用 actor switching；
- 保持 TD3 主体结构不变。

## 相比 v4 的新增机制

在 `yield_priority_reward` 中增加短期让行状态：

1. 低优先级车继续往前顶：惩罚；
2. 低优先级车等待：奖励；
3. 等待过程中与最近车距离变大：额外奖励；
4. 危险解除后重新前进：奖励；
5. 已经安全但继续停住：惩罚。

主要代码：

- `TD3/multi_agent_velodyne_env.py`
- `TD3/train_velodyne_td3_multi.py`
- `scripts/start_training_independent_dense_actor_from_5a.sh`
- `tests/test_multi_agent_reward.py`

单元测试：

`Ran 40 tests OK`

## 训练配置

- 模型名：`independent_dense_actor_from_5a_coordination_v5_s20260729`
- 初始模型：`TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best`
- 训练集：`datasets/fixed_v1/dense/train.json.gz`
- validation：`datasets/fixed_v1/views/dense_validation_monitor_ultrafast_v3/validation.json.gz`
- validation episodes：50
- best metric：`full_success`

## 结果

| epoch | success | full success | collision | timeout |
|---|---:|---:|---:|---:|
| 1 | 0.748 | 0.400 | 0.252 | 0.000 |
| 3 | 0.764 | 0.440 | 0.236 | 0.000 |
| 5 | 0.776 | 0.460 | 0.224 | 0.000 |
| 9 | 0.824 | 0.500 | 0.164 | 0.060 |
| 13 | 0.796 | 0.440 | 0.172 | 0.160 |
| 17 | 0.768 | 0.320 | 0.120 | 0.400 |
| 20 | 0.608 | 0.200 | 0.128 | 0.680 |

最好结果出现在 epoch 9。

best checkpoint：

`checkpoints/independent_dense_actor_from_5a_coordination_v5_s20260729_best.pt`

latest checkpoint：

`checkpoints/independent_dense_actor_from_5a_coordination_v5_s20260729_latest.pt`

训练日志：

`logs/archive/rejected/independent_dense_actor/train_independent_dense_actor_from_5a_coordination_v5_s20260729_20260729_234629.log`

## 与 v4 对比

| 模型 | best epoch | success | full success | collision | timeout |
|---|---:|---:|---:|---:|---:|
| v4 让行停车 | 7 | 0.796 | 0.480 | 0.204 | 0.000 |
| v5 协调重启 | 9 | 0.824 | 0.500 | 0.164 | 0.060 |

v5 的最好点比 v4 略好：

- success 提升；
- full success 提升；
- collision 明显下降。

但 v5 后期退化严重：

- epoch 20 success 降到 0.608；
- full success 降到 0.200；
- timeout 升到 0.680。

因此，不能使用 latest checkpoint。应使用 epoch 9 best checkpoint。

## 退化分析

训练后期不是撞得更多，而是变得太保守。

分阶段训练 episode 统计：

| 阶段 | succ_n | coll_n | episode_env_steps | mean_lin | yield_priority_reward |
|---|---:|---:|---:|---:|---:|
| early 1-700 | 3.57 | 1.43 | 27.25 | 0.793 | -0.041 |
| mid 701-1300 | 3.67 | 1.32 | 32.27 | 0.730 | -0.051 |
| late >1300 | 4.05 | 0.90 | 82.29 | 0.611 | -0.109 |
| last 300 | 4.08 | 0.81 | 115.92 | 0.549 | -0.139 |

可以看到：

- collision 确实下降；
- 成功 agent 数在训练 episode 里上升；
- 但 episode 变得越来越长；
- mean linear velocity 明显下降；
- yield reward 的影响越来越大；
- validation timeout 快速上升。

这说明模型后期学到的主要不是“更会通过”，而是“更保守地等待”。

进一步看 late 阶段 timeout episode：

| 类型 | episode_env_steps | mean_lin | context_neighbors_mean | robot_proximity_reward | yield_priority_reward |
|---|---:|---:|---:|---:|---:|
| full success | 69.96 | 0.687 | 1.55 | -0.354 | -0.104 |
| partial 2-4/5 | 91.28 | 0.555 | 1.67 | -0.534 | -0.113 |
| bad 0-1/5 | 132.25 | 0.315 | 1.83 | -0.632 | -0.126 |
| timeout 300 steps | 300.00 | 0.395 | 0.79 | -0.191 | -0.132 |

关键点：

> timeout case 的 `context_neighbors_mean` 只有 0.79，并不高。

也就是说，很多 timeout 不是因为一直被多车堵住，而是危险解除后仍然不够积极地走。

## 结论

v5 的方向是有价值的，但 reward 比例不平衡。

它解决了一部分碰撞问题，但引入了严重保守性：

- 会让；
- 会慢；
- 但安全后恢复不够强；
- 长期等待副作用太大。

因此，不建议继续 v5 原样训练。

## 下一步建议

建议做 v6，但不是继续加强等待，而是削弱保守副作用：

1. 降低等待相关奖励；
2. 加强安全后恢复前进的奖励；
3. 对“无近邻仍低速/不前进”增加惩罚；
4. 保留 emergency stop，但只在真正危险距离内强约束；
5. 最好先用 epoch 9 best checkpoint 做逐场景评估，确认哪些 case 是 collision 失败、哪些是 timeout 失败。

一句话：

> v4 主要问题是撞；v5 主要问题是等。v6 应该保留 v5 的避撞收益，但把“安全后继续走”拉回来。
