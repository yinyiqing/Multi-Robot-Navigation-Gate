# 20260729 v3 独立 Dense Actor：减速约束实验

## 目的

验证“保持一个独立 Dense Actor，仅在训练 loss 中加入危险接近减速约束”是否能解决强交互场景中的连环碰撞问题。

## 设置

- warm start：5A actor only
- rollout：单 actor 独立控制完整 episode
- oracle/gate：关闭
- local critic context：ego_motion
- 新增约束：`DRL_MULTI_ACTOR_SLOWDOWN_SAFETY_WEIGHT=3.0`
- 危险接近速度上限：`DRL_MULTI_ACTOR_SLOWDOWN_MAX_LINEAR_ACTION=-0.4`
- 训练集：fixed_v1 dense train，cycle
- validation：dense_validation_monitor_ultrafast_v3，50 episodes

## 结果

| epoch | success | full_success | collision | timeout |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.752 | 0.420 | 0.248 | 0.000 |
| 2 | 0.744 | 0.380 | 0.256 | 0.000 |
| 3 | 0.752 | 0.340 | 0.248 | 0.000 |
| 4 | 0.744 | 0.400 | 0.256 | 0.000 |
| 5 | 0.768 | 0.420 | 0.232 | 0.000 |
| 6 | 0.772 | 0.400 | 0.224 | 0.020 |
| 7 | 0.764 | 0.380 | 0.236 | 0.000 |
| 8 | 0.768 | 0.360 | 0.220 | 0.060 |

## 结论

减速约束有效地抑制了 v2 中的“危险状态继续加速”漂移，但没有显著提高 full success。失败仍主要是碰撞型失败，尤其是多车同时进入冲突区后的连环碰撞。

`success=0/5` 或 `1/5` 的共性：

- 平均 collision 为 4.12/5
- `context_neighbors_max` 基本为 4
- `active_neighbor_step_rate` 高，长期处于多车交互
- `robot_proximity_reward` 很负，说明近距离压迫强
- 既有高速撞，也有低速撞；仅靠“慢下来”不足以学会“谁让谁”

## 下一步

停止继续堆 epoch。下一版应加入“危险停车 + 让行优先级”的训练信号：

- 危险距离内，低优先级车继续前进要重罚，停车/低速等待给正向信号；
- 优先级使用目标剩余距离：离目标远者让，离目标近者先过；
- 不做运行时接管，仍训练一个独立 actor。

## 归档文件

- `logs/archive/rejected/independent_dense_actor/train_independent_dense_actor_from_5a_full_v3_s20260729_20260729_171813.log`
- `checkpoints/independent_dense_actor_from_5a_full_v3_s20260729_best.pt`
- `checkpoints/independent_dense_actor_from_5a_full_v3_s20260729_latest.pt`
