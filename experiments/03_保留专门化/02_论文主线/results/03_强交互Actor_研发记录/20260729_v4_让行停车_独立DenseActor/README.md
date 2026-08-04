# 20260729 v4：让行/停车奖励下的独立 Dense Actor

## 目的

训练一个可以独立完成 dense / 强交互场景的 actor。

约束：

- 从 5A actor warm-start；
- actor 独立控制完整 episode；
- 不使用 5A/5D runtime takeover；
- 不使用 oracle rollout；
- 不做 actor switching；
- 保持 TD3 主体结构。

## 本次改动

在 v3 的减速约束基础上，增加了更明确的多车交互奖励：

- 进入危险距离时鼓励停车；
- 进入危险距离时惩罚继续前进；
- 两车接近时，离目标更远的一方作为低优先级车；
- 低优先级车继续向前走会被惩罚；
- 低优先级车低速等待会被奖励。

主要代码：

- `TD3/multi_agent_velodyne_env.py`
- `TD3/train_velodyne_td3_multi.py`
- `TD3/actor_objectives.py`
- `scripts/start_training_independent_dense_actor_from_5a.sh`

## 训练配置

- 模型名：`independent_dense_actor_from_5a_yield_v4_s20260729`
- 初始模型：`TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best`
- 训练集：`datasets/fixed_v1/dense/train.json.gz`
- validation：`datasets/fixed_v1/views/dense_validation_monitor_ultrafast_v3/validation.json.gz`
- validation episodes：50
- best metric：`full_success`

## 结果

| epoch | success | full success | collision | timeout |
|---|---:|---:|---:|---:|
| 1 | 0.736 | 0.360 | 0.264 | 0.000 |
| 4 | 0.772 | 0.400 | 0.224 | 0.020 |
| 6 | 0.792 | 0.420 | 0.204 | 0.020 |
| 7 | 0.796 | 0.480 | 0.204 | 0.000 |
| 8 | 0.784 | 0.440 | 0.212 | 0.020 |
| 9 | 0.792 | 0.440 | 0.196 | 0.060 |
| 10 | 0.744 | 0.400 | 0.236 | 0.080 |

最好结果出现在 epoch 7。

best checkpoint：

`checkpoints/independent_dense_actor_from_5a_yield_v4_s20260729_best.pt`

latest checkpoint：

`checkpoints/independent_dense_actor_from_5a_yield_v4_s20260729_latest.pt`

训练日志：

`logs/archive/rejected/independent_dense_actor/train_independent_dense_actor_from_5a_yield_v4_s20260729_20260729_200520.log`

## 结论

这次训练有一定效果，但不值得继续原样长跑。

原因：

1. epoch 7 达到最好；
2. epoch 8-10 没有继续提升；
3. epoch 10 明显退化；
4. timeout 开始增加；
5. 失败原因已经不只是高速碰撞。

最近训练 episode 的严重失败 case 中，平均线速度已经明显下降：

| 类型 | mean_lin | context_neighbors_mean | robot_proximity_reward | mean_reward |
|---|---:|---:|---:|---:|
| full success | 0.789 | 1.74 | -0.403 | 112.5 |
| partial 2-4/5 | 0.674 | 1.93 | -0.677 | 28.5 |
| bad 0-1/5 | 0.378 | 2.17 | -1.087 | -65.5 |

这说明策略不是完全不会减速。

真正问题更像是：

> 多车密集交互时，actor 会变慢，但不会稳定决定谁先过、谁等待、什么时候重新启动、怎么绕开。

所以单纯继续加大减速/停车惩罚，可能会让模型更保守，并增加 timeout，而不一定减少碰撞。

## 场景难度检查

本次 validation 并没有明显比 train 更难。

简单几何统计：

| split | hard 比例 | extreme 比例 |
|---|---:|---:|
| train | 46.1% | 28.7% |
| validation monitor | 48.0% | 32.0% |
| dense validation 全集 | 45.4% | 30.4% |

因此，不能简单说“validation 太难”。更合理的判断是：

> 当前 actor 对 dense 强交互场景的能力还不稳定，尤其是多车链式冲突。

## 下一步建议

不要继续 v4 原样训练。

建议下一步先做分析型验证：

1. 用 epoch 7 best checkpoint 跑一次带 scenario_id 的验证；
2. 记录每个场景的 success/collision/timeout；
3. 把场景分成 normal dense / hard dense / stress dense；
4. 看失败是否集中在少量 stress case；
5. 再决定是：
   - 从训练集中移除极端 stress case；
   - 或者单独建立 stress test；
   - 或者继续改 reward，让 actor 学会多车通行顺序。

当前不建议直接开启新训练。
