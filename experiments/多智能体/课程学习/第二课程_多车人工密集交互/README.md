# 第二课程：多车人工密集交互

## 目的

第二课程从第一课程的单车 best warm-start，不再继续单独补墙边 case，而是让同一个 policy 开始处理多车交错、靠近、会车、目标区域聚集等交互压力。

这一步要回答两个问题：

- 第一课程学到的单车局部能力迁移到多车后会不会重新出现左右摇摆、局部停滞或撞墙。
- 多车失败主要来自交互压力，还是仍然来自单车墙边局部导航缺陷。

## 当前阶段

第二课程内部采用难度递进，而不是再拆出很多新的大课程。

| 阶段 | 中文说明 | 状态 | 说明 |
| --- | --- | --- | --- |
| `stage2_pairwise_diagnostic` | 诊断：双车交互拆解 | running | 已测 `stage1g best`，下一步测双车预热 best，确认预热是否带来提升。 |
| `stage2_pre_pairwise_warmup` | 预热：双车基础交互 | completed / weak | 2 车会车、交叉、同向超车、目标区轻聚集；未形成稳定提升。 |
| `stage2a_manual_dense_crossing` | 正式：三车人工密集交互 | paused | 直接从第一课程进入该阶段过难，先降回预热。 |

## 直接三车密集尝试

`stage2a_manual_dense_crossing` 从 `stage1g best` warm-start 后，前 4 个 eval 没有形成上升趋势：

| epoch | success_rate | collision_rate | full_success_rate | timeout_episode_rate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.521 | 0.389 | 0.208 | 0.125 |
| 2 | 0.465 | 0.521 | 0.125 | 0.083 |
| 3 | 0.403 | 0.590 | 0.125 | 0.062 |
| 4 | 0.507 | 0.486 | 0.083 | 0.083 |

结论：直接进入三车密集交互跨度太大，当前结果作为“过难尝试”保留，不作为主线继续训练。

日志：

- `logs/too_hard/train_multi_curriculum_stage2a_manual_dense_crossing_detached_20260605_152613.log`

## 预热训练口径

- agents: 2
- case file: `../cases/stage2_pre_pairwise_warmup_cases.json`
- warm-start: `TD3_velodyne_multi_v4_curriculum_stage1g_collision_guard_from_stage1f_best`
- actor lr: `0.00004`
- critic lr: `0.00004`
- exploration noise: `0.045`
- exploration min: `0.015`
- max epochs: 6
- eval episodes: 48
- local-navigation reward: on
- dynamic interaction reward: light `interaction_only`

这里的交互 reward 只作为轻量防撞压力，不改变 actor 输入维度；目标是先让模型在简单双车交互中学会减速、绕开、错峰接近目标，再回到三车人工密集。

## 双车预热结果

`stage2_pre_pairwise_warmup` 从 `stage1g best` warm-start，跑满 6 个 epoch。best checkpoint 在 epoch 1，后续没有继续变好：

| epoch | success_rate | collision_rate | full_success_rate | timeout_episode_rate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.719 | 0.292 | 0.583 | 0.021 |
| 2 | 0.583 | 0.427 | 0.396 | 0.042 |
| 3 | 0.510 | 0.531 | 0.354 | 0.000 |
| 4 | 0.604 | 0.427 | 0.500 | 0.021 |
| 5 | 0.635 | 0.396 | 0.500 | 0.000 |
| 6 | 0.604 | 0.396 | 0.458 | 0.042 |

判断：双车预热比直接三车密集容易，但仍没有学成稳定交互策略，collision 仍然偏高。继续用当前 reward 和 case 续训的收益不明确，下一步应先复盘失败 case 或降低交互 reward 干扰，再决定是否重新跑预热。

日志：

- valid: `logs/train/train_multi_curriculum_stage2_pre_pairwise_warmup_detached_20260605_170738.log`
- failed startup: `logs/failed/train_multi_curriculum_stage2_pre_pairwise_warmup_detached_20260605_170534.log`

## 双车诊断

双车预热里 collision 仍偏高，因此下一步先测、不训练。诊断集会把双车交互拆成同向、轻交叉、正面对穿和目标区合流，分别测试 `stage1g best` 与双车预热 best。

case file: `../cases/stage2_pairwise_diagnostic_cases.json`

### `stage1g best` 结果

测试 160 个 episode，每个 case 10 次，模型为 `TD3_velodyne_multi_v4_curriculum_stage1g_collision_guard_from_stage1f_best`。

整体结果：

| 指标 | 数值 |
| --- | ---: |
| agent success | 222 / 320 = 0.694 |
| agent collision | 70 / 320 = 0.219 |
| agent unresolved | 28 / 320 = 0.088 |
| full success | 88 / 160 = 0.550 |
| timeout episode | 21 / 160 = 0.131 |

case 结果：

| case | success | collision | full success | timeout | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `goal_merge_close_center` | 1.000 | 0.000 | 1.000 | 0.000 | 稳定 |
| `goal_merge_easy_a` | 0.950 | 0.000 | 0.900 | 0.100 | 基本可用 |
| `goal_merge_easy_b` | 1.000 | 0.000 | 1.000 | 0.000 | 稳定 |
| `goal_merge_wall_a` | 0.150 | 0.000 | 0.000 | 1.000 | 贴墙目标汇合失败 |
| `head_on_offset_lane` | 0.650 | 0.300 | 0.600 | 0.100 | 不稳定 |
| `head_on_symmetric_center` | 0.650 | 0.250 | 0.500 | 0.200 | 不稳定 |
| `head_on_wall_offset` | 0.800 | 0.200 | 0.700 | 0.000 | 可用但需防撞 |
| `head_on_wall_side` | 0.900 | 0.100 | 0.800 | 0.000 | 基本可用 |
| `offset_cross_easy_a` | 0.700 | 0.100 | 0.500 | 0.400 | 交叉等待失败 |
| `offset_cross_easy_b` | 0.600 | 0.400 | 0.400 | 0.000 | 交叉碰撞偏高 |
| `perpendicular_cross_center` | 0.650 | 0.350 | 0.500 | 0.000 | 交叉碰撞偏高 |
| `perpendicular_cross_staggered` | 0.500 | 0.350 | 0.300 | 0.300 | 交叉不稳定 |
| `same_direction_far_lane_a` | 0.700 | 0.300 | 0.600 | 0.000 | 简单双车也有碰撞 |
| `same_direction_far_lane_b` | 1.000 | 0.000 | 1.000 | 0.000 | 稳定 |
| `same_direction_overtake_light` | 0.400 | 0.600 | 0.000 | 0.000 | 同向追越失败 |
| `same_direction_wall_parallel` | 0.450 | 0.550 | 0.000 | 0.000 | 靠墙并行失败 |

结论：

- `stage1g best` 不是不能迁移到双车，普通目标汇合、部分会车可以完成。
- 失败集中在“同向接近时的速度协调”、“交叉路口的先后顺序”、“贴墙目标附近的等待/绕行”。
- 第二课程不应直接推进三车密集，也不应继续泛泛训练双车随机；应该把上述三类失败作为核心样本。
- 下一步先用同一诊断集测试双车预热 best。如果双车预热 best 没有明显改善，说明当前预热设置没有学到有效交互语义，需要重做预热设置。

日志：

- `logs/test/test_multi_curriculum_stage2_pairwise_diagnostic_TD3_velodyne_multi_v4_curriculum_stage1g_collision_guard_from_stage1f_best_detached_20260605_200327.log`

## 运行命令

```bash
scripts/start_training_detached_multi_curriculum.sh stage2_pre_pairwise_warmup
```

预热稳定后再回到三车密集：

```bash
DRL_MULTI_LOAD_MODEL_NAME=TD3_velodyne_multi_v4_curriculum_stage2_pre_pairwise_warmup_from_stage1g_best \
scripts/start_training_detached_multi_curriculum.sh stage2a_manual_dense_crossing
```
