# 01 冲突验证

这一步先不急着做切换。

先回答一个最关键的问题：

**单一 actor 继续往密集场景训练时，会不会破坏它原来已经学好的普通导航能力。**

## 现在要验证什么

一句话：

**普通能力和 dense 能力不能只靠继续硬训同一个 actor 来兼顾。**

## 第一轮先用现成模型

先不训练新模型，先拿已经有的模型做交叉测试。

### A. 普通 actor 候选 1

- `TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best`

这是现在最自然的普通五车 actor。
优点是它和第三课程真实 warm start 的来源一致。

### B. 普通 actor 候选 2

- `TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best`

这是课程学习里带几何邻域 critic 的五车普通模型。
虽然它最终没有超过 `5A best`，但它更适合作为“带邻域 critic 的普通能力候选”。

### C. 密集 actor

- `TD3_velodyne_multi_v4_curriculum_stage3_asym_pair_5_from_5a_cleanstart_v2_best`

这是现在最自然的密集 actor 候选。

### D. 继续训练后的单一 actor

先选一版已经继续往 `stage3_asym_three_5` 训练过的模型，作为 overwrite 路线代表。

第一轮先看：

- `TD3_velodyne_multi_v4_curriculum_stage3_asym_three_5_joint_action_critic_midcheck_from_stage3_pair_cleanstart_v2_best_best`

后面如果需要，再补别的 `stage3_asym_three_5` 版本做对照。

## 当前测试口径

- `standard_5`
  - 普通场景主测试
- `stage3_asym_three_5`
  - dense 场景主测试
- `stage2_dense`
  - 只作为压力测试，不作为当前主 benchmark

## 现在的简单判断标准

- `5A` 在 `standard_5` 更稳
- dense actor 在 `stage3_asym_three_5` 应该更强
- 如果继续训练后的单一 actor 两边都没赢，就说明 overwrite 路线不划算

## 当前记录

### 当前候选收缩

普通 actor 不再从所有历史五车模型里乱选。

当前只保留 2 个最值得继续看的普通候选：

1. `5A best`
   - `TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best`
2. `5D best`
   - `TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best`

密集 actor 当前改为：

1. `stage3_asym_pair_5 from 5D`
   - `TD3_velodyne_multi_v4_curriculum_stage3_asym_pair_5_from_5d_best`

补充说明：

- 旧 `PAIR(from_5a)` 只保留为历史对照，不再作为优先 dense actor。
- 原因是：`5A` 的交互底子偏弱，作为 dense 专门化 warm start 不够自然。
- 新补做的 `5D -> stage3_asym_pair_5` 已完成 3 epoch：
  - Epoch 1：`0.900 / 0.104 / 0.729`
  - Epoch 2：`0.912 / 0.087 / 0.729`
  - Epoch 3：`0.921 / 0.079 / 0.750`
- 其中 Epoch 3 已明显优于旧 `PAIR(from_5a)`，所以后续主口径切到 `PAIR(from_5d)`。

同时保留一个重要区分：

- `5D` 不是失败候选，而是当前最强的 bridge baseline
- `PAIR(from_5d)` 才是 bridge baseline 继续专门化后的 dense actor
- 因此后面如果做 attention / gate：
  - `5D` 适合作为“未专门化但已具 dense 泛化”的对照
  - `PAIR(from_5d)` 适合作为“当前 dense 主基线”

overwrite actor 当前先保留：

1. `joint_action_critic_midcheck best`
   - `TD3_velodyne_multi_v4_curriculum_stage3_asym_three_5_joint_action_critic_midcheck_from_stage3_pair_cleanstart_v2_best_best`

### 已完成

1. `5A best -> stage3_asym_three_5`
   - 120 episodes
   - `success_rate=0.885`
   - `collision_rate=0.118`
   - `full_success_rate=0.567`
2. `stage3_asym_pair_5 best -> 标准五车`
   - 120 episodes
   - `success_rate=0.885`
   - `collision_rate=0.087`
   - `full_success_rate=0.542`
   - `timeout_episode_rate=0.142`
3. `5D best -> 标准五车`
   - 120 episodes
   - `success_rate=0.882`
   - `collision_rate=0.098`
   - `full_success_rate=0.550`
   - `timeout_episode_rate=0.100`
   - 补充观察：
     - 前 100 episode 一度到过 `full_success_rate=0.600`
     - 最后 20 episode 明显变差
     - 说明 `5D` 在标准五车上没有明显优于 `5A`
4. `5D best -> stage3_asym_three_5`
   - 120 episodes
   - `success_rate=0.902`
   - `collision_rate=0.097`
   - `full_success_rate=0.650`
   - `timeout_episode_rate=0.017`
   - case 观察：
     - `three_cross_main_pair_with_late_third`
       - `success_rate=0.830`
       - `collision_rate=0.170`
       - `full_success_rate=0.475`
     - `three_goal_merge_main_pair_with_outer_third`
       - `success_rate=1.000`
       - `collision_rate=0.005`
       - `full_success_rate=1.000`
     - `three_wall_pair_with_far_third`
     - `success_rate=0.875`
     - `collision_rate=0.115`
     - `full_success_rate=0.475`
5. `THREE_MID -> 标准五车`
   - 120 episodes
   - `success_rate=0.870`
   - `collision_rate=0.115`
   - `unresolved_rate=0.018`
   - `full_success_rate=0.517`
   - `timeout_episode_rate=0.083`
   - 补充观察：
     - 前段一度表现正常
     - 后半程明显变差
     - 最后 10 个 episode 只有 `full_success_rate=0.200`
     - 说明 overwrite 后，普通场景能力确实被破坏了
6. `THREE_MID -> stage3_asym_three_5`
   - 120 episodes
   - `success_rate=0.870`
   - `collision_rate=0.130`
   - `unresolved_rate=0.005`
   - `full_success_rate=0.517`
   - `timeout_episode_rate=0.025`
   - case 观察：
     - `three_cross_main_pair_with_late_third`
       - `success_rate=0.785`
       - `collision_rate=0.210`
       - `full_success_rate=0.275`
     - `three_goal_merge_main_pair_with_outer_third`
       - `success_rate=1.000`
       - `collision_rate=0.010`
       - `full_success_rate=1.000`
     - `three_wall_pair_with_far_third`
       - `success_rate=0.825`
       - `collision_rate=0.170`
       - `full_success_rate=0.275`
   - 补充观察：
     - 不仅普通场景退化
     - dense 表现也没有超过 `5A/5D`
     - 说明 overwrite 路线整体不划算
7. `PAIR(from_5a) -> stage3_asym_three_5`
   - 120 episodes
   - `success_rate=0.900`
   - `collision_rate=0.102`
   - `unresolved_rate=0.000`
   - `full_success_rate=0.642`
   - `timeout_episode_rate=0.000`
   - case 观察：
     - `three_cross_main_pair_with_late_third`
       - `success_rate=0.820`
       - `collision_rate=0.180`
       - `full_success_rate=0.400`
     - `three_goal_merge_main_pair_with_outer_third`
       - `success_rate=0.995`
       - `collision_rate=0.010`
       - `full_success_rate=0.975`
     - `three_wall_pair_with_far_third`
       - `success_rate=0.885`
       - `collision_rate=0.115`
       - `full_success_rate=0.550`
   - 补充观察：
     - dense 上明显强于 `5A`
     - 和 `5D` 非常接近
  - 说明它曾经可以作为专门化 actor 候选，但现在优先级已低于 `PAIR(from_5d)`

8. `5D -> PAIR`
   - 3 epochs
   - Epoch 1：`success_rate=0.900`
   - Epoch 1：`collision_rate=0.104`
   - Epoch 1：`full_success_rate=0.729`
   - Epoch 2：`success_rate=0.912`
   - Epoch 2：`collision_rate=0.087`
   - Epoch 2：`full_success_rate=0.729`
   - Epoch 3：`success_rate=0.921`
   - Epoch 3：`collision_rate=0.079`
   - Epoch 3：`full_success_rate=0.750`
   - 当前判断：
     - 前两轮与旧 `PAIR(from_5a)` 持平
     - 第三轮明显更好
     - 当前应把它作为新的优先 dense actor 候选
   - 但补充说明：
     - 这一条目前还是训练内 eval
     - 在作为后续 attention 基线前，最好先补正式 test

9. `PAIR(from_5d) -> 标准五车`
   - 300 episodes
   - `success_rate=0.891`
   - `collision_rate=0.085`
   - `unresolved_rate=0.023`
   - `full_success_rate=0.573`
   - `timeout_episode_rate=0.107`
   - 当前判断：
     - 明显优于旧 `PAIR(from_5a)` 的标准五车结果
     - 但仍未超过 `5A` 和 `5D`
     - 因此它的定位仍然不是“普通 actor”，而是更健康的 dense specialized actor

10. `PAIR(from_5d) -> stage3_asym_three_5`
   - 120 episodes
   - `success_rate=0.880`
   - `collision_rate=0.122`
   - `unresolved_rate=0.000`
   - `full_success_rate=0.575`
   - `timeout_episode_rate=0.000`
   - 分 case：
     - `three_cross_main_pair_with_late_third`
       - `success_rate=0.770`
       - `collision_rate=0.230`
       - `full_success_rate=0.300`
     - `three_goal_merge_main_pair_with_outer_third`
       - `success_rate=0.985`
       - `collision_rate=0.020`
       - `full_success_rate=0.925`
     - `three_wall_pair_with_far_third`
       - `success_rate=0.885`
       - `collision_rate=0.115`
       - `full_success_rate=0.500`
   - 当前判断：
     - 比 `5D -> dense (0.902 / 0.097 / 0.650)` 更差
     - 也低于旧 `PAIR(from_5a) -> dense (0.900 / 0.102 / 0.642)`
     - 说明 `PAIR(from_5d)` 训练链路更顺，不等于它已经成为当前最强 dense test actor
     - 所以下一步做门控时，dense expert 不能默认直接选它

### 当前横向对比

| 组别 | success_rate | collision_rate | full_success_rate | timeout/unresolved |
| --- | ---: | ---: | ---: | ---: |
| `5A -> 标准五车` | 0.897 | 0.087 | 0.600 | timeout 0.087 |
| `5A -> dense` | 0.885 | 0.118 | 0.567 | timeout 0.008, unresolved 0.002 |
| `5D -> 标准五车` | 0.882 | 0.098 | 0.550 | timeout 0.100 |
| `5D -> dense` | 0.902 | 0.097 | 0.650 | timeout 0.017, unresolved 0.003 |
| `PAIR(from_5a) -> 标准五车` | 0.885 | 0.087 | 0.542 | timeout 0.142, unresolved 0.028 |
| `PAIR(from_5a) -> dense` | 0.900 | 0.102 | 0.642 | timeout 0.000, unresolved 0.000 |
| `PAIR(from_5d) -> 标准五车` | 0.891 | 0.085 | 0.573 | timeout 0.107, unresolved 0.023 |
| `PAIR(from_5d) -> dense` | 0.880 | 0.122 | 0.575 | timeout 0.000, unresolved 0.000 |
| `THREE_MID -> 标准五车` | 0.870 | 0.115 | 0.517 | timeout 0.083, unresolved 0.018 |
| `THREE_MID -> dense` | 0.870 | 0.130 | 0.517 | timeout 0.025, unresolved 0.005 |

### 当前判断

- `5A` 在标准五车上仍然最稳，说明它更像“普通 actor”候选
- `5D` 在 dense 上明显好于 `5A`，但在标准五车上没有超过 `5A`
- 旧 `PAIR(from_5a)` 回到标准五车后，collision 还行，但 `full_success` 和 `timeout` 更差；而它在 dense 上明显更强，更像“偏向密集交互的专门 actor”
- 新 `PAIR(from_5d)` 在标准五车上优于旧 `PAIR(from_5a)`，说明 `5D` 作为 warm start 更合理
- 但它在正式 dense benchmark 上反而低于 `5D` 和旧 `PAIR(from_5a)`，说明“训练更顺”不等于“最终 dense test 更强”
- `THREE_MID` 回到标准五车后比 `5A`、`5D`、`PAIR` 都更差，说明继续覆盖式训练会破坏普通场景能力
- `THREE_MID` 在 dense 上也没有换来更强表现，说明 overwrite 不是值得继续走的方向
- 现在已经能初步看出：
  - `5A` 更偏普通场景
  - `5D` 是当前最稳的 dense/bridge actor
  - `PAIR(from_5d)` 更像一次“专门化尝试”，但还不是最强 dense expert
- `THREE_MID` 说明 overwrite 路线会进一步放大这种偏移，而且收益不够
- 当前最稳妥的门控起点应改成：
  - 普通 actor：`5A`
  - dense actor：`5D`
- `PAIR(from_5d)` 先保留为对照 / 备选 dense actor，不直接升为主搭配

### 简短记录

- `2026-07-07`：`THREE_MID -> 标准五车` 首次测试因旧 ROS/Gazebo 进程残留报错，日志作废，已清环境后从 `episode 1` 重跑。
- `2026-07-07`：`THREE_MID -> 标准五车` 重跑完成，结果差于 `5A/5D/PAIR`，支持“继续覆盖训练会伤到普通能力”这个判断。
- `2026-07-08`：`THREE_MID -> dense` 跑完，dense 上也没有更强，overwrite 路线基本可以先停。
- `2026-07-12`：`PAIR -> dense` 跑完，结果接近 `5D -> dense`，当前可以把 `5A + PAIR` 作为最自然的一组保留-专门化候选。
- `2026-07-13`：`PAIR(from_5d) -> 标准五车` 正式 test 完成，结果优于旧 `PAIR(from_5a)`，但仍未超过 `5A/5D`，支持它作为 dense specialized actor，而不是普通 actor。
- `2026-07-13`：`PAIR(from_5d) -> stage3_asym_three_5` 正式 test 完成，结果为 `0.880 / 0.122 / 0.575`，低于 `5D` 和旧 `PAIR(from_5a)`；因此门控第一版应优先用 `5A + 5D`，`PAIR(from_5d)` 作为对照保留。
