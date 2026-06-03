# Stage 1c wall-clearance 失败诊断

## 口径

- 训练模型：`TD3_velodyne_multi_v4_curriculum_stage1c_wall_clearance`
- warm-start：`TD3_velodyne_multi_v4_curriculum_stage1b_single_best`
- 训练 case：`../cases/stage1c_wall_clearance_cases.json`
- 后续诊断 case：`../cases/stage1d_wall_feasible_cases.json`
- 机器人数量：1
- 主要日志：
  - `train_multi_curriculum_stage1c_wall_clearance_detached_20260603_094124.log`
  - `test_multi_curriculum_stage1d_wall_feasible_TD3_velodyne_multi_v4_curriculum_stage1b_single_best_detached_20260603_195224.log`

## Stage 1c 训练结果

Stage 1c 使用 wall-clearance shaping，希望减少侧墙平行通过时的碰撞。但训练结果没有改善，后期转为停滞。

| epoch | success_rate | collision_rate | unresolved_rate | timeout_episode_rate | avg_final_distance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.104 | 0.250 | 0.646 | 0.646 | 2.266 |
| 2 | 0.042 | 0.333 | 0.625 | 0.625 | 1.991 |
| 3 | 0.104 | 0.271 | 0.646 | 0.646 | 1.729 |
| 4 | 0.042 | 0.292 | 0.667 | 0.667 | 1.667 |
| 5 | 0.062 | 0.312 | 0.625 | 0.625 | 0.710 |
| 6 | 0.062 | 0.271 | 0.667 | 0.667 | 0.712 |
| 7 | 0.125 | 0.208 | 0.667 | 0.667 | 0.668 |
| 8 | 0.062 | 0.229 | 0.708 | 0.708 | 0.968 |
| 9 | 0.062 | 0.250 | 0.688 | 0.688 | 0.884 |
| 10 | 0.000 | 0.083 | 0.917 | 0.917 | 0.787 |

## 失败原因判断

Stage 1c 的 reward shaping 和部分 case 几何存在冲突：过近的侧墙目标线要求机器人贴墙通过，而 wall-clearance shaping 又持续惩罚贴墙行为，容易把策略推向保守或停滞。

因此 Stage 1c 不应继续训练，也不应作为后续 warm-start。

## Stage 1d 可行侧墙诊断

为排除“随机箱子”和“过紧目标线”干扰，新增 `stage1d_wall_feasible_cases.json`，固定无箱子，只保留几何上更可行的侧墙平行通过 case。

使用 `TD3_velodyne_multi_v4_curriculum_stage1b_single_best` 测试，提前停止于 26 episodes。

| episodes | success | collision | unresolved | timeout |
| ---: | ---: | ---: | ---: | ---: |
| 26 | 0 | 9 | 17 | 17 |

前 20 episodes 的滚动统计：

| success_rate | collision_rate | unresolved_rate | timeout_episode_rate |
| ---: | ---: | ---: | ---: |
| 0.000 | 0.600 | 0.400 | 0.400 |

## 结论

当前单车 policy 没有稳定学会侧墙平行通过。这个缺陷在五车场景中会被放大，但根因不在五车 reward 或 critic。

后续不建议继续同类 reward shaping。更合理的下一步是引入导航先验，把“直接朝最终目标走”改成“先跟随可行路径点/子目标，再到最终目标”，至少先在单车侧墙 case 上验证。
