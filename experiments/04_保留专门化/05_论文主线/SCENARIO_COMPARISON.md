# Existing Scenario Comparison

本文整理 2026-07-17 之前已经实际运行的五车场景。它回答“现在手里有什么”，不代表 procedural low/medium/high 已经完成。

## 1. 场景参数

| 参数 | `standard-5` | random dense default | random tight1 / tight2 | fixed moderate cases |
| --- | --- | --- | --- | --- |
| 机器人 | 5 | 5 | 5 | 5 |
| scenario mode | `standard` | `dense` | `dense` | `curriculum` |
| 起点 | `[-4.5, 4.5]^2` 随机 | `[-2.0, 2.0]^2` 随机 | 完整参数丢失 | 每个 case 固定 |
| 名义采样面积 | `81 m^2` | `16 m^2` | 未知 | 不适用 |
| 名义空间密度 | `0.062 robot/m^2` | `0.313 robot/m^2` | 未知 | 不适用 |
| 起点最小车距 | `0.95 m` | 配置 `0.9 m`，实际至少 `1.2 m` | 未知 | `1.221-1.562 m` |
| goal offset | x `[-2.2,2.2]`, y `[-2.4,2.4]` | x/y `[-1.2,1.2]` | 未知 | 每个 case 固定 |
| 单车任务距离 | 参数 `[0.8,3.5] m`，几何上约至 `3.26 m` | `[0.6,1.7] m` | 未知 | 实际 `[0.304,3.206] m` |
| 目标间距 | 从 `0.85 m` 开始，失败时可放宽至 `0.45 m` | 从 `0.8 m` 开始，可放宽至 `0.48 m` | 未知 | `1.040-1.160 m` |
| 目标与任意机器人间距 | `0.75 m` | `0.8 m` | 未知 | 由固定 case 决定 |
| 额外箱子 | 每次 reset 随机 4 个 | 每次 reset 随机 4 个 | 每次 reset 随机 4 个 | 无额外箱子，仅静态地图墙体 |
| 交互设计 | 无，只随机出现 | 无，只缩小采样区域 | 无，只改变随机采样压力 | 人工交叉、汇流、逆向通道、weave |

说明：

- “名义采样面积”是起点采样方形面积，不是扣除墙体和障碍后的 `A_free`，不能直接作为论文中的严格 spatial density。
- standard 和 random dense 都让初始朝向大致指向目标，并加入约 `+-0.2 rad` 扰动。
- standard 的四个随机箱子在 `[-6,6]^2` 内采样，并与机器人/目标保持 `1.0 m`；random dense 的箱子保持 `2.0 m`。
- tight1/tight2 的环境变量没有进入日志、state 或 shell history，无法恢复完整配置，禁止作为正式可复现实验。

## 2. 5D 现有结果

| 场景 | episodes / seed | outcome 口径 | agent success | collision | unresolved | full success | mean episode steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `standard-5` | `1000 / 1000` | 修复后互斥 | `0.8816` | `0.0990` | `0.0194` | `0.5690` | `70.156` |
| random dense default | `120 / 0` | 旧口径 | `0.978` | `0.020` | `0.002` | `0.917` | `24.642` |
| random tight1 | `120 / 0` | 旧口径 | `0.873` | `0.128` | `0.003` | `0.658` | `28.542` |
| random tight2 | `120 / 0` | 旧口径 | `0.802` | `0.203` | `0.003` | `0.483` | `36.792` |
| fixed moderate cases | `120 / 0` | 修复后互斥 | `0.488` | `0.510` | `0.002` | `0.033` | `40.558` |

不能把这些行当作正式横向排名：

- standard 使用 1000 episodes 和 seed 1000，其余只有 120 episodes 和 seed 0。
- 三个 random dense 使用旧 outcome 统计，success 和 collision 可能重叠；例如 tight1 三类计数为 `524 + 77 + 2 = 603 > 600`。
- standard 与 random dense 的任务距离、采样面积和箱子 clearance 不同。
- fixed moderate 没有随机箱子，且任务距离分布没有与 standard/random dense 匹配。

因此目前只能得出趋势结论：random default 的高成功率主要受短任务影响；tight1/tight2 增大了空间压力；fixed moderate 的同步路径冲突远比单纯缩小空间更难。

## 3. Fixed Moderate 细分

| case | 五车任务距离均值 | 任务距离范围 | 最小起点间距 | 最小目标间距 | agent success | collision | unresolved | full success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `moderate_offset_pair_cross_with_side_pressure` | `1.772` | `[0.304,3.206]` | `1.503` | `1.059` | `0.650` | `0.350` | `0.000` | `0.083` |
| `moderate_three_agent_staggered_merge` | `2.273` | `[1.050,3.148]` | `1.453` | `1.118` | `0.292` | `0.708` | `0.000` | `0.000` |
| `moderate_wall_channel_reverse_with_open_side` | `2.012` | `[1.166,2.984]` | `1.562` | `1.040` | `0.708` | `0.283` | `0.008` | `0.083` |
| `moderate_cluster_release_wide_goals` | `2.184` | `[1.163,3.132]` | `1.221` | `1.077` | `0.542` | `0.458` | `0.000` | `0.000` |
| `moderate_offset_weave_three_plus_retention` | `2.259` | `[0.985,3.150]` | `1.350` | `1.160` | `0.250` | `0.750` | `0.000` | `0.000` |

每个 case 在 120 episodes 的轮转测试中约出现 24 次。上述结果来自修复后的碰撞优先互斥口径。

第一条 case 中有一台 retention robot 的任务距离只有 `0.304 m`，几乎等于 `0.3 m` 到达阈值。这是明显的 case-specific 设计，不适合作为统一难度训练样本。

## 4. 当前定位

| 场景 | 后续角色 |
| --- | --- |
| `standard-5` | generalist retention baseline |
| random dense default | spatial-density 机制诊断 |
| random tight1 / tight2 | 不可复现探索记录，不进入论文表格 |
| fixed moderate cases | canonical held-out interaction tests |
| procedural low/medium/high | 尚未实现；未来正式训练和 density sweep 数据 |

正式 low/medium/high 必须统一任务距离分布、静态/随机障碍条件和安全间距，只按 synchronized conflict graph 改变 interaction density。

## 5. 数据位置

- standard 1000-episode 结果：[D3 generalist baseline](results/D3_generalist_baseline/README.md)
- random/fixed dense 日志与诊断：[dense definition diagnostics](../03_dense专家训练/logs/test/dense_definition_20260716/README.md)
- fixed moderate case 坐标：`experiments/02_课程学习/cases/stage4_asym_dense_5_moderate_cases.json`
