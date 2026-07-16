# Dense Definition Diagnostics - 2026-07-16

本目录归档固定人工 case 与随机 dense 采样的 `5D` 诊断结果。目的不是比较最终算法，而是确认“空间密度”和“交互难度”不是同一个概念。

## 结果

| 场景 | 口径 | agent success | collision | unresolved | full success |
| --- | --- | ---: | ---: | ---: | ---: |
| random default | 旧口径 | 0.978 | 0.020 | 0.002 | 0.917 |
| random tight1 | 旧口径 | 0.873 | 0.128 | 0.003 | 0.658 |
| random tight2 | 旧口径 | 0.802 | 0.203 | 0.003 | 0.483 |
| fixed moderate cases | 修复后互斥口径 | 0.488 | 0.510 | 0.002 | 0.033 |

固定 moderate case 的逐 case 结果：

| case | agent success | collision | unresolved | full success |
| --- | ---: | ---: | ---: | ---: |
| `moderate_offset_pair_cross_with_side_pressure` | 0.650 | 0.350 | 0.000 | 0.083 |
| `moderate_wall_channel_reverse_with_open_side` | 0.708 | 0.283 | 0.008 | 0.083 |
| `moderate_cluster_release_wide_goals` | 0.542 | 0.458 | 0.000 | 0.000 |
| `moderate_three_agent_staggered_merge` | 0.292 | 0.708 | 0.000 | 0.000 |
| `moderate_offset_weave_three_plus_retention` | 0.250 | 0.750 | 0.000 | 0.000 |

## 解释

- random default/tight1/tight2 主要改变机器人和目标的随机采样范围，体现空间占用压力。
- fixed moderate cases 主动制造同步交叉、合流、逆向通道和路径占用，体现交互冲突压力。
- tight2 的 full success 已降到约一半，但 agent success 仍有 0.802；固定 case 则同时把 agent success 压到 0.488。这说明人工 case 的困难主要来自路径冲突结构，不只是机器人靠得更近。
- 后续论文不应把二者都笼统称为 dense。应分别称为 `spatial density` 和 `interaction density`。

## 可复现性限制

四组测试均使用 `5D`、5 个机器人、120 episodes、seed 0。

random default 的配置可从提交和当前脚本恢复：

```text
start x/y range: [-2.0, 2.0]
goal x/y offset: [-1.2, 1.2]
goal distance: [0.6, 1.7]
nominal robot clearance: 0.9m
goal clearance: 0.8m
```

tight1/tight2 的启动环境变量没有写入日志、state 文件或 shell history，无法完整恢复。它们只能作为探索性诊断，不能直接作为论文正式表格。正式重跑前，测试程序必须把完整 scenario manifest 写入日志和结果文件。

旧 random 日志还使用 success/collision 可重叠的旧统计口径，因此可能出现三类计数之和大于 600。fixed moderate metric-fix 日志使用碰撞优先的互斥口径。旧、新绝对数值不能直接做严格显著性比较，但 full-success 难度趋势仍可用于场景定义诊断。

## 文件

- `test_dense5_random_5d_20260716_220706.log`
- `test_dense5_random_tight1_5d_20260716_223139.log`
- `test_dense5_random_tight2_5d_20260716_225203.log`
- `test_stage4_asym_dense_5_moderate_5D_METRICFIX_20260716_234501.log`
