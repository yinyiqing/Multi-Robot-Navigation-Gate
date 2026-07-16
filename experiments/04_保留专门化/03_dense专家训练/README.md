# 03 dense 专家训练

状态：`diagnostic / failed training archive`。本目录记录 dense 定义诊断、full fine-tune 失败和 residual 代码脚手架；正式训练分布尚未实现，因此当前禁止从本目录启动训练。

后续目标是训练 interaction-dense residual specialist。开始条件和评测协议以 `../05_论文主线/README.md` 的 D1-D5 决策门为准。

## 历史 fixed moderate 定义

此前 full fine-tune 使用的场景不是旧 hard set，也不是 pair/three 诊断集，而是：

```bash
stage4_asym_dense_5_moderate
```

case 文件：

```bash
experiments/02_课程学习/cases/stage4_asym_dense_5_moderate_cases.json
```

这个场景的定义：

- 5 车在同一个有墙世界里同时运行；
- 2-3 个主交互智能体产生会车、交叉、合流；
- 2-3 个压力/保持智能体让场景保持 dense，但不强迫五车全挤同一个点；
- 起点不贴脸，目标不扎堆；
- 每个 case 有 1-3 组直线路径交叉；
- 目标是“可学习的密集交互”，不是一上来做病态压力测试。

几何约束：

| 指标 | 目标 |
| --- | --- |
| 最小起点距离 | `>= 1.2m` |
| 最小目标距离 | `>= 1.0m` |
| 直线路径交叉 | 每个 case `1-3` 组 |
| 主交互智能体 | `2-3` 个 |
| 压力/保持智能体 | `2-3` 个 |

## 和 pair / three 的区别

`stage3_asym_pair_5` 和 `stage3_asym_three_5` 是局部冲突诊断：主要看两车或三车冲突，其余车基本是背景。

`stage4_asym_dense_5_moderate` 是 dense 专家训练环境：五车都在场，主冲突仍然是 2-3 车，但旁边的车会形成空间压力和路径占用。它比 pair/three 更接近 dense，但比旧 hard set 更可学。

## 历史 full fine-tune 脚本

```bash
scripts/start_training_detached_dense5_moderate_geo_critic_from_5d.sh
```

默认设置：

- warmstart：`5D` actor；
- 场景：`stage4_asym_dense_5_moderate`；
- critic：几何邻域 local critic；
- reward：`average_plus_interaction`；
- 日志：先放仓库根目录 `logs/`，跑完后再归档到本目录。

## 已知失败诊断

旧 hard dense set 太急，固定 baseline 本身就不高：

| 模型 | 旧 hard dense set | success | collision | full |
| --- | --- | ---: | ---: | ---: |
| `5D` | `stage4_asym_dense_5_bridge` | 0.540 | 0.475 | 0.250 |
| `5A` | `stage4_asym_dense_5_bridge` | 0.530 | 0.475 | 0.275 |

之前的训练现象：

- `5D` full actor fine-tune 容易退化；
- `5A` full actor fine-tune 不明显退化，但也没有提升；
- head-only 能稳住，但不足以形成 dense 专家；
- 所以当前先把 dense 场景定义降到“中等密度、可学习”，再评估训练是否真的有改进。

## moderate dense baseline

固定 `5D` 在 `stage4_asym_dense_5_moderate` 上的 120 episodes 测试：

| 模型 | 场景 | success | collision | unresolved | full |
| --- | --- | ---: | ---: | ---: | ---: |
| `5D` | `stage4_asym_dense_5_moderate` | 0.513 | 0.502 | 0.002 | 0.058 |

按 case：

| case | success | collision | full |
| --- | ---: | ---: | ---: |
| `moderate_offset_pair_cross_with_side_pressure` | 0.708 | 0.292 | 0.208 |
| `moderate_wall_channel_reverse_with_open_side` | 0.700 | 0.317 | 0.083 |
| `moderate_cluster_release_wide_goals` | 0.583 | 0.417 | 0.000 |
| `moderate_three_agent_staggered_merge` | 0.308 | 0.742 | 0.000 |
| `moderate_offset_weave_three_plus_retention` | 0.267 | 0.742 | 0.000 |

判断：

- 总体难度合适：固定 `5D` 不是 0.8-0.9，也不是 0.2-0.3；
- 碰撞率仍然高，说明它确实测到了 dense 交互短板；
- 后两个 case 偏硬，训练时要监控它们是否继续拖垮整体。

## full fine-tune 结果

`5D -> moderate dense` 的 full actor fine-tune 已确认失败：

| epoch | success | collision | full |
| ---: | ---: | ---: | ---: |
| 1 | 0.485 | 0.525 | 0.025 |
| 2 | 0.510 | 0.515 | 0.075 |
| 3 | 0.450 | 0.555 | 0.000 |
| 4 | 0.395 | 0.595 | 0.000 |
| 5 | 0.370 | 0.600 | 0.000 |
| 6 | 0.295 | 0.695 | 0.000 |
| 7 | 0.295 | 0.690 | 0.000 |
| 8 | 0.230 | 0.765 | 0.000 |

结论：

- full actor 解冻继续破坏已有导航能力；
- 最好结果只在 epoch 2 贴近 baseline，随后持续退化；
- 这条训练产物已删除，只保留失败日志作为诊断。

下一步不要继续 full actor fine-tune。更合理的尝试是冻结 actor 主干，只训练一个很小的 residual/head adapter，或者先只训练 critic/表示层，再用保守策略更新 actor。

## residual 实现

当前已实现第一版最小 residual：

```text
frozen 5D action + 0.15 * tanh(adapter(state))
adapter: 24 -> 128 -> 2
```

最后一层零初始化，因此训练开始前的动作与固定 `5D` 完全一致。基础 Actor 参数不参与反向传播，checkpoint 同时保存基础 Actor、residual 参数和 residual scale。

预留训练入口（D1-D3 完成前禁止执行）：

```bash
scripts/start_training_detached_dense5_moderate_residual_from_5d.sh
```

预留测试入口：

```bash
scripts/start_test_detached_dense5_moderate_residual_best.sh
scripts/start_test_detached_standard5_residual_best.sh
```

第一版只使用 individual reward，保留训练期几何邻域 Critic。只有 moderate dense 提升且 standard_5 不明显下降，才能认为 residual 专门化成立。

## dense 定义诊断

2026-07-16 对固定人工 case 和三档随机 dense 采样做了对照，归档见：

```text
logs/test/dense_definition_20260716/README.md
```

关键结论是：随机 tight2 的 agent success / full success 约为 `0.802 / 0.483`，而固定 moderate case 在修复口径下约为 `0.488 / 0.033`。固定 case 测到的是同步路径冲突，远比单纯缩小随机采样区域更强。后续需要分别定义空间密度与交互密度，固定 case 更适合作为 held-out stress test，不应作为唯一训练分布。
