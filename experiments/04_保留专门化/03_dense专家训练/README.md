# 03 dense 专家训练

当前目标：先训练一个真正适合 dense 情景的第二专家 actor。只有当 dense 专家和普通专家都足够稳定后，再冻结两个 actor，单独训练门控网络。

## 当前 dense 定义

当前训练入口不是旧的 hard set，也不是 pair/three 诊断集，而是：

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

## 当前训练脚本

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

## 下一步

先测固定 `5D` 在 `stage4_asym_dense_5_moderate` 上的 baseline。只有 baseline 落在合理区间，才开始训练 dense 专家。
