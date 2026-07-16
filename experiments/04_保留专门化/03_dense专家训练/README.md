# 03 dense 专家训练

这里放当前主线：训练第二个 dense 专家 actor。

## 当前实验

当前先把 dense 训练环境换成中等密度：

```bash
scripts/start_training_detached_dense5_gentle_geo_critic_from_5d.sh
```

训练设置：

- warmstart：`5D` actor
- 训练环境：`stage4_asym_dense_5_gentle`
- critic：几何邻域 local critic
- 目标：先训练出一个会处理中等 dense 交互的 actor，再回到 `stage4_asym_dense_5_bridge` 做 hard test

## dense 密度定义

训练用 dense 不是五车同时抢同一个点，而是：

- 5 车在同一个有墙世界里运行；
- 2-3 个主交互智能体；
- 2-3 个压力/保持智能体；
- 起点不贴脸，目标不扎堆；
- 路径有交叉，但不做五路中心强交叉。

当前训练集：

`experiments/02_课程学习/cases/stage4_asym_dense_5_gentle_cases.json`

几何约束：

| 指标 | 目标 |
| --- | --- |
| 最小起点距离 | 约 `>= 1.2m` |
| 最小目标距离 | 约 `>= 1.0m` |
| 直线路径交叉 | 每个 case `0-3` 组 |
| hard bridge 用途 | 训练后测试，不再作为第一训练入口 |

## Baseline

| 模型 | 场景 | success | collision | full |
| --- | --- | ---: | ---: | ---: |
| `5D` | `stage4_asym_dense_5_bridge` | 0.540 | 0.475 | 0.250 |
| `5A` | `stage4_asym_dense_5_bridge` | 0.530 | 0.475 | 0.275 |

已清理的 hard stage4 不再作为训练或测试入口。它目标点过近，固定 `5D` 也只有约 `0.355 / 0.680 / 0.025`，更像病态压力测试。

## 当前判断

- `5D -> bridge` full fine-tune 会退化。
- `5A -> bridge` full fine-tune 基本维持原状，随后也没有突破。
- 这说明问题不只是 `5D` 起点太拟合，而是 full actor 解冻本身容易把已有导航能力拉坏。
- `5D -> bridge` head-only 没有崩，但也没有超过 baseline，说明只训练最后动作头不够。
- `stage4_asym_dense_5_bridge` 直接训练过硬：五车路径冲突过于集中，更适合 hard test。
- 下一步先在 `stage4_asym_dense_5_gentle` 上训练 dense 专家，再用 bridge 验证泛化。

## 已完成结果

| 实验 | 场景 | 最好 success | 最好 collision | 最好 full | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `5D -> bridge` full actor | `stage4_asym_dense_5_bridge` | 约 0.58 | 后续约 0.79 | 后续约 0.00 | 解冻后退化，不作为主线 |
| `5A -> bridge` full actor | `stage4_asym_dense_5_bridge` | 约 0.53 | 约 0.48 | 约 0.28 | 基本维持，没学出 dense 专家 |
| `5D -> bridge` head-only | `stage4_asym_dense_5_bridge` | 0.580 | 0.430 | 0.300 | 稳住了，但没有实质超过 `5D` baseline |
| `5D -> bridge` geo critic | `stage4_asym_dense_5_bridge` | 0.550 | 0.460 | 0.275 | 场景太硬，前期多为 critic-only，停止 |

`5D -> bridge` head-only 8 轮 eval：

| epoch | success | collision | full |
| ---: | ---: | ---: | ---: |
| 1 | 0.530 | 0.460 | 0.275 |
| 2 | 0.580 | 0.430 | 0.275 |
| 3 | 0.550 | 0.465 | 0.275 |
| 4 | 0.535 | 0.475 | 0.300 |
| 5 | 0.525 | 0.480 | 0.200 |
| 6 | 0.555 | 0.460 | 0.275 |
| 7 | 0.545 | 0.455 | 0.300 |
| 8 | 0.535 | 0.480 | 0.300 |

## 规则

- 运行中的日志临时放仓库根目录 `logs/`。
- 跑完后，有效日志归档到本目录；无效日志直接删。
- 不再把 attention / gate 实验混进这里。
