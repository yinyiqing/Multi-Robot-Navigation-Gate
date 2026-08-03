# 实验索引

历史README中的“当前”和“下一步”只代表当时判断。跨阶段的唯一当前结论以
[论文主线](03_保留专门化/02_论文主线/README.md)为准。

## 当前最短路径

```text
实验总览
  -> 03_保留专门化/02_论文主线
     -> 09_单冲突Residual组合主线/05_单冲突Residual_ActorB
     -> 07_冲突拓扑组合泛化
     -> 08_exact_edge2零样本确认
```

## 阶段

| 目录 | 状态 | 作用 |
| --- | --- | --- |
| [`01_第一次尝试/`](01_第一次尝试/) | historical | 基础TD3、多车共享策略和reward机制验证 |
| [`02_课程学习/`](02_课程学习/) | historical / baseline | 形成5A、5D并记录覆盖训练退化 |
| [`03_保留专门化/`](03_保留专门化/README.md) | current | 单冲突技能、Residual Actor B、Gate和组合泛化 |

## 当前实验

| 实验ID | 状态 | 目的 |
| --- | --- | --- |
| `generalist-5a` | frozen | 普通导航Actor A；0-edge full success `0.875` |
| `interaction-teacher-epoch16` | frozen teacher | 单冲突局部避让教师；不作为最终独立Actor |
| `actor-b-single-edge-residual` | approved / not started | 冻结5A，只训练epoch-16指导的Residual |
| `learned-gate-g2a` | frozen baseline | exact-edge-2有`+0.080`方向性收益，但未通过最终准入 |
| `single-to-multi-conflict-gate` | pending Actor B | 冻结A/B后，只用0-edge和单冲突数据训练Gate |

## 已拒绝但保留证据

- standard/dense两个完整场景专家；
- 5D + 零初始化Residual；
- epoch-16整网全程续训；
- 只在5A轨迹上的双教师离线蒸馏；
- 在完整dense和多冲突场景上训练独立Actor B。

这些实验只保留在历史诊断与`results/`中，不保留可误启动的当前训练入口。

## 阅读顺序

1. [实验总览](实验总览.md)
2. [论文主线](03_保留专门化/02_论文主线/README.md)
3. [Residual Actor B协议](03_保留专门化/02_论文主线/09_单冲突Residual组合主线/05_单冲突Residual_ActorB/README.md)
4. [数据集索引](03_保留专门化/02_论文主线/datasets/README.md)
5. [结果索引](03_保留专门化/02_论文主线/results/README.md)

## 归档规则

- 正式run必须记录实验ID、模型ID、scenario ID、seed、commit和完整配置。
- 当前日志写入根目录`logs/`，形成结论后归档到对应实验目录。
- 已归档叶子目录不覆盖；协议变化建立新实验ID。
- train、validation和sealed test必须保持scenario ID互斥。
- 禁止依据模型成败删除test场景。
- 历史失败保留结论和必要证据，但其启动脚本不得继续伪装成当前入口。
