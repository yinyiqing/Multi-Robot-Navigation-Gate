# 结果目录索引

本目录按证据作用归档，所有“当前路线”判断以父目录
[论文主线README](../README.md)为准。

## 当前需要的证据

```text
5A普通导航基线
  + epoch-16单冲突局部技能证据
  + 历史Gate的组合上限与失败边界
  -> 新的单冲突Residual Actor B
  -> 冻结A/B后的新Gate
```

## 目录职责

| 目录 | 状态 | 作用 |
| --- | --- | --- |
| [`01_基线评估/`](01_基线评估/README.md) | baseline | 5A、5D和fixed-v1基线 |
| [`02_普通Actor_失败对照/`](02_普通Actor_失败对照/README.md) | historical / failed | standard expert微调退化证据 |
| [`03_强交互Actor_研发记录/`](03_强交互Actor_研发记录/README.md) | historical | epoch-16形成过程及独立Dense Actor失败记录 |
| [`04_Gate前置验证/`](04_Gate前置验证/README.md) | diagnostic | 场景可解性、风险信号和感知边界 |
| [`05_当前冻结方案/`](05_当前冻结方案/README.md) | frozen evidence | 5A、epoch-16和独立Actor比较；不再表示当前训练方案 |
| [`06_Gate开发/`](06_Gate开发/README.md) | frozen baseline | G0/G1/G2-A和历史Gate，作为新Gate基线 |
| [`90_中止与无效运行/`](90_中止与无效运行/README.md) | invalid | 不得用于模型选择的运行 |

新的Residual Actor B结果进入
[`../09_单冲突Residual组合主线/05_单冲突Residual_ActorB`](../09_单冲突Residual组合主线/05_单冲突Residual_ActorB/README.md)，
避免继续混入旧独立Dense Actor研发记录。

## 归档规则

- 已完成实验保留事实、日志和必要checkpoint说明，不删除失败证据。
- 历史目录中的旧“下一步”不构成执行授权。
- 新实验必须先冻结数据边界；Actor B和Gate训练不得读取多冲突validation/test。
- `90_中止与无效运行`中的数值不得进入论文主表。
