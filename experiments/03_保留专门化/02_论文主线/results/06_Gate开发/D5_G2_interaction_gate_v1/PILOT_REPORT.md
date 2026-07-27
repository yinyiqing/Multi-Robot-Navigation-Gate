# G2-A 交互 Gate 可观测性 Pilot

日期：2026-07-27

结论：当前本机观测可以明显优于最小 LiDAR 距离规则，但不能以预定的 recall/FPR 同时复现 `2.0 m` 真值 Oracle。G2-A 未通过，暂不进入 G2-B 反事实标签和 G2-C 最终 Gate。

## 数据

- train / validation 各 100 场，互不重叠，sealed test 未读取；
- standard/dense 与 0-edge/positive-edge 四层各 25 场；
- 冻结 5A 采集轨迹，两个 Actor 均未更新；
- train / validation 分别包含 4004 / 4074 个活动机器人帧；
- v3 shard 同时保存 360 度 Oracle 和前方 180 度可观测标签。

| split | 场景 | 帧 | 候选 | 前方正例 | 360 度正例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 100 | 4004 | 14119 | 1697 | 2342 |
| validation | 100 | 4074 | 14412 | 1613 | 2312 |

## 方法

Gate 输入只有可部署信息：原 24 维 Actor 状态、冻结 G0 形状分数，以及 G1 产生的轨迹年龄、速度、闭合速度、CPA 和 TTC。网络为 `76 -> 128 -> 64 -> 1` MLP。训练按场景等权并做类别平衡，validation 选择 epoch 和阈值。

准入要求同时满足：recall `>=0.90`、总体 FPR `<=0.10`、standard/0-edge FPR `<=0.10`。

## 结果

| 标签/方法 | best epoch | precision | recall | 总体 FPR | standard/0-edge FPR | F1 | 通过 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 360 度 Oracle 模仿 | 7 | 0.790 | 0.829 | 0.289 | 0.253 | 0.809 | 否 |
| 前方 180 度 Oracle 模仿 | 8 | 0.836 | 0.861 | 0.111 | 0.070 | 0.848 | 否 |
| 前方最小 LiDAR 距离规则 | - | 0.444 | 0.914 | 0.749 | 0.750 | 0.598 | 否 |

前方 Gate 的 validation 分层结果：

| 层 | precision | recall | FPR |
| --- | ---: | ---: | ---: |
| dense / positive-edge | 0.898 | 0.919 | 0.187 |
| dense / 0-edge | 0.861 | 0.920 | 0.194 |
| standard / positive-edge | 0.790 | 0.804 | 0.092 |
| standard / 0-edge | 0.708 | 0.691 | 0.070 |

阈值边界审计：前方 Gate 在总体和 standard/0-edge FPR 都不超过 `0.10` 时，最高 recall 为 `0.845`；不存在 recall `>=0.90` 且同时满足两项 FPR 的阈值。

## 诊断

- 前方正例帧中 `1532/1613 = 94.98%` 至少存在一个匹配到机器人真值的候选。这说明 proposal 漏检不是唯一瓶颈，但 v3 shard 未保存每个目标机器人的逐帧距离，因此该数值不能冒充“2 m 内目标”的精确 proposal recall。
- 前方 Gate 明显优于不区分机器人和静态障碍的最小 LiDAR 距离规则，证明形状与运动特征有信息增益。
- train loss 从 `0.612` 降到 `0.013`，但 validation 最佳点在 epoch 8。继续训练只会扩大过拟合，不能靠增加 epoch 达到准入线。
- 360 度标签包含当前前视 LiDAR 完全看不到的后方机器人，结果显著更差，不能作为可部署 Gate 的直接监督目标。
- dense 两层保持高 recall 但静态误报较高；standard 两层 FPR 较低但 recall 明显下降。当前模型仍未学到跨场景稳定的机器人交互边界。

## 决策

1. 不进入 G2-B/G2-C，不读取 sealed test，也不做端到端 Gate 调参。
2. 不重复增加 epoch、简单 EMA、手工速度阈值或最小 LiDAR 距离规则。
3. 下一步先讨论监督契约：是继续要求复现几何 `2.0 m` Oracle，还是改为只标记传感器可见且确实影响 Actor 选择的状态。改变标签前必须先修改主协议。
4. 两个 Actor 继续冻结。
