# 20260731 v8：Critic 危险加速退化（独立 Dense Actor）

## 目的

移除 v3-v7 的统一减速、目标优先级和 Actor gradient gate，只保留 TD3 Q 目标，并检查此前的等待退化是否来自过强安全约束。

## 协议

- 5A Actor warm-start，新 Actor 独立控制完整 episode；
- 24维 Actor 输入，ego-motion local Critic；
- 无 oracle、无 Actor 切换；
- `0.8自身 + 0.2邻车`合作 reward；
- fixed-v1 dense train 按 cycle 采样；
- 固定50场 dense validation monitor；
- Actor 在21000 agent samples后解冻。

## 结果

| epoch | 状态 | success | full success | collision | timeout |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 冻结5A | 0.768 | 0.380 | 0.232 | 0.000 |
| 2 | 冻结5A | 0.728 | 0.380 | 0.272 | 0.000 |
| 3 | Actor解冻 | 0.716 | 0.280 | 0.284 | 0.000 |
| 4 | Actor解冻 | 0.712 | 0.320 | 0.288 | 0.000 |
| 5 | Actor解冻 | 0.680 | 0.180 | 0.312 | 0.020 |

第6轮验证只完成20/50场后人工停止，不作为结果。

## 根因审计

- epoch-5 Actor 相对5A的 raw linear action 平均增加 `0.1151`，绝对变化也是 `0.1151`，说明变化几乎全部是统一加速；
- `<=0.8m`危险状态仍平均加速 `0.1162`；
- 对明显变化状态，Critic有 `80.13%`偏好新Actor；危险变化状态中仍有 `59.82%`偏好加速；
- replay中 `1.2-2.0m`状态占 `40.15%`，但当时没有对应速度约束；
- `75%`交互采样配合 safe-only anchor，使实际受anchor约束的样本只有约 `25%`；
- Q均值由约 `7.5`涨到 `43`，Max Q由约 `22`涨到 `133`，未归一化的Q梯度逐渐压过固定anchor。

因此失败原因不是训练轮数不足，而是 Critic 的动作外推偏差重新学成了危险加速。

## 后续修正

v9不恢复旧的统一低速目标，而是：

- 危险接近时只禁止新Actor比5A更快，允许减速和改变转向；
- 恢复Q尺度归一化；
- 使用全状态轻量5A trust-region anchor；
- 机器人进入安全距离时取消基础速度奖励，但保留目标进展奖励。

完整日志、epoch 1-5权重、checkpoint、评估数组、TensorBoard和replay审计均保存在本目录；权重与checkpoint仅本地保留，不进入Git。
