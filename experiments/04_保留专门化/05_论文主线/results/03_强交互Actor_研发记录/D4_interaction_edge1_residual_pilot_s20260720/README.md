# D4 Edge-1 Interaction Residual Pilot

状态：`rejected candidate`。同一 423 场 edge-1 validation 上，冻结 5D 的 epoch 1 优于 residual 更新后的 epoch 2；不启动更多 seed，也不扩展到 edge 1-2。

## 协议

- train：512 场，standard/dense edge-1 各 256，固定顺序 cycle。
- validation：423 场，standard edge-1 212 + dense edge-1 211。
- base Actor：冻结 5D。
- residual：hidden 128，scale `0.10`。
- reward：距离加权 `0.8` self + `0.2` visible-neighbor。
- epoch 1：Actor 在 `41000` agent samples 前冻结，只预热 Critic。
- epoch 2：更新 residual；每 `40000` agent samples 完整验证一次。

## 结果

| Epoch | Actor state | Agent success | Collision | Unresolved | Full success | Timeout | Mean steps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | frozen 5D + zero residual | 0.8440 | 0.1546 | 0.0014 | 0.5130 | 0.0071 | 36.835 |
| 2 | trained residual | 0.8274 | 0.1721 | 0.0005 | 0.4704 | 0.0024 | 34.021 |
| delta | epoch 2 - epoch 1 | -0.0165 | +0.0175 | -0.0009 | -0.0426 | -0.0047 | -2.813 |

epoch 2 少成功 18 个完整场景（`217 -> 199`），同时单机器人碰撞增加，因此未达到 full success 至少 `0.60` 且碰撞下降的准入条件。

## 失败机制

在 latest replay buffer 的 80045 个 state 上离线检查 epoch 2 Actor：

- residual 原始输出均值约为 `[+0.1001, -0.1001]`，两个维度几乎对所有状态都饱和到边界；
- 93.18% 的 state 最终动作发生变化；
- clipped action delta 的平均绝对值为 `[0.0242, 0.0598]`；
- 平均 Q 在训练末段约 `72.7`，最大 Q 约 `135.0`，但真实 validation 下降。

Residual 没有学到随交互状态变化的避让修正，而是被 Critic 的单调偏好推成近似固定的“加速并向同一方向转向”偏置。这是价值高估/动作外推问题，不是继续增加 epoch 或 seed 能合理解决的问题。

归档保留 epoch 1 best、epoch 2 Actor/Critic、完整训练日志、两轮 evaluation 数组和 TensorBoard event。完整哈希见 `summary.json`；大型 replay checkpoint 不纳入 Git。
