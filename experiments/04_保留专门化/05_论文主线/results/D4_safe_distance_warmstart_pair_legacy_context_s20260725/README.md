# Safe-Distance Warm-Start Pair (Legacy Context)

状态：停止。该组使用旧版 local-Critic context；5D 在 epoch 1 完成后恢复训练，并在 Actor 解锁后因 context 审计发现实现问题而停止，不再补跑 epoch 2。

## 配置

- 五车固定 strong-interaction stage2 train / 140 场 validation
- Actor-only warm-start，分别从 5A 与 5D 开始
- 新初始化 geometry-only local Critic：Actor 24 维，Critic 69 维
- reward：`0.8 self + 0.2 distance-weighted neighbor`，另加 `1.0 m` robot safe-distance penalty
- Actor 在 21000 agent samples 前冻结，之后只使用 interaction replay 更新

## 结果

| warm-start | epoch | Actor更新 | agent success | collision | full success | close full | deep full | margin full |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5A | 1 | 否 | 83.7% | 16.3% | 48.6% | 62.5% | 18.3% | 80.0% |
| 5A | 2 | 是 | 81.3% | 18.6% | 42.1% | 47.5% | 15.0% | 77.5% |
| 5D | 1 | 否 | 82.3% | 17.7% | 48.6% | 62.5% | 11.7% | 90.0% |

5A 解锁后全面退化。5D 在第一次运行时因遗留 Gazebo 占满内存而中止，清理后从 checkpoint 恢复；epoch 1 基线完成后继续到 22926 samples。随后审计确认旧 Critic context 的 `relative_x/y` 实际位于世界坐标系，且 geometry-only context 不含邻车运动。继续训练已没有解释价值，因此停止。

## Context 审计

- stage2 train 初始目标方向左右比例约 `50.1% / 49.9%`，邻居方位左右比例约 `47.8% / 52.2%`，没有明显场景方向失衡。
- 在 replay state 上只旋转旧 context 的世界坐标 `relative_x/y` 90 度，保持本车 observation、邻居距离和本车 bearing 不变：
  - 5A Critic 的 Q 平均绝对变化为 `2.53`；相同状态下将 raw linear action 增加 `0.1`，Q 平均只变化 `0.27`。
  - 5D epoch-1 Critic 对应数值为 `2.75` 与 `0.30`。
- 这说明 Critic 明显利用了 Actor 不可见的全局方向捷径。强交互回报又取决于邻车运动，而旧 geometry-only context 只有位置、距离和 bearing，形成额外的状态混叠。

后续使用版本化 `ego_motion` context：邻车相对位置和相对速度都转换到本车坐标系；Actor、TD3结构和部署输入保持不变。legacy 模式保留，仅用于复现历史 checkpoint。
