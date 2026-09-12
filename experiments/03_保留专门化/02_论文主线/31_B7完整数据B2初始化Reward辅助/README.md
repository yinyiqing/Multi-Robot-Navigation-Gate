# G31 / B7：完整 Router 数据、B2 初始化与 Reward 辅助 head

状态：离线训练完成，等待独立 Dense256 development 闭环。

## 与 B6 的差异

- 主 interaction head 使用完整 Router 训练帧，而不是 251 个 reward 锚点；
- GRU 和 interaction head 从冻结 B2 初始化；
- reward 差只训练独立的辅助 preference head，权重固定为 0.10；
- 部署时先使用 interaction 概率，只有概率位于 `[0.40, 0.60]` 且 reward head 置信度足够时，
  才用 reward preference 做 tie-break；hysteresis、minimum hold、stride=2 保持不变。

## 数据与证据边界

- 两个 Actor、G0/G1 和 B2 均冻结；
- `Delta r = r_I - r_N` 来自同状态双 Actor 一步反事实记录；
- B6 的错误归一化运行与结果不读取；
- 闭环准入使用全新 256 场 development，并和同 manifest/seed 的 B2 配对。

## 离线结果

训练 40 epoch，完整主训练帧约 70,981，验证帧约 4,785，reward 锚点 251/57；验证分类准确率
在训练过程中约 80.9% 附近，未出现 B6 的 interaction-selection collapse。
