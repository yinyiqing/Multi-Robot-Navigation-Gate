# 20260801 简化TD3参数实验A

## 目的

检查此前Dense Actor训练失败是否主要来自基础TD3参数，而不是继续增加奖励、约束或网络模块。

## 唯一改动范围

- 从5A Actor warm-start，使用原始24维TD3 Actor和Critic；
- Critic重新初始化，因为5A Critic文件没有保留；
- replay达到6000条前不更新网络，之后Actor与Critic同时更新；
- `actor_lr=1e-5`，`critic_lr=1e-4`，`batch_size=256`，`gamma=0.995`；
- 探索噪声从`0.10`衰减到`0.03`；
- 保留原始基础奖励系数和`0.8 self + 0.2 neighbor`，关闭所有附加奖励、Actor约束和局部Critic。

## 协议

- train：fixed-v1 dense train，按cycle采样；
- validation：固定50场dense monitor；
- 每5000 agent samples验证一次，共3轮；
- epoch 1位于replay warmup阶段，用于记录5A基线；
- epoch 2至3用于观察基础参数调整后的学习方向。

本实验只判断是否出现稳定学习信号。若有效，再进行实验B，仅调整基础奖励中的前进、转弯和障碍系数。

## 结果

| checkpoint | success | full success | collision | timeout |
|---|---:|---:|---:|---:|
| epoch 1（未训练5A基线） | 0.760 | 0.400 | 0.236 | 0.020 |
| epoch 2前30场（训练后） | 0.460 | 0.000 | 0.533 | 0.033 |

epoch 2验证到30/50场时人工停止。退化幅度已远超仿真波动，不继续消耗时间。

Replay审计显示，Actor平均环境线速度由`0.740`升至`0.922`，在激光距离不超过`0.5m`的状态中由`0.558`升至`0.766`。Critic对新Actor动作的偏好比例约为`79%`，并在危险状态中单调偏好更高速度。

因此实验A失败。直接原因是随机Critic没有预训练便与Actor同时更新；基础奖励的速度偏向、动作覆盖不足和过高学习率共同放大了该问题。后续由相邻的`20260801_简化TD3整体调参实验B`替代。

## 运行记录

首次运行在epoch 1 validation完成`10/50`场后，Gazebo固定物理步进停止推进并抛出`TimeoutError`。当时Actor尚未更新，因此没有产生训练结论。后续从最新checkpoint恢复，关闭固定物理步进；正式比较使用固定manifest，并对候选checkpoint做重复评测，不把单次仿真轨迹当作确定性结果。
