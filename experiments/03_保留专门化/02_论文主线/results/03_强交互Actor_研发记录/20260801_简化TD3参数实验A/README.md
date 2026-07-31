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
