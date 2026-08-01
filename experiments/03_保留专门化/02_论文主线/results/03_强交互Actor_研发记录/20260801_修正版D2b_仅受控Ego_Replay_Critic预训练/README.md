# 20260801 修正版D2b：仅受控Ego Replay的Critic预训练

## 目的

D2虽然每个联合环境步只随机一台ego的线速度，但仍存入所有活跃车的transition。非选中ego的转移因此受到Critic不可见的邻车当前随机动作影响。D2b只排除这类混杂样本，不同时改reward、网络或优化器。

## 单变量协议

- Actor：加载5A并全程冻结，24维输入；
- Critic：从头初始化，52维ego-motion输入；
- 动作：每个联合环境步只随机一台活跃ego的线速度，其余车严格执行5A；
- Replay：每步只保留该受控ego的一条transition；
- Critic更新：Replay达到`3000`后，每个联合环境步更新一次；
- 预算：`12000` Replay样本，约`9000`次Critic更新，只在结尾validation一次；
- reward、batch size、discount、Critic学习率与D2相同；
- 不启用ranking loss、anchor、gradient gate、TTC或附加安全reward。

## 准入标准

日志必须满足`episode_agent_samples = episode_env_steps = random_linear_samples`，Replay达到`3000`后必须满足`updates_per_env_step=1.000`，且`actor_unlocked=0`始终成立。

训练结束后使用与D2相同的同状态Gazebo N-step标定。只有Critic Q排序与实际折扣reward呈稳定一致，才能进入Actor初始化对照；否则不解冻Actor。

## 脚本

- 启动：`scripts/start_training_dense_simple_td3_hparam_d2b.sh`
- 停止：`scripts/stop_training_dense_simple_td3_hparam_d2b.sh`
