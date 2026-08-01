# 20260801 修正版D2：受控单车Critic预训练

## 目的

先回答一个单独的问题：在不修改5A Actor的前提下，TD3 Critic能否学会与真实短期回报一致的动作排序。Critic没有通过校准前，不训练Dense Actor，也不调整reward、规则约束或观测。

## 协议

- Actor：加载5A并全程冻结，Actor输入仍为原24维观测；
- Critic：重新初始化，不继承实验D的checkpoint或Replay；
- 探索：每个联合环境步只随机一台仍活跃车辆的线速度，其他车辆严格执行冻结5A；
- Critic输入：24维本车状态，加4个邻车的ego-motion上下文，共52维；
- 更新频率：Replay达到6000条后，每个联合环境步执行一次Critic更新；
- reward：`progress=10`、`forward=0`、`turn=0.05`、`obstacle=1`、`timeout=-150`，保持`0.8自身 + 0.2活跃邻车`；
- 不启用ranking loss、anchor、gradient gate、TTC或附加安全reward。

ego-motion上下文只在训练Critic时使用，包括邻车相对位置、距离、方位、相对速度和mask。部署Actor没有新增输入。

## 为什么这样修改

旧实验D同时随机五台车，但24维Critic看不到其他四台车的动作，导致同一个ego状态和动作对应不同的隐藏转移。D2只随机一台ego，并把上一时刻邻车相对运动提供给Critic，尽量去除该混杂，同时保持基础TD3 Actor结构不变。

## 准入标准

训练日志必须先确认：

- `Critic state dim: 52`；
- `actor_unlocked=0`始终成立；
- Replay开始更新后，`updates_per_env_step=1.000`；
- `random_linear_samples`每个episode不超过该episode联合环境步数。

正式判断使用`TD3/calibrate_simple_td3_critic.py`的同状态Gazebo N-step校准：固定其他四台车第一步动作和ego转向，只替换ego第一步线速度，在通过重放一致性检查的分支内比较Critic Q排序与真实N-step回报排序。

validation成功率不会变化，因为Actor被冻结；旧的`min_laser <= 0.5m`五档Q扫描只能用于退化报警，不能作为动作正确性的真值。

只有同状态校准显示Critic排序具备稳定一致性后，才进入小范围Actor/Critic学习率和探索比例扫描。否则继续查Critic训练与校准，不解冻Actor。

## 脚本

- 启动：`scripts/start_training_dense_simple_td3_hparam_d2.sh`
- 停止：`scripts/stop_training_dense_simple_td3_hparam_d2.sh`
- 校准：`TD3/calibrate_simple_td3_critic.py`
