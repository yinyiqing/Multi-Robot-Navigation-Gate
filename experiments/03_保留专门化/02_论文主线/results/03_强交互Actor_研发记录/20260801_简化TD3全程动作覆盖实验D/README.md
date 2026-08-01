# 20260801 简化TD3全程动作覆盖实验D

## 目的

验证在整个Critic warmup期间保持线速度均衡覆盖，能否稳定修正危险状态的速度Q排序。

实验C已经证明：前`10000`条动作均衡时，危险状态`Q(full)-Q(stop)=-0.565`，全速偏好率为`0%`；恢复5A采样后，中间速度样本消失，最终退化到`+1.915/100%`。

## 与实验C的唯一差异

- C：`DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=10000`；
- D：`DRL_MULTI_RANDOM_LINEAR_EXPLORATION_STEPS=21000`。

因此D的4个epoch中，5A继续提供转向，raw linear action持续均匀采样`[-1, 1]`。Actor仍在`21000 agent samples`后才允许更新，本阶段不会修改Actor。

其余网络、reward、训练场景、validation、学习率、batch、gamma、tau和随机种子均与实验C相同。

## 判断标准

运行约`20000 agent samples`后停止。必须同时满足：

- Actor optimizer steps为`0`；
- 危险状态`Q(full)-Q(stop)<0`；
- 危险状态偏好全速的样本比例显著低于50%，且不能随epoch重新升到100%。

通过后才讨论从同一checkpoint恢复并解冻Actor；不通过则停止继续增加warmup时长，转向Critic分层采样。

## 脚本

- 启动：`scripts/start_training_dense_simple_td3_hparam_d.sh`
- 停止：`scripts/stop_training_dense_simple_td3_hparam_d.sh`
- 审计：`scripts/audit_simple_td3_critic.py`
