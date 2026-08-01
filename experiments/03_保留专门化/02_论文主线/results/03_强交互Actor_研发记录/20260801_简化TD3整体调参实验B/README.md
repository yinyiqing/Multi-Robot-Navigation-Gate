# 20260801 简化TD3整体调参实验B

## 目的

修正实验A中已经确认的三个问题：危险状态仍持续获得正奖励、timeout没有终止处罚，以及随机Critic尚未学稳就更新Actor。

本实验继续使用原始24维TD3，不启用TTC、让行、恢复、anchor、gradient gate或额外网络模块。

## 配置

- warm-start：5A Actor；Critic重新初始化；
- train：fixed-v1 dense train，按cycle采样；
- validation：固定50场dense monitor；
- `0.8自身 + 0.2邻车`合作reward保持不变；
- 每5000 agent samples验证一次；第一段只跑4轮；
- replay达到6000后训练Critic；Actor在21000 agent samples后才允许更新，因此第一段不会改动Actor。

### 基础奖励

| 参数 | 实验A | 实验B |
|---|---:|---:|
| progress | 20.0 | 10.0 |
| forward | 0.5 | 0.0 |
| turn penalty | 0.2 | 0.05 |
| obstacle penalty | 0.5 | 1.0 |
| stagnation penalty | 0.0 | 0.0 |
| timeout terminal reward | 无 | -150.0 |

### TD3参数

| 参数 | 实验A | 实验B |
|---|---:|---:|
| Actor LR | 1e-5 | 2e-6 |
| Critic LR | 1e-4 | 2e-5 |
| batch size | 256 | 128 |
| gamma | 0.995 | 0.999 |
| Actor开始更新 | replay 6000 | agent samples 21000 |
| Critic warmup noise | 0.10 | 0.25 |
| Actor训练noise | 0.10 | 0.05 |

## 判断标准

使用实验A replay离线重算后，新系数把普通非终止状态的平均基础奖励由`+1.654`降至约`+0.590`，把激光距离不超过`0.5m`状态的平均基础奖励由`+1.476`降至约`+0.189`，同时取消直接速度奖励。`gamma=0.999`时，第300步的`-150` timeout回报折算到episode起点仍约为`-111`，不会像实验A那样被过度折扣成便宜的退出方式。

第一段4轮只训练和检查Critic，Actor保持5A不变。解冻前必须确认危险状态的Q值不再单调偏好全速；若仍偏好危险加速，不恢复Actor训练。

第一段结束后的审计命令：

```bash
source env.python.sh
python3 scripts/audit_simple_td3_critic.py \
  --checkpoint TD3/checkpoints/independent_dense_actor_simple_td3_hparam_b_s20260801_latest.pt
```

审计通过后从同一checkpoint恢复，设置`DRL_MULTI_RESUME_TRAINING=1 DRL_MULTI_MAX_EPOCHS=8`，不重新收集前20000条replay。

Actor解冻后只接受同时满足以下条件的趋势：full success上升、collision不升、timeout不持续增加。任一已知退化重现时停止，不依赖继续增加epoch碰运气。

## 脚本

- 启动：`scripts/start_training_dense_simple_td3_hparam_b.sh`
- 停止：`scripts/stop_training_dense_simple_td3_hparam_b.sh`
