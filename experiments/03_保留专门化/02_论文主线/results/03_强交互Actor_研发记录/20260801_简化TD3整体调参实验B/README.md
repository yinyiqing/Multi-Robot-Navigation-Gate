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
  --checkpoint experiments/03_保留专门化/02_论文主线/results/03_强交互Actor_研发记录/20260801_简化TD3整体调参实验B/checkpoints/independent_dense_actor_simple_td3_hparam_b_s20260801_latest.pt
```

审计通过后从同一checkpoint恢复，设置`DRL_MULTI_RESUME_TRAINING=1 DRL_MULTI_MAX_EPOCHS=8`，不重新收集前20000条replay。

Actor解冻后只接受同时满足以下条件的趋势：full success上升、collision不升、timeout不持续增加。任一已知退化重现时停止，不依赖继续增加epoch碰运气。

## 脚本

- 启动：`scripts/start_training_dense_simple_td3_hparam_b.sh`
- 停止：`scripts/stop_training_dense_simple_td3_hparam_b.sh`

## 启动检查

首次启动采集250条transition后发现公共启动脚本未显式启用`active_neighbors_only`，会让已终止车辆继续参与邻车reward。该运行在Critic开始更新前停止，没有产生checkpoint；修正为只使用仍活跃邻车后从零重启。

## 正式结果

| epoch | success rate | full success rate |
|---:|---:|---:|
| 1 | 0.760 | 0.440 |
| 2 | 0.760 | 0.440 |
| 3 | 0.756 | 0.400 |
| 4 | 0.756 | 0.380 |

- 共采集`20000`条transition，Critic更新`2892`次，Actor更新`0`次；
- Actor参数与5A完全相同；
- 危险状态（最小激光距离不超过`0.5m`）中，`Q(full)-Q(stop)`从epoch 1的`+0.118`持续增至epoch 4的`+6.609`；
- 最新Critic在上述危险状态的`100%`样本中错误偏好全速；
- 实际即时reward中，危险状态全速已经比停车差，因此失败不是reward符号错误，而是行为数据缺少危险状态下的中低速反事实覆盖。

结论：实验B不允许解冻Actor。下一步实验C保持本实验所有奖励和TD3超参数不变，只修正Critic warmup的线速度动作覆盖。

正式日志位于`logs/formal/`，完整审计位于`audits/critic_audit_latest.json`。checkpoint、模型和TensorBoard文件保留在本地归档中，不提交GitHub。
