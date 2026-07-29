# D5 Independent Dense Actor from 5A

状态：`stopped / rejected`。该实验用于训练一个可以从起点到终点全程独立控制的Dense Actor，但Actor在7个完整epoch后仍基本等于5A，因此停止，不允许续训。

启动信息：

- 有效启动时间：`2026-07-28 22:51:11 Asia/Shanghai`；
- 协议起点commit：`d7061bb`；
- PID file：`.train_independent_dense_actor_from_5a_full_v1_s20260728.pid`；
- 停止时间：`2026-07-29 13:14 Asia/Shanghai`；
- 完整epoch：`7`，epoch 8在`458142 agent samples`附近停止；
- 归档日志：`train.log.gz`。

首次启动在约760 agent samples时主动停止，当时Actor尚未解冻。启动审计发现旧代码将Replay的interaction标记错误绑定到oracle rollout开关：关闭oracle后，即使Critic已看到邻车，`interaction_replay`仍恒为0，使`critic_interaction_fraction=0.5`实际失效。修复后，oracle flag只决定Actor切换，Replay interaction flag独立根据邻车context生成。旧Replay和checkpoint已删除，无效启动日志保留为`aborted_startup_interaction_replay_bug.*`。

有效重启第1个episode为`replay=51, interaction_replay=51`，确认修复生效；oracle和interaction-only Actor更新仍均为关闭。

## 为什么重训

- epoch-16条件Actor与5A按`2.0 m`oracle组合时，dense validation full success从`0.3090`提高到`0.5450`；
- 它独立控制256场时full success只有`0.2305`，且`51.56%`场景超时；
- 原因是它训练时只在邻车`<=2.0 m`时执行，且Actor只从危险交互样本更新，没有学会完整导航和停车后恢复。

## 固定协议

| 项目 | 设定 |
| --- | --- |
| Actor起点 | 冻结5A Actor |
| Critic起点 | 新建ego-motion local Critic |
| Actor结构 | 原TD3单帧24维输入，不改网络 |
| 控制方式 | 新Actor在整个episode全程控制 |
| train | 固定`dense/train`，6000场，`cycle`无放回遍历 |
| frequent validation | 200场policy-independent dense monitor |
| final validation | 完整`dense/validation`，1000场 |
| reward | `0.8 self + 0.2`距离加权邻车reward |
| 安全信号 | Critic邻车运动context + 危险动作排序 |
| 能力保留 | `>2.0 m`无近距离邻车样本的动作约束到5A |
| Actor预热 | 前65000 agent samples仅训Critic，epoch 1为5A基线 |
| 训练量 | 16 x 60000 agent samples，目标覆盖完整6000场 |

Actor不读取邻车真值、冲突边或dense标签。邻车运动context和安全状态anchor mask只在训练时给Critic/目标函数使用，部署输入仍是原24维。

## 准入条件

1. frequent monitor上的full success稳定高于epoch 1的5A基线，不只是单轮峰值；
2. collision下降不能主要被unresolved/timeout上升抵消；
3. 候选checkpoint在完整1000场dense validation上超过5A/5D的`0.3090/0.3140`；
4. 候选固定前不读取sealed test。

## 结果

| epoch | agent success | collision | full success | timeout |
| ---: | ---: | ---: | ---: | ---: |
| 1（冻结5A） | `0.698` | `0.301` | `0.315` | `0.005` |
| 2 | `0.701` | `0.299` | `0.305` | `0.000` |
| 3 | `0.687` | `0.313` | `0.290` | `0.000` |
| 4 | `0.702` | `0.297` | `0.315` | `0.005` |
| 5 | `0.710` | `0.289` | `0.310` | `0.005` |
| 6 | `0.703` | `0.297` | `0.305` | `0.000` |
| 7 | `0.693` | `0.306` | `0.300` | `0.005` |

没有任何完整epoch稳定超过冻结5A基线；epoch 4仅与基线持平。

## 停止后审计

`replay_action_audit_epoch007.json`使用停止时保存的`457956`条Replay，对比5A与epoch 7：

- `<=0.8 m`危险状态共`21146`条，占`4.62%`；
- 危险状态原始线速度变化均值仅`-0.0066`，绝对变化均值`0.0075`；
- 危险状态只有`12.49%`的动作变化超过`0.05`；
- Critic在全部危险状态上偏好候选动作的比例只有`45.43%`；
- 仅看动作确实发生明显变化的危险状态，Critic偏好候选的比例为`73.58%`。

结论：不是Actor学错后大幅退化，而是当前更新协议过度保守，Actor几乎没有离开5A。`actor_lr=1e-6`、约`58`的Q归一化尺度和全状态Actor batch共同稀释了危险动作梯度。旧reward还只奖励“远离邻车且同时向目标前进”，没有正确覆盖停车让行和短暂绕行。

## 产物

- `epoch004_best_actor.pth`、`epoch004_best_critic.pth`：monitor指标选出的持平checkpoint；
- `epoch007_final_actor.pth`、`epoch007_final_critic.pth`：最后一个完整epoch；
- `evaluations.npy`：7轮固定monitor结果；
- `train.log.gz`、`tensorboard.tfevents`：完整训练记录；
- `replay_action_audit_epoch007.json`：停止后Replay与动作审计。

该实验不得通过下面命令恢复。启动脚本现已指向修复后的v2新实验：

```bash
scripts/start_training_independent_dense_actor_from_5a.sh
```
