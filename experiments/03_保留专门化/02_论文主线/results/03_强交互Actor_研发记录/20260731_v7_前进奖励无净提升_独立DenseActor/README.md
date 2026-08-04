# 20260731 v7：前进奖励无净提升（独立 Dense Actor）

## 目的

在v6基础上增加安全区域前进奖励、低速无进展惩罚，以及高优先级车辆前进奖励，检查能否阻止后期等待退化。

## 协议

- 5A Actor warm-start，新Actor独立控制完整episode；
- 无oracle、无Actor切换；
- fixed-v1 dense train按cycle采样；
- 固定50场dense ultrafast monitor；
- Actor在21000 agent samples后更新；
- 计划10轮，完成第8轮后停止。

## 结果

| epoch | success | full success | collision | timeout | avg steps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1（冻结5A） | 0.780 | 0.440 | 0.220 | 0.000 | 30.6 |
| 3 | 0.760 | 0.440 | 0.240 | 0.000 | 25.2 |
| 5 | 0.740 | 0.380 | 0.260 | 0.000 | 27.3 |
| 7 | 0.748 | 0.360 | 0.252 | 0.000 | 26.2 |
| 8 | 0.784 | 0.440 | 0.208 | 0.040 | 44.9 |

epoch 8的full success与冻结5A完全相同。agent success只增加`0.004`，同时出现`0.040` timeout，平均步数增加`14.3`。这不构成有效提升。

## 根因

v7没有解除v3-v6的核心减速机制：

- Critic仍被要求在所有`1.0 m`内接近状态偏好更慢动作；
- Actor仍被辅助loss要求把这些状态的raw linear action压到`-0.4`以下；
- 新增reward却要求高优先级车辆继续前进，训练目标互相冲突；
- 优先级由对方剩余目标距离决定，但Actor和Critic都没有该输入。

到epoch 8，Actor batch相对5A的raw linear action已下降约`0.25-0.30`，与v5/v6进入等待退化前的轨迹一致。因此停止，不再增加epoch。

## 决策

该checkpoint不作为Dense Actor候选。后续禁止在同一配置上继续叠加reward。v8关闭隐藏目标优先级、统一减速辅助loss、Critic减速排序和Actor Q归一化，恢复以TD3 Q目标为主；启动前先复核v6 epoch-11的200场配对结果。

归档日志：

`logs/archive/rejected/independent_dense_actor/train_independent_dense_actor_from_5a_recovery_v7_s20260731_20260731_101319.log`

best/latest checkpoint与模型权重只保留本机，不进入Git。
