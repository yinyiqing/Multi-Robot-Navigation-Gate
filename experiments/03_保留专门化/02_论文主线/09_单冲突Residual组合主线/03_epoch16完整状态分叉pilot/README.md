# epoch-16完整训练状态分叉pilot

状态：`historical / rejected after fixed-200 validation`。50场短pilot的正向趋势没有在
固定200场复现，不得据此恢复epoch-16整网续训。

## 假设

epoch-16失败于独立导航，不等于它的训练状态不可继续。原完整checkpoint包含：

- 与冻结epoch-16逐张量一致的24维Actor；
- 已经支持有效局部避障的87维ego-motion Critic；
- `320000`条replay，其中`147696`条为交互样本；
- 冻结5A负责普通状态、epoch-16负责交互状态时采集的完整轨迹。

因此直接分叉这套训练状态，比从随机Critic重新学习更符合“局部专家全程化”的目标。

## 单变量协议

相对epoch-16原训练只改变执行和Actor更新范围：

- 训练和验证中Actor B全程独立执行，关闭Oracle rollout；
- Actor从interaction-only更新改为全部状态更新；
- 训练期Bellman target继续使用原教师契约：`>2.0 m`由冻结5A产生target action，`<=2.0 m`由Actor B产生；
- Oracle target只进入训练Critic，不参与环境执行或验证；
- Critic batch按`50% interaction + 50% normal`采样；
- 保持原reward、Actor/Critic学习率、安全ranking和gradient guard；
- 使用完整dense train，validation为固定50场ultrafast monitor。

源checkpoint：

```text
TD3/checkpoints/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_latest.pt
```

训练分叉不会修改源checkpoint。

## 两轮预算

- epoch 17：从`320000`累计到约`325000` agent samples，Actor保持冻结，获得同协议独立运行基线；
- epoch 18：Actor在`326500` samples后解冻，运行到约`330000` samples，检查第一段全状态更新趋势；
- 每轮验证50个固定scenario；任一明显碰撞、timeout或全局动作偏置退化立即停止。

该pilot只回答“复用专家完整训练状态后，第一步全程化是否有正向趋势”，不直接作为论文最终结果。

## 结果

运行时间：`2026-08-02 23:40`至`2026-08-03 00:59`。

日志：

```text
logs/train_actor_b_from_epoch16_full_pilot_v1_s20260802_20260802_234053.log
```

固定50场结果：

| 指标 | epoch 17冻结基线 | epoch 18短训 | 变化 |
|---|---:|---:|---:|
| agent success | 0.760 | 0.780 | +0.020 |
| collision | 0.144 | 0.124 | -0.020 |
| unresolved | 0.096 | 0.096 | 0.000 |
| full success | 0.300 | 0.420 | +0.120 |
| timeout episode | 0.380 | 0.340 | -0.040 |
| avg env steps | 161.90 | 146.14 | -15.76 |
| avg final distance | 0.346 | 0.327 | -0.019 |
| avg reward | 39.654 | 50.843 | +11.189 |

epoch 18在碰撞没有反弹的情况下提高了full success并缩短episode，说明复用原专家Critic/replay后进行全状态更新这条方向成立。它修复的是部分“交互后不重启/不能收尾”问题，而不是把停车简单变成碰撞。

但该结果只通过趋势筛选，没有通过独立Actor准入：`timeout=0.340`仍高于`0.30`停止线，`full=0.420`也不足以支持Actor B已经具备可靠全程导航能力。因此不继续叠加epoch或改多个超参数，先扩大固定场景复核。

## 机制核验

- epoch 17 Actor与原epoch-16 Actor逐张量一致，最大参数差为`0`，冻结基线有效；
- Actor只在`326500` agent samples后解锁，解锁前日志均为`actor_unlocked=0`；
- 观察到的Actor gradient gate全部通过，危险样本上的梯度未出现恒定加速或单侧转向；
- epoch 17到epoch 18的相对参数L2漂移为`3.75e-4`，最大单参数变化为`7.00e-4`；
- 在8192个真实replay状态上，Actor输出平均变化为`[+0.0144, -0.0065]`，绝对变化均值为`[0.0144, 0.0109]`；仅`15.3%`状态的最大动作变化超过`0.05`。

因此当前改善来自小范围状态相关修正，不是全局动作偏置。

## 当时决策

1. 冻结epoch 18候选，不继续当前配置长训。
2. 固定200场复核已经完成，结果为full success下降、collision上升。
3. 当前正式决策是拒绝epoch-16整网续训，转向冻结5A的Residual Actor B。

候选checkpoint：

```text
TD3/checkpoints/actor_b_from_epoch16_full_pilot_v1_s20260802_best.pt
TD3/pytorch_models/actor_b_from_epoch16_full_pilot_v1_s20260802_epoch_018_actor.pth
```

历史启动入口已移除，防止误运行被拒绝协议。
