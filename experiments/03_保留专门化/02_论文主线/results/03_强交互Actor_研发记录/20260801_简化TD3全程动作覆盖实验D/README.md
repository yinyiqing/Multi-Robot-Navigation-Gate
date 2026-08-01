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

## 结果

Actor全程冻结，参数与5A完全相同。固定50场validation的波动不代表学习：

| epoch | success | full success |
| ---: | ---: | ---: |
| 1 | `0.752` | `0.400` |
| 2 | `0.748` | `0.380` |
| 3 | `0.752` | `0.380` |
| 4 | `0.736` | `0.340` |

最终采集`20038`条agent transition、`8823`个联合环境步，只执行`2850`次Critic更新。危险距离样本的五档线速度数量为`263/248/272/250/269`，但`Q(full)-Q(stop)`仍从epoch 1的`+0.104`增加到epoch 4的`+0.575`，最终全速偏好率为`100%`。Critic loss仅从前五次约`552`降到后五次约`499`，没有收敛证据。

## 结论修正

本实验不允许解冻Actor，但原因不能再表述为“均衡动作覆盖也无法修正危险加速”。后续代码审查发现三项协议问题：

1. 训练更新按`ceil(agent_samples / 5)`计算。部分车辆提前终止后，该值显著低于真实联合环境步；本实验在replay warmup之后只得到约一半的历史训练更新强度。
2. 五台车同时随机线速度，而原始24维Critic只观察ego状态和ego动作。其他车辆的随机动作会改变转移和`0.8自身 + 0.2邻车`reward，却对Critic不可见，因此动作效果被混杂。
3. `min_laser <= 0.5m`同时包含墙、箱子、靠近机器人和远离机器人。统一要求所有这些状态`Q(full)<Q(stop)`没有真实反事实依据。

因此D只能证明“当前Critic未达到解冻条件”，不能证明reward符号错误、动作覆盖路线无效或Dense场景不可学。旧的五档Q扫描降级为退化报警，不再作为正确动作标签。

训练更新口径随后恢复为每个联合环境步一次更新。新的Critic校准必须从同一Gazebo锚点重放，固定其他四台车和ego转向，只替换ego第一步线速度，再比较预测Q排序与真实N-step回报排序。校准工具为`TD3/calibrate_simple_td3_critic.py`；在校准通过前不启动Actor训练。

严格冻结第一步联合动作后的单锚点冒烟中，5个速度分支只有2个通过完整Actor状态重放阈值；这两个分支均在短期内发生ego碰撞，真实N-step目标为`-95.80/-95.26`，D Critic却预测`+2.21/+2.47`。该结果确认D Critic在有效分支上严重欠拟合，同时说明Gazebo重放仍有传感器级非确定性；样本不足以报告总体动作排序准确率。
