# D5 Independent Dense Actor from 5A

状态：`prepared`。该实验用于训练一个可以从起点到终点全程独立控制的Dense Actor，不再训练局部条件策略。

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

启动：

```bash
scripts/start_training_independent_dense_actor_from_5a.sh
```
