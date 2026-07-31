# ICRA论文主线：两个独立Actor + Gate

状态：`5A作为普通Actor冻结；v8确认Critic危险加速偏差；v9修正协议已就绪但未启动；Gate继续暂停`。

导师沟通后的当前目标是：

```text
普通/低交互状态 -> 独立Actor A（5A）
Dense/强交互状态 -> 独立Actor B（待重训）
实际执行           -> Gate在A/B之间选择
```

Gate的前提是A和B都能单独从起点完成导航。旧epoch-16策略只能在局部交互区使用，因此不能直接作为Actor B。

旧的“5A + 条件交互Actor + 状态Gate”完整协议保留在 [历史文档](HISTORY_CONDITIONAL_ACTOR_GATE.md)，用于追溯，不再代表当前实验决策。

## 1. 当前证据

### 普通Actor

5A和5D在248个固定0-edge validation场景上基本等价：

| Actor | agent success | full success |
| --- | ---: | ---: |
| 5A | `0.9726` | `0.8750` |
| 5D | `0.9718` | `0.8710` |

选择5A作为普通Actor，也作为Dense Actor的warm-start。5D继续作为历史generalist基线。

### Dense validation

完整`dense/validation`共1000场，平均有`2.457`条名义冲突边；`74.7%`的场景至少有2条冲突边，`45.3%`至少有3条。

| 方法 | agent success | collision | unresolved | full success | timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5A独立 | `0.7064` | `0.2930` | `0.0006` | `0.3090` | `0.0030` |
| 5D独立 | `0.7122` | `0.2872` | `0.0006` | `0.3140` | `0.0030` |
| 5A + epoch-16 oracle | `0.8476` | `0.1456` | `0.0068` | `0.5450` | `0.0180` |
| epoch-16独立（256场） | `0.7070` | `0.1531` | `0.1398` | `0.2305` | `0.5156` |

结论：epoch-16确实学到了局部避碰，但会把大量碰撞转化成停车超时，不是合格的独立Dense Actor。完整对比见 [Dense validation Actor comparison](results/05_当前冻结方案/D4_dense_validation_actor_comparison_s20260728/README.md)。

### 为什么历史`full=0.700`不矛盾

历史140场strong validation每场恰好只有1条冲突边；完整dense validation平均2.457条。前者验证单次局部避让，后者要求连续处理多组冲突并让5辆车全部到达，不能直接比较绝对成绩。

## 2. 数据协议

| Pool | train | validation | sealed test |
| --- | ---: | ---: | ---: |
| standard | 3000 | 500 | 1000 |
| dense | 6000 | 1000 | 2000 |

- train、validation和test使用同一pool的生成参数，seed和scenario ID互斥。
- 场景只按初始碰撞、传感器失效、无静态路径等策略无关规则过滤。
- 禁止依据5A、5D、新Actor或Gate的成绩删除场景。
- sealed test在Actor、Gate和超参冻结前保持未读。
- 训练期间高频趋势评估使用固定50场ultrafast monitor；候选复核使用固定200场dense monitor。两者都在新Actor运行前按冲突分布固定，不根据模型表现挑选。

数据详情见 [datasets](datasets/README.md) 和 [dense ultrafast monitor](datasets/fixed_v1/views/dense_validation_monitor_ultrafast_v3/README.md)。

## 3. 独立Dense Actor训练

失败对照：[D5 independent Dense Actor v1](results/05_当前冻结方案/D5_independent_dense_actor_from_5a_full_v1_s20260728/README.md)。v2-v8的后续修复与失败记录见[强交互Actor研发记录](results/03_强交互Actor_研发记录/README.md)。当前启动脚本已切换到v9，但尚未启动。

固定规则：

1. 从5A Actor warm-start，不从epoch-16或5D开始。
2. 新Actor在每个episode中全程控制，禁止`2.0 m` oracle或另一Actor接管。
3. 保持TD3、原24维Actor输入和`0.8自身 + 0.2邻车`加权reward。
4. Critic使用训练期邻车相对位置/速度；Actor部署输入不变。
5. 全状态使用轻量5A动作anchor限制策略漂移；危险接近时额外禁止新Actor比5A加速，但允许减速和改变转向。
6. 使用`dense/train`的6000场无放回循环，不再用只有单冲突边的140场子集代替完整Dense训练。
7. 同时检查full success、collision、unresolved、timeout和平均步数；不接受“碰撞下降但全部变成超时”。

### v3-v7 复盘

v3-v7共享同一个全局减速机制：`1.0 m`内正在接近的状态同时受到Critic减速排序和Actor减速辅助loss。随着Actor相对5A的线速度持续下降，碰撞先减少，随后大量转化为timeout。v5与v6都从中期高点退化到`full_success=0.200`、`timeout=0.680`；v7第8轮`full_success=0.440`，与冻结5A基线相同，因此停止。

这条路线还有两个协议问题：

- 基于“双方剩余目标距离”的让行优先级对24维Actor不可见，ego-motion Critic也没有对方目标距离；
- 让高优先级车前进的reward与“所有近距离接近车辆都减速”的辅助loss直接冲突。

固定200场配对复核表明，v6 epoch-11相对5A的full success从`0.335`提高到`0.445`，McNemar exact `p=0.00535`；该中期收益真实。但v6同时产生`0.120` timeout，平均步数从`26.96`增至`69.77`，因此仍不符合独立Dense Actor要求。完整结果见 [v6 epoch-11固定200场复核](results/03_强交互Actor_研发记录/20260731_v6_epoch11_固定200场配对复核/README.md)。

v8关闭统一减速后，epoch-5的success由冻结基线`0.768`降至`0.680`，full success由`0.380`降至`0.180`。replay审计显示Actor在危险状态相对5A平均加速约`0.116`，Critic仍偏好这些动作；同时未归一化Q持续膨胀，safe-only anchor只覆盖约25%样本。因此v8也已否定，详见[v8归档](results/03_强交互Actor_研发记录/20260731_v8_Critic危险加速退化_独立DenseActor/README.md)。

v9只针对上述已确认机制修正：恢复Q尺度归一化；全状态使用权重`0.5`的5A anchor；危险接近时采用相对5A的单边加速上限，不设置固定低速目标；机器人进入安全距离时取消基础速度奖励，但保留目标进展奖励。`0.5`来自v8 replay/Critic梯度复核：`0.05`对应的预期动作漂移约为`1.04/0.77`，约束实际无效；`0.5`将其限制到约`0.10/0.08`，仍允许策略学习。

## 4. 后续顺序

1. 固定200场复核已完成：v6有真实收益，但因`0.120` timeout和`2.59`倍平均步数未通过验收，不再运行其完整1000场validation。
2. 使用已锁定的v9协议重新训练独立Dense Actor，在monitor上先判断是否同时消除危险加速和等待退化。
3. 在完整1000场dense validation上确认候选独立超过5A/5D，并且没有系统性超时。
4. 补测新Actor的0-edge/standard validation能力，确认它与5A是否真的互补；如果新Actor全面优于5A，则不强行训练无意义的Gate。
5. 只有两个独立Actor存在稳定互补性时，才冻结它们并训练Gate。
6. Gate使用可部署的本机感知做状态选择，不读取场景名、dense标签或其他机器人真值。
7. Actor、Gate和所有阈值冻结后，最后一次运行sealed test。

## 5. 论文主张边界

当前可讲的主线是：多机器人导航的难度不只由空间密度决定，更由同步路径冲突决定；普通导航和多冲突导航需要不同的策略偏好，因此保留可靠普通Actor，训练独立Dense Actor，再用可部署Gate进行选择。

这条主张只有在以下两点都成立时才保留：

- 独立Dense Actor在dense validation上稳定超过普通Actor；
- 普通Actor与Dense Actor在逐场结果上存在可利用的互补性。

如果其中一点不成立，应该收缩或改变论文主张，不为了保留Gate而强行挑场景。
