# 单冲突Residual Actor B协议

状态：`R0 completed / rejected`。不启动R1、R2和Gazebo长训练。

以下R1/R2内容是实验前预注册的历史方案，不是当前执行计划。当前方法直接冻结5A和
epoch-16并开发Gate。

## R0结果

使用epoch-16 checkpoint中的320000条单冲突Replay做了最小离线测试：前80%训练，
后20%按时间留出验证。普通状态目标为零Residual，交互状态目标为
`action_epoch16 - action_5A`。

| 指标 | 结果 | 准入要求 | 判断 |
| --- | ---: | ---: | --- |
| 交互动作MSE相对零Residual改善 | `59.8%` | `>=40%` | 通过 |
| 普通状态Residual MAE（线/角） | `0.349 / 0.074` | 均`<=0.03` | 失败 |
| 普通状态平均修正（线/角） | `-0.264 / -0.038` | 接近零 | 失败 |
| 交互状态teacher-choice accuracy | `0.487` | `>=0.70` | 失败 |
| validation动作范围覆盖（线/角） | `0.989 / 0.988` | 均`>=0.95` | 通过 |

结论：Residual能拟合一部分epoch-16动作差，但不能可靠判断何时使用，最终把普通状态
也普遍减速。失败不是Residual范围过小造成的；两个动作维度已分别校准到约
`2.000 / 1.889`，validation覆盖率接近99%。按预先规定的准入规则，当前24维单帧
Residual方案停止，不用闭环长训练代替这个失败的表达能力测试。

结果文件：

```text
local_data/r0_replay_temporal_v2/summary.json
```

限制：Replay没有逐transition的scenario ID，因此80/20是时间切分，不是正式的场景级
独立validation；但当前结果离准入线很远，已经足以否决继续投入长训练。

## 目标

训练一个能全程独立导航的Actor B，同时满足：

1. 普通导航能力由冻结5A保留；
2. 局部避让能力由epoch-16教师迁移到Residual；
3. 训练只见单冲突，复杂多冲突留作零样本组合测试。

## 模型

```text
a_B(s) = clip(a_5A(s) + delta_theta(s))
```

- `a_5A`：冻结的原24维5A Actor；
- `delta_theta`：小型Residual网络，唯一可训练的Actor参数；
- 两个动作维度使用分别校准的Residual范围，不沿用旧实验统一`0.10`范围；
- checkpoint必须同时保存冻结5A哈希、Residual参数和动作范围。

Actor B的部署输入仍只有本机Actor状态。训练标签可以使用仿真真值选择教师，但真值
不得进入Actor B或Gate的部署输入。

## 数据

### Actor B训练集

- 五车固定场景；
- `conflict_edge_count=1`；
- `max_conflict_degree=1`；
- `simultaneous_conflict_count=1`；
- deep/close/margin按冻结比例遍历；
- 保留完整episode中的普通路段、接近、避让和恢复阶段。

### 禁止进入训练

- exact-edge-2及以上；
- 多高度数冲突；
- 同时多冲突；
- dense validation和sealed test。

## 三阶段训练

### R0：表达能力审计

在互斥单冲突train/validation上统计：

```text
normal target      = [0, 0]
interaction target = action_epoch16 - action_5A
```

先确定两个动作维度所需Residual范围，并按scenario划分检查动作差是否能由当前24维输入
预测。若validation仍只学到全局偏置，不启动Gazebo长训练。

### R1：闭环教师初始化

不能只使用5A轨迹。数据至少包含：

1. `5A + epoch-16真值组合`访问的状态；
2. 当时Residual学生自己访问的状态；
3. 普通状态的零Residual目标；
4. 交互状态的epoch-16动作差目标。

每轮由学生独立运行，再聚合新状态并查询冻结教师。离线选择只看互斥validation，
不读取多冲突集。

### R2：TD3闭环训练

- Actor B全程控制；
- 5A base永久冻结；
- 只更新Residual和对应Critic；
- Critic学习实际执行的组合动作；
- Actor loss由TD3的Q项和逐渐衰减的教师动作约束组成；
- reward保持既定`0.8 self + 0.2 neighbor`协议；
- 不增加统一停车、统一减速、靠右或手工通行顺序奖励。

教师约束降为零后，Actor B仍须依赖导航回报独立完成episode。

## 与历史实验的控制变量

| 项目 | 旧Residual | 本次已拒绝候选 |
| --- | --- | --- |
| base | 冻结5D | 冻结5A |
| Residual起点 | 零，靠Critic探索 | epoch-16动作差监督初始化 |
| 训练数据 | 混合edge-1 | 冻结单冲突分层 |
| 闭环分布 | 仅TD3 rollout | 教师组合与学生状态聚合 |
| 训练目标 | Q最大化/anchor | 教师初始化后再逐步转TD3 |
| 最终要求 | 局部提升 | Actor B全程独立导航 |

因此该候选不是旧Residual的重复，但R0结果仍然否决了它。

## 准入

短pilot按顺序通过以下条件后才能增加训练预算：

1. 普通状态Residual接近零，没有固定加速或单侧转向；
2. 单冲突validation相对5A降低collision；
3. full success提高且timeout不增加；
4. Actor B独立运行，不使用Oracle接管；
5. 固定200场复核保持方向，不依据50场波动宣布成功。

该候选没有通过，因此后续阶段未执行。当前Gate直接在冻结5A和epoch-16之间选择。
