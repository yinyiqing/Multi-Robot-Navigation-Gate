# ICRA论文主线：从单冲突学习到多冲突组合泛化

状态：`current / frozen on 2026-08-03`。

跨目录的当前结论、实验准入和下一步只以本文件为准。历史README中的“当前”或
“下一步”只代表当时判断。

## 1. 一句话主张

> 只在单冲突场景中学习局部避让修正，再通过可部署Gate反复调用，零样本处理训练
> 中未见的多冲突组合，同时保留普通导航能力。

创新不写成“两个Actor加一个Gate”，而写成：

```text
single-conflict skill learning -> compositional generalization to multi-conflict navigation
```

## 2. 最终系统

```text
Actor A：冻结5A，负责可靠的普通导航
Actor B：冻结5A + 可训练Residual，可从起点独立运行到终点
Gate：使用本机传感器，在Actor A和Actor B之间切换
```

Actor B的动作定义为：

```text
a_B(s) = clip(a_5A(s) + delta_actor(s))
```

- `a_5A`永久冻结，保证普通导航能力不会被覆盖。
- `delta_actor`只在单冲突训练场景学习局部避让修正。
- epoch-16条件交互Actor只作为Residual的教师，不直接作为最终Actor B。
- Actor B训练和验证时必须全程独立控制，禁止Oracle或另一Actor接管。

详细协议见[单冲突Residual Actor B](09_单冲突Residual组合主线/05_单冲突Residual_ActorB/README.md)。

## 3. 训练与测试边界

### 训练可见

- 五车场景，`conflict_edge_count=1`、`max_conflict_degree=1`；
- 普通路段和一次主要冲突都保留，Actor B学习完整episode；
- 冻结5A提供普通导航基座；
- 冻结epoch-16为冲突状态提供避让动作参考；
- Gate只能使用0-edge和单冲突训练数据。

### 训练不可见

- `conflict_edge_count>=2`；
- `max_conflict_degree>=2`；
- `simultaneous_conflict_count>=2`；
- dense sealed test。

### 最终测试

先在互斥validation上完成模型和阈值选择，再一次性测试：

1. 0-edge能力保持；
2. exact-edge-1已见结构；
3. exact-edge-2零样本组合；
4. edge>=3、高度数和同时多冲突的泛化边界；
5. sealed test。

如果Actor B或Gate在训练中读取多冲突场景，就不能再声称零样本组合泛化。

## 4. 已有证据

### 冻结组件

- 5A在248个固定0-edge validation场景上的full success为`0.875`。
- epoch-16只在单冲突场景训练，并在匹配调用协议下把full success从`0.421`
  提高到`0.700`。
- epoch-16独立全程运行会产生大量timeout，因此它是局部技能教师，不是最终Actor B。

### 组合泛化上限

在完整dense validation上，5A与`5A + epoch-16真值Gate`的结果为：

| 方法 | agent success | collision | full success | timeout |
| --- | ---: | ---: | ---: | ---: |
| 5A | `0.7064` | `0.2930` | `0.3090` | `0.0030` |
| 真值Gate组合 | `0.8476` | `0.1456` | `0.5450` | `0.0180` |

按未见冲突结构分层：

| 测试层 | 5A full | 真值Gate | 绝对提升 |
| --- | ---: | ---: | ---: |
| `edge=1` | `0.502` | `0.730` | `+0.227` |
| `edge>=2` | `0.221` | `0.470` | `+0.249` |
| `edge>=3` | `0.135` | `0.393` | `+0.258` |
| `max_degree>=2` | `0.201` | `0.466` | `+0.265` |
| `simultaneous>=2` | `0.156` | `0.396` | `+0.240` |

这证明局部技能具有重复调用价值，但真值Gate不可部署。

### 可部署Gate现状

冻结learned Gate在独立exact-edge-2的200场确认中：

- full success：`0.325 -> 0.405`；
- agent success：`0.737 -> 0.812`；
- timeout：`0 -> 0`；
- McNemar exact `p=0.06812`；
- 只恢复真值Gate收益的`30.2%`。

方向为正，但没有通过预注册统计和Oracle收益恢复门槛。它是新Gate的基线，不是最终
论文结果。完整证据见[冲突拓扑组合泛化](07_冲突拓扑组合泛化/README.md)和
[exact-edge-2确认](08_exact_edge2零样本确认/README.md)。

## 5. 已拒绝路线

以下路线不得恢复为当前方案：

1. 在完整dense或多冲突场景上训练Actor B；这会破坏零样本组合泛化命题。
2. 直接复制epoch-16整网，再解冻为全程Actor；固定200场复核中full success
   `0.305 -> 0.260`，collision `0.111 -> 0.157`。
3. 在5A轨迹上离线蒸馏两个教师；teacher-choice accuracy仅`0.540-0.586`，且没有
   覆盖学生自己访问的状态。
4. 冻结5D并让Residual从零依赖Critic探索；曾产生恒定加速和单侧转向偏置。
5. 训练standard/dense两个完整场景专家；空间密度不是运行时交互状态，历史微调也
   多次覆盖已有能力。

这些结果只作为失败对照保留在`results/`和
[`09_单冲突Residual组合主线`](09_单冲突Residual组合主线/README.md)，不保留可误启动
的训练入口。

## 6. 当前执行顺序

1. 构建`5A + epoch-16教师`在单冲突场景中的闭环数据，包含教师组合和学生访问状态。
2. 离线训练Residual拟合`epoch-16动作 - 5A动作`，普通状态目标为零。
3. 先验证Residual的动作表达范围和场景级泛化，不启动长训练。
4. 让Actor B全程独立控制，只更新Residual，用TD3回报和逐渐衰减的教师约束训练。
5. 固定Actor B，在单冲突validation验证独立导航、碰撞和timeout。
6. 冻结A/B，只用0-edge和单冲突数据训练Gate。
7. 在未见的exact-edge-2及更复杂拓扑上做零样本验证。
8. 全部模型、阈值和协议冻结后才读取sealed test。

当前阶段是第1步。没有完成短pilot准入前，不启动多epoch训练。

## 7. 论文判定

论文主张至少需要同时成立：

- Actor B单独运行时具备完整导航能力，不能只把碰撞变成timeout；
- Actor B相对5A在单冲突上有稳定避让增益；
- Gate在0-edge上的能力下降不超过3个百分点；
- Gate在未见exact-edge-2上显著优于5A；
- 多冲突训练数据没有泄漏到Actor B或Gate训练阶段。

若Actor B全面优于Actor A，则不强行保留无意义的Gate；若可部署Gate仍不能转化局部
技能收益，则论文只能报告方法边界，不能用真值Gate冒充最终结果。
