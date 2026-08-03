# ICRA论文方向决策：两个Actor与一个Gate

状态：`diagnosis complete / two-Actor role definition pending on 2026-08-03`。

跨目录的当前结论、实验准入和下一步只以本文件为准。历史README中的“当前”或
“下一步”只代表当时判断。

## 1. 不变部分

最终系统固定为两个Actor和一个可部署Gate。当前待决定的是两个Actor分别承担什么
角色，不再混淆为同一个问题。

## 2. 两条候选路线

### 路线A：强弱独立Actor

```text
Actor A：独立完成普通/弱交互导航
Actor B：独立完成强交互导航
Gate：判断当前更适合哪个完整策略
```

优点是直观，符合最初设想。缺点是Actor B必须同时学会普通前进、冲突协调、恢复和
到达；历史多轮训练始终在碰撞与保守timeout之间退化。若继续，最小必要变化是给
Actor加入目标级相对运动/TTC信息，不能只继续调reward。

### 路线B：普通Actor与局部避险技能

```text
Actor A：5A，负责普通导航
Actor B：epoch-16，负责短时局部避险
Gate：判断何时进入避险、何时返回普通导航
```

它对应的候选论文主张是：

> 只在单冲突场景中学习局部避险，通过Gate反复调用，零样本处理训练中未见的多冲突
> 组合，同时保持普通导航能力。

创新不写成“两个Actor加一个Gate”，而写成：

```text
single-conflict skill learning -> compositional generalization to multi-conflict navigation
```

| 路线 | Actor A | Actor B | 当前证据 |
| --- | --- | --- | --- |
| A | 普通/弱交互独立导航 | 强交互独立导航 | 多次训练未同时解决碰撞和timeout |
| B | 5A普通导航 | epoch-16局部避险 | Oracle多冲突复用显著有效，learned Gate方向为正 |

当前证据更支持路线B，但在确认前不启动新Gate训练。把二者融合成单一Residual Actor
的R0测试已经失败，不属于两条候选路线，见
[失败记录](09_单冲突Residual组合主线/05_单冲突Residual_ActorB/README.md)。

## 3. 路线B的训练与测试边界

### 训练可见

- 五车场景，`conflict_edge_count=1`、`max_conflict_degree=1`；
- 普通路段和一次主要冲突都保留，用于学习Gate的进入与退出；
- 冻结5A与epoch-16，不再融合或解冻；
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

如果局部Actor或Gate在训练中读取多冲突场景，就不能再声称零样本组合泛化。

## 4. 已有证据

### 冻结组件

- 5A在248个固定0-edge validation场景上的full success为`0.875`。
- epoch-16只在单冲突场景训练，并在匹配调用协议下把full success从`0.421`
  提高到`0.700`。
- epoch-16独立全程运行会产生大量timeout，因此它是局部技能教师，不是最终Actor B。

完整路径审计发现epoch-16的2560场训练集中有`11`场因旧8秒标签截断实际为edge-2，
占`0.43%`。现有结果作为路线可行性证据保留；正式零样本主张需要在修正后的纯
edge-1训练集重新训练局部Actor。

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

1. 按完整路径重算冲突标签，修正edge-1视图中的3场多冲突泄漏。
2. 不再尝试24维Residual融合，也不继续扫描独立强Actor的reward。
3. 先讨论并冻结两个Actor的角色：强弱独立Actor，或普通Actor+局部避险技能。
4. 若选择局部技能路线，再使用已有目标级检测、跟踪、闭合速度和TTC证据改进Gate。
5. Gate训练仍只使用0-edge和单冲突数据，不读取多冲突validation或sealed test。
6. 冻结Gate后，在新的exact-edge-2确认集做一次零样本验证。
7. 通过预注册门槛后才扩大到edge>=3和sealed test。

完整场景、输入和训练责任诊断见
[全场景数据质量诊断](results/04_Gate前置验证/20260803_全场景数据质量诊断/README.md)。
没有完成标签修正和Gate短pilot准入前，不启动多epoch Actor训练。

## 7. 路线选择与论文判定

路线A只有在增加最小相对运动观测后，独立强Actor能稳定超过5A且不把碰撞转成timeout，
才值得恢复训练。当前没有这个证据。

路线B的论文主张至少需要同时成立：

- epoch-16在单冲突局部调用时相对5A有稳定避让增益；
- Gate在0-edge上的能力下降不超过3个百分点；
- Gate在未见exact-edge-2上显著优于5A；
- 多冲突训练数据没有泄漏到局部Actor或Gate训练阶段。

若可部署Gate仍不能转化局部技能收益，则路线B不能只用真值Gate成绩发表。此时再决定
是否给路线A增加最小观测，或更换论文方向。
