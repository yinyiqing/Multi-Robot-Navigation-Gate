# G12-R 公平参数匹配单Actor路线

状态：`route revised / R2-S0 passed / S1 repair pilot rejected`。
日期：`2026-08-07`。

## 1. 研究问题

本路线只回答：

> 在相近Actor参数量、相同24维部署观测、可比累计训练预算和不更窄的完整五车训练
> 分布下，一个单一大Actor能否达到两个专门化Actor与在线Gate的导航表现？

G12-P1只证明“5A函数保持扩宽 + fresh Critic + 无行为约束”的TD3更新发生训练坍塌，
不能回答容量问题。G12-R1只定位该坍塌是否由扩宽引起，也不进入论文主性能表。

此前把Gate的0-edge/edge-1数据边界直接用于正式大Actor训练是不合理的。Gate学习“何时
切换”，而单Actor必须学习完整五车导航；两者训练任务不同。正式大Actor从R2开始使用
完整课程和完整五车train split，不能因冲突拓扑删掉multi-edge训练场景。

## 2. 对照矩阵

| ID | Actor | 初始化与训练 | 用途 |
| --- | --- | --- | --- |
| `C0` | `24-800-600-2` | 冻结generalist-5a | 普通导航基线 |
| `R1` | `24-800-600-2` | 5A warm start，复现P1前40k协议 | 训练坍塌诊断 |
| `R2` | `24-1137-855-2` | 随机初始化，复现单车到五车完整课程 | 大Actor普通导航底座 |
| `R3` | `24-1137-855-2` | R2继续做完整五车混合分布40k pilot | 稳定性与能力保持检查 |
| `R4` | `24-1137-855-2` | 固定R3协议扩展到320k | 正式参数匹配单Actor |
| `Ours` | 两个`24-800-600-2`Actor + Gate | 冻结5A和epoch-16 | 当前方法 |

加宽Actor共`1,003,127`参数，两个冻结Actor合计`1,003,604`参数，相差`477`
（`0.0475%`）。该匹配只覆盖Actor bank，不包含Gate前端和GRU；论文必须同时报告部署时
实际执行的Actor MACs、Gate开销和模型总参数，不能把Actor参数匹配写成整个系统匹配。

## 3. 公平性原则

正式R2-R5同时满足：

1. **完整任务**：每个episode从起点到终止始终由同一个加宽Actor控制全部五车；没有
   oracle切换，也不把单冲突片段改成独立任务。
2. **完整训练分布**：R3/R4读取完整`fixed_v1/standard/train`和`dense/train`，不按
   0-edge、edge-1或multi-edge删除场景；强交互池只作为重采样分支，不是唯一训练集。
3. **相同部署信息**：Actor推理始终只读取本车24维观测。训练期Critic和loss mask可使用
   与epoch-16同级的邻车几何真值，但不得进入Actor输入。
4. **可比预算**：R2记录形成大Actor普通导航底座的全部agent samples；R4再使用
   `320,000` samples，对齐epoch-16专门化阶段。最终同时报告总样本和Actor更新次数。
5. **不使用评测调参**：G11-D2、G11-E和sealed test均不能选择R2-R4 checkpoint、
   sampling比例、anchor权重或学习率。

R2-R5的具体数据、采样和准入见[完整场景协议](FULL_SCENE_PROTOCOL.md)。

## 4. 执行阶段

### R0：归档P1并完成机制诊断

状态：`completed / archived`。完整记录见[R0诊断](R0_P1_DIAGNOSTIC.md)。

- 初始加宽Actor与5A最大输出误差：`4.62e-7`；
- Epoch 1 full success：`0.717`；
- Actor更新20k samples后的Epoch 2：`0.050`；
- 动作漂移到近似固定`[0.992, -0.913]`；
- 结论：训练坍塌，不得表述为容量失败。

### R1：原宽度训练稳定性控制

状态：`completed / archived diagnostic`。冻结参数见[R1协议](R1_PROTOCOL.md)，完整结果见
[R1诊断](R1_DIAGNOSTIC.md)。

R1只改变P1的Actor宽度并硬停止于40k。原宽度full success从`0.717`降至`0.283`，证明
主要问题是无约束TD3/fresh Critic，而不是参数量翻倍；P1降至`0.050`，说明扩宽可能放大
不稳定性，但单seed不足以形成独立因果结论。R1不是正式大Actor，也不决定完整场景中
单Actor的最终能力。

### R2：加宽Actor从头完成普通导航课程

- Actor和Critic随机初始化，固定架构`24 -> 1137 -> 855 -> 2`；
- 按历史有效链路复现单车局部导航到2A/2D、3A/3D2和五车5A的完整课程；
- 每一阶段在启动前冻结case manifest、seed、样本预算、reward和validation；
- 不从5A做函数扩宽warm start，不读取Gate数据、D2、G11-E或sealed test；
- 保存每阶段全部sample数和Actor更新数，不能只报告最后320k。

S0已经完成：100k内五次单车broad validation full success为
`0.983/1.000/1.000/0.992/1.000`，冻结epoch 2 best。启动S1更新前先评测固定困难case，
只修复实际缺口；不得因为历史课程存在S1名称就自动消耗80k预算。

固定困难case诊断也已完成：`126/126`场full success为`0.5714`，其中`20/42`个
stage-case项达到repair条件。基础目标推进通过，缺口系统性集中在贴墙与脱离恢复。S1将只
训练这些缺口，并在每段更新后回测broad n1；该诊断不是大Actor最终容量结果。

首段repair-only 20k已经完成，但broad n1降至`69/120=0.575`并出现49次timeout，候选
按预注册门槛拒绝。S0 epoch 2 best仍是唯一回滚点。固定困难case不再阻塞多车课程；S2
直接从S0 best进入两车完整broad训练，不能从失败S1 checkpoint续训。S2首段配置见
[S2两车协议](R2_S2_N2_PROTOCOL.md)。

进入R3的最低条件：

1. 新建的G12内部完整场景validation上，0-edge full success相对5A下降不超过`0.03`；
2. 总体agent success下降不超过`0.02`，timeout增加不超过`0.02`；
3. standard和dense两个来源均无明显单侧退化；
4. 动作无持续饱和，训练无NaN或Critic爆炸。

单seed未达到条件时先审计课程复现和优化稳定性，不能直接写“大Actor容量不足”。

### R3：完整五车混合分布40k pilot

- broad stream：完整`standard/train`和`dense/train`，两来源等概率；
- interaction replay stream：`strong_interaction_curriculum_v1/full_train`；
- 两条stream按预注册`1:1`采样，strong pool是broad stream的子集，明确作为重采样披露；
- 加宽Actor全程控制所有机器人；冲突标签只可用于训练期行为保持mask和分层统计；
- 非交互状态锚定R2，交互状态允许策略更新；Actor loss使用Q尺度归一化和梯度裁剪；
- 20k只作诊断checkpoint，40k作pilot判断，不根据第一次波动改超参数。

继续到R4至少满足：

1. 0-edge full success下降不超过`0.03`；
2. 完整validation总体full success不低于R2，且edge-1或multi-edge至少一个分层改善；
3. standard与dense均不出现超过`0.05`的full success下降；
4. timeout增加不超过`0.03`；
5. 动作饱和、梯度、Q值与真实回报诊断正常。

### R4：320k正式完整场景训练

R3通过后保持数据、`1:1`采样、loss和优化器不变，扩展到`320,000 agent samples`。
每40k在G12内部完整场景validation评估。模型选择采用约束目标：

1. 先满足0-edge能力保持、standard/dense来源保持和timeout限制；
2. 在满足约束的checkpoint中最大化完整validation总体full success；
3. edge-1和multi-edge均报告，不按某一个孤立峰值挑模型；
4. 不用D2、G11-E或sealed test重新选择checkpoint。

### R5：复核与最终比较

- 正式协议至少3个seed；主seed预注册，不从seed中挑最好结果；
- 模型冻结后才按顺序运行G12内部holdout、G11-D2、G11-E和sealed test；
- 与5A、epoch-16 always-on、规则Gate、A1、B2、最终Gate及2m oracle重新运行同一个冻结
  manifest；scenario ID、顺序、evaluation seed列表、重复次数、物理参数和终止条件完全一致；
- 逐场保存`scenario_id + repeat`，分析前强制审计manifest哈希及缺失、重复、错序ID；审计
  不通过时不得生成方法对比表；
- 同时报告0-edge、edge-1、multi-edge、standard/dense、平均步数和碰撞；
- 若大Actor达到或超过双Actor+Gate，如实收窄或否定“专门化优于容量”的主张。

## 5. 可写结论边界

只有R4/R5完成后，才能判断参数匹配单Actor是否达到双Actor+Gate。允许的表述是：

```text
在固定的24维单帧Actor观测、近似Actor参数量、披露的完整训练分布和累计样本预算下，
参数匹配单Actor与专门化双Actor在线路由的性能比较为……
```

不得表述为“任何大Actor都无法解决多机器人导航”。R2不能稳定形成普通导航底座时，
结论是训练流程尚不足以形成公平容量对照，而不是模型容量失败。

## 6. 两周执行顺序

| 顺序 | 工作 |
| --- | --- |
| 1 | 完成并归档R1，形成训练稳定性结论（已完成） |
| 2 | 构建并冻结G12完整场景内部validation，审计与D2/G11-E互斥（已完成） |
| 3 | 盘点历史5A课程的case、预算和有效节点，冻结R2逐阶段运行manifest（已完成） |
| 4 | R2从头训练加宽Actor并通过普通导航准入（S0已完成并通过） |
| 5 | R3完整五车混合分布40k pilot |
| 6 | R3通过后执行R4 320k和复核seed |
| 7 | 冻结模型后完成统一评测、统计、图表和论文写作 |

Gate主线结果和论文写作优先于容量对照。任何阶段出现协议错误、动作饱和或连续退化时，
立即停止并保留诊断，不进行无边界超参数扫描。
