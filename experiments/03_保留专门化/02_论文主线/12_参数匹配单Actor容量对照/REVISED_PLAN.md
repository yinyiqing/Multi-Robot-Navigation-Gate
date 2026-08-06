# G12-R 公平参数匹配单Actor路线

状态：`route agreed / R0 archived / R1 preregistered`。
日期：`2026-08-06`。

## 1. 研究问题

本路线只回答：

> 在相近Actor参数量、相同部署观测、合理训练流程和可比训练预算下，一个单一大Actor
> 能否达到两个专门化Actor与在线Gate的导航表现？

当前G12-P1不回答这个问题。它证明加宽初始化没有破坏5A，但在无行为约束的fresh-Critic
TD3更新后发生动作饱和和性能坍塌，只能作为训练稳定性诊断。

本对照只匹配Actor bank的参数量和24维单帧Actor观测，不声称匹配Gate感知前端、GRU
或整个系统参数量。最终论文必须明确这一范围。

## 2. 对照矩阵

| ID | Actor | 初始化与训练 | 用途 |
| --- | --- | --- | --- |
| `C0` | `24-800-600-2` | 冻结generalist-5a | 普通导航基线 |
| `R1` | `24-800-600-2` | 5A warm start，复现P1前40k协议 | 判断P1坍塌是否与扩宽有关，只作诊断 |
| `R2` | `24-1137-855-2` | 随机初始化，完整普通导航课程 | 参数匹配单Actor的Stage A |
| `R3/R4` | `24-1137-855-2` | R2继续做0-edge/edge-1联合训练 | 参数匹配单Actor正式候选 |
| `Ours` | 两个`24-800-600-2`Actor + Gate | 冻结5A和epoch-16 | 当前方法 |

加宽Actor共`1,003,127`参数，两个冻结Actor合计`1,003,604`参数，相差`477`
（`0.0475%`）。

## 3. 执行阶段

### R0：归档当前P1并完成机制诊断

状态：`completed / archived`。完整记录见[R0诊断](R0_P1_DIAGNOSTIC.md)。

- 初始加宽Actor与5A最大输出误差：`4.62e-7`；
- Epoch 1 validation：full success `0.717`；
- Actor更新20k samples后的Epoch 2：full success `0.050`；
- 同一replay状态上，平均动作从`[0.546, -0.040]`漂移到`[0.992, -0.913]`；
- 结论：训练坍塌，不得表述为容量失败。

### R1：原宽度训练稳定性控制

状态：`preregistered`。冻结运行参数见[R1协议](R1_PROTOCOL.md)。

目的：只改变网络宽度，复现P1的5A warm start、fresh Critic、数据、seed、学习率、
20k Critic warm-up和40k总预算。

停止与解释：

- 原宽度也坍塌：根因是无约束TD3/fresh Critic，不是扩宽；
- 只有加宽Actor坍塌：重点修正扩宽初始化和新增参数优化尺度；
- R1不进入论文主性能表，只决定R2/R3的稳定性设计。

### R2：加宽Actor从头完成普通导航课程

- Actor和Critic均随机初始化；
- 固定架构`24 -> 1137 -> 855 -> 2`；
- 复现形成5A的单车到五车课程顺序、reward、场景边界和阶段validation；
- 不读取Gate标签、其他机器人真值或sealed test；
- Stage A checkpoint只按普通导航validation选择。

进入R3的最低条件：

1. 0-edge full success相对同场5A下降不超过`0.03`；
2. agent success下降不超过`0.02`；
3. timeout增加不超过`0.02`；
4. 动作输出无单侧饱和，训练无NaN或Critic爆炸。

单个seed未达到条件时只能说明该次课程未通过，不能直接形成“大Actor容量不足”的论文
结论。先审计训练，再决定是否使用预注册的第二seed复核。

### R3：0-edge/edge-1联合训练40k稳定性pilot

- 数据：navigation-train内部`standard/dense x 0-edge/edge-1`四层等权；
- 单一Actor逐机器人、逐时刻全程控制，不使用Gate；
- 冲突拓扑只用于训练采样和loss掩码，不进入Actor输入；
- 参考策略：冻结R2 Stage A checkpoint；
- 0-edge样本加入行为保持项，edge-1样本允许学习交互动作；
- Actor loss采用Q尺度归一化，加入梯度裁剪并记录动作漂移与饱和比例；
- 20k只作诊断checkpoint，40k作pilot判断，不在第一次下降时推导容量结论。

建议目标：

```text
L_actor = -normalized_Q(s, pi(s))
          + lambda_keep * I[0-edge] * ||pi(s) - pi_stageA(s)||^2
```

`lambda_keep`、梯度裁剪阈值和Q归一化系数必须在启动前写入运行manifest。最多允许一个
小规模smoke检查数值量级，不允许依据闭环full success扫描超参数。

R3继续条件：

1. 0-edge full success下降不超过`0.03`；
2. edge-1 full success或collision至少出现一个明确正向趋势；
3. timeout不增加超过`0.03`；
4. 任一动作维度在绝对值`>=0.98`的比例不得出现持续异常增长；
5. Critic预测上升不能与真实回报持续反向。

### R4：320k正式单Actor容量训练

R3通过后，保持所有超参数和数据边界不变，将联合训练扩展至`320,000 agent samples`，
与interaction-epoch16的后续训练预算对齐。每`40k`在固定内部validation评估。

模型选择采用约束目标：

1. 先满足0-edge能力保持和timeout限制；
2. 在满足约束的checkpoint中最大化edge-1 full success；
3. 不按孤立峰值选择，不读取D2或sealed test调参。

### R5：复核与最终比较

- 正式协议至少运行`3`个seed；
- 主seed预注册，不从多个seed中挑最好模型；
- 模型冻结后才在G11-D2 validation、G11-E multi-edge和最终sealed test上运行；
- 与5A、epoch-16 always-on、A1、B2和2m特权距离规则使用相同场景和配对统计；
- 同时报告0-edge、edge-1、multi-edge、standard/dense分层结果。

## 4. 最终指标

主指标：`full_success_rate`。

必须同时报告：

- `agent_success_rate`；
- `collision_rate`；
- `unresolved_rate`；
- `timeout_episode_rate`；
- 平均步数；
- 动作饱和比例；
- 参数量、Actor MACs和训练样本预算；
- 多seed均值、标准差和同场配对检验。

## 5. 可写结论边界

只有R4/R5完成后，才能判断参数匹配单Actor是否达到双Actor+Gate。允许的表述是：

```text
在本文固定的24维单帧Actor观测、训练信息和预算下，参数匹配单Actor与专门化双Actor
在线路由的性能比较为……
```

不得表述为“任何大Actor都无法解决多机器人导航”。若R2无法稳定复现普通导航，结论应是
训练流程尚不足以形成公平容量对照，而不是模型容量失败。

## 6. 两周执行安排

| 时间 | 工作 |
| --- | --- |
| Day 1 | R0归档、R1原宽度40k诊断、冻结R2/R3超参数 |
| Day 2-4 | R2加宽Actor从头课程训练与普通导航准入 |
| Day 5 | R3联合训练40k稳定性pilot |
| Day 6-8 | R4扩展到320k |
| Day 9-10 | 复核seed；未通过则如实停止，不继续扫参 |
| Day 11 | 冻结模型并完成D2/G11-E对比 |
| Day 12-14 | sealed test、统计、图表和论文写作 |

执行优先级：Gate主线结果和论文写作高于容量对照。任何阶段若出现协议错误、动作饱和或
连续退化，应立即停止并保留诊断，不得占用剩余时间进行无边界调参。
