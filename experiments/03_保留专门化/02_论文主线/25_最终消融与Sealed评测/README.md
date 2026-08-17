# G25 最终闭环消融与 Sealed 评测预注册

状态：`方案冻结，尚未启动`
登记日期：`2026-08-17`

## 1. 目的与边界

本阶段只补齐论文的闭环因果消融、部署成本和一次性 sealed test。主方法继续冻结为
`generalist-5a + avoidance-epoch16 + router-b2`。**禁止训练或更新任何 Actor**；允许为
结构消融从同一冻结数据重新训练 Router，但消融结果不得反向修改 B2 的结构、阈值或 checkpoint。

Dense256 和 G17 已参与模型选择，全部现有结果均定义为 development validation evidence。
它们可以回答“为什么选择当前方法”，不能承担最终确认性结论。只有本协议冻结后的一次性
sealed test 可以形成投稿主结论。

## 2. 冻结 artifact

| 组件 | SHA-256 |
|---|---|
| generalist-5a | `fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5` |
| avoidance-epoch16 | `6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b` |
| router-b2 | `fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52` |
| G0 soft detector | `0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56` |

B2 固定为 82 维输入、8 帧 GRU、Gate stride 2、on/off 阈值 `0.43/0.33`、最短保持
3 个 Gate 帧。除下列显式消融外，任何运行不得改变这些值。

## 3. 先做 validation 闭环消融

所有消融先在现有 Dense256 validation manifest 和相同仿真 seed `20260810` 上运行，使用
同一场景顺序、物理步长、终止条件和逐场结果格式。不得用结果修改主方法 B2。

| ID | 方法 | 唯一变化 | 需要训练 |
|---|---|---|---:|
| V0 | 5A | 普通 Actor 全程执行 | 否 |
| V1 | epoch16 always-on | 交互 Actor 全程执行 | 否 |
| V2 | min-LiDAR rule | 仅用原始本机 LiDAR 最小距离、滞回和保持切换 | 否 |
| V3 | TTC rule | 仅用 G0/G1 的可部署软候选与 TTC/CPA 连续量按固定规则切换 | 否 |
| V4 | B2 single-frame | 保持同一82维输入和GRU hidden width，但序列长度为1且每帧重置hidden | 仅 Router |
| V5 | B2 no-action-difference | 保留 8 帧 GRU，但删除两 Actor 候选动作及差分，输入为同一 76 维局部表征 | 仅 Router |
| V6 | B2 no-hysteresis/hold | 使用原 B2 checkpoint，单阈值 `0.43`，最短保持为 0 | 否 |
| V7 | A1 | 无 student-rollout 聚合 | 否 |
| V8 | B2/PIRoute | 冻结主方法 | 否 |
| V9 | 2 m privileged distance rule | 使用真值距离，仅作不可部署诊断，不称性能上界 | 否 |
| V10 | R2B-best | 同 5A recipe 的流程匹配容量控制 | 否 |

V2/V3 的阈值只能在 navigation-train 内部冻结数据上选择，目标依次为：先满足 0-edge 误触发
上限，再在该约束内最大化 interaction recall。不得在 Dense256 或 sealed test 上扫描阈值。
V4/V5 使用与 B2 相同的原始 shard、student shard、场景权重、类别权重、训练 seed 和
checkpoint 选择规则；只改变表中指定因素。V4 仍使用同一 GRU cell 和输出头，但训练与推理时
序列长度固定为1、每个Gate帧重置hidden，以隔离“Router显式8帧历史”因素。G1已有的短窗
跟踪特征仍保留，因此该消融必须称为 `no Router history`，不能称整个系统完全无时序。

## 4. 部署成本审计

在固定硬件、batch size 1、预热 200 次、计时 2000 次的协议下分别测量：

- 5A 单 Actor 前向延迟；
- 两个 Actor 顺序前向和可用时的并行前向延迟；
- G0/G1 前端、Router 和端到端控制延迟；
- 参数量、理论 MACs/FLOPs、Gate stride、控制频率和峰值显存。

报告中不得用“双 Actor 参数量约等于 R2B”替代运行时成本。PIRoute 每个路由帧需要两个 Actor
候选动作，这是相对单 Actor 的明确部署代价。

## 5. Sealed test 一次性协议

### 5.1 数据与 seed

主要确认集固定为 `fixed_v1/dense/test.json.gz` 原始冻结顺序的前 256 个场景，不根据 Actor、
Router、冲突边数或历史结果筛选。生成只包含这 256 个场景的 manifest 后，必须先记录路径、
场景数和 SHA-256，再启动任何策略。完整混合 test 只作为时间允许时的次要外部有效性检查，
不得覆盖 dense 主结论。

环境 repeat seed 预先固定为：

```text
20260901, 20260902, 20260903
```

三个 repeat 均对 V0-V10 使用相同的场景和 seed。方法顺序采用循环移位，避免某一方法总在
仿真进程的相同位置运行。失败重启只允许从已落盘的最后完整 scene ID 继续，不得重跑后择优。

### 5.2 假设与指标

唯一主要假设：在 dense sealed test 上，B2/PIRoute 的 episode-level full success 高于 5A。
主要效应量是二者 full-success 差值。collision、timeout、steps、interaction share 和切换次数
均为预注册次要指标，必须与成功率同表报告，不得因方向不利而省略。

统计单位是 scene。三个 repeat 不能当作 768 个相互独立场景：

1. 报告每个 repeat 的点估计和改善/退化/持平数；
2. pooled McNemar 只作参考；
3. 主推断使用按 scene 聚类的 paired bootstrap 95% CI 和 scene-level sign-flip test；
4. collision、timeout、steps、interaction share 同样使用 scene-cluster paired bootstrap CI；
5. 按 0/1/2/3+ 冲突边分层只作预注册异质性分析，不分别宣称新的主要显著性结论。

V1-V7、V9-V10 是机制/边界对照。B2 对 A1、single-frame 或 no-action-difference 的比较即使
显著，也不改变主方法或主要假设；B2 对 student-rollout 的现有非显著结论必须如实保留。

## 6. 图表与停止规则

最终输出固定包括：

- success-collision-steps Pareto 图，validation 与 sealed 分面显示；
- full success、collision、timeout、steps、interaction share 的配对差值及 95% CI；
- 至少一个成功路由、一个晚退出和一个误触发案例的时间轴；
- 参数量、前向次数、FLOPs、延迟和控制频率表。

读取 sealed 后禁止：重新选择 Actor/Router、调整阈值、删除不利 seed、改变主要指标、改变
dense 场景范围或新增用于挽救结论的模型。若主要假设不成立，论文必须按阴性结果修改，而不是
返回 Dense256 重新选型。

## 7. 容量基线披露

R2B-best 只回答“相同 5A recipe、数据、预算和优化流程下的参数扩展是否成功”，因为其 best
位于有效 Actor 更新前。补充材料必须同时披露：跨协议 R2-10k 在同一 Dense256 validation 上
达到 full success `0.5273`，高于当前 PIRoute 的 `0.4258`；R2-10k 使用额外的
`n1 -> n2 -> n3 -> n5` 课程和不同训练预算，因此不能进入公平排名，也不能被隐藏后再声称
“训练充分的大单 Actor 不如双 Actor”。

## 8. 执行顺序

1. 实现并单元测试 V2-V6，核对它们只改变登记因素；
2. 运行 Dense256 validation 闭环消融和部署成本审计；
3. 冻结脚本、容器/环境、manifest、顺序与统计输出，完成一次 dry-run；
4. 一次性运行三个 seed 的 dense sealed test；
5. 只运行预注册统计脚本并生成最终表图。

本文件不授权任何 Actor 训练。
