# G26 数量泛化与外部切换基线

状态：`已登记，等待 G25 sealed 完成后执行`

登记日期：`2026-08-19`

## 1. 目的与边界

本阶段补充两个不改变论文主方法的问题：

1. 冻结的五车系统能否不经任何更新直接部署到 3 车和 7 车；
2. PIRoute 是否优于一个来自已发表工作的通用分布外切换思路。

主方法继续冻结为 `generalist-5a + avoidance-epoch16 + router-b2`。本阶段禁止训练或更新
Actor，禁止微调 B2，禁止根据 G25 sealed 结果修改阈值、场景范围或指标。G25 的七方法、
三 repeat sealed test 仍是唯一确认性实验；G26 是单独预登记的补充外部有效性实验，不修改、
插队或中断 G25 队列。

## 2. 冻结组件

| 组件 | SHA-256 |
|---|---|
| generalist-5a | `fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5` |
| avoidance-epoch16 | `6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b` |
| router-b2 | `fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52` |
| G0 soft detector | `0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56` |

B2 的 82 维输入、8 帧 GRU、stride 2、`0.43/0.33` 阈值和最短保持 3 个 Gate 帧全部保持
不变。部署输入继续只包含本机可部署信息。

## 3. Q1：3/7 车数量泛化

### 3.1 要回答的问题

5A、epoch16 和 B2 均在五车流程中确定，但底层策略共享参数，Router 也逐机器人独立执行。
因此 Q1 检查冻结系统是否依赖固定的五车规模：

- 3 车检查低负载下是否因误触发交互 Actor 而破坏普通导航；
- 7 车检查更大规模、更高并发交互下是否仍相对 5A 保持收益。

由于 Actor 和 Router 都没有在 3/7 车上更新，若结果成立，只能称为**冻结系统的机器人数量
泛化**，不能单独归因于 Router，也不能声称对任意数量成立。

### 3.2 数据与预算

为 3 车和 7 车分别生成 `128` 个冻结测试场景，使用同一地图、机器人动力学、目标采样规则、
静态障碍配置、控制周期、最大步数和终止定义。生成器与 launch 配置必须先通过以下审计：

1. 起点、目标和静态路径可行性检查；
2. 不同车辆数下碰撞半径、速度限制和传感器配置一致；
3. 记录 manifest SHA-256、场景生成 seed 和 0/1/2/3+ 冲突边构成；
4. 在运行策略前冻结场景原始顺序，不按冲突数或预跑结果筛选。

评测 seed 固定为 `20260911`、`20260912`。每个车辆数只运行以下两个方法：

```text
Q1-N  5A always-on
Q1-P  epoch16 + B2/PIRoute
```

最小预算为 `2 vehicle counts x 128 scenes x 2 repeats x 2 methods = 1024 episodes`。
5 车结果直接引用 G25 sealed，不重复运行。若时间允许，epoch16 always-on 和 2 m 特权距离
规则只能作为预先标记的补充诊断统一加入两个车辆数，不能只在结果较好的一侧追加。

### 3.3 指标与解释

主要报告每个车辆数内 B2 相对 5A 的配对差值，而不是直接比较 3/5/7 车的绝对 full success。
固定报告：

- full success 与 agent success；
- 每车 collision、unresolved 和 episode timeout；
- raw、paired-success 和 penalized completion steps；
- interaction Actor 占比与切换次数；
- 0/1/2/3+ 冲突边分层结果。

车辆增多会机械性降低 full success，也会改变冲突图分布，因此不能把 7 车更低的绝对成功率
直接解释为数量泛化失败。核心判断是相同车辆数、相同场景和相同 repeat 下，B2 相对 5A 的
收益方向、碰撞变化及效率代价。

## 4. E1：Normalizing-flow 外部切换基线

### 4.1 文献来源与定位

候选工作为 Matsumoto et al., “Crowd-Aware Robot Navigation with Switching Between
Learning-Based and Rule-Based Methods Using Normalizing Flows,” IROS 2024，
DOI：`10.1109/IROS58592.2024.10802676`。

该工作用 normalizing flow 识别学习策略未覆盖的异常状态，并在学习方法和规则方法之间在线
切换。它与 PIRoute 的共同问题是“何时离开默认策略”，但监督和语义不同：NF 判断分布外状态，
PIRoute 估计由特权标签监督的交互状态代理。

截至登记时未找到作者公开代码，且现有公开元数据不足以保证逐项复现。因此本文只能将实现称为
`normalizing-flow-based switching baseline inspired by Matsumoto et al.`，不得称为官方实现、
严格复现或作者方法的完整性能。

### 4.2 实现冻结前置条件

E1 不更新两个 Actor。正式实现前先取得并核对全文，记录以下内容：

- flow 的输入、结构和训练数据；
- 异常分数与阈值确定方式；
- 切换频率、退出条件及是否包含保持机制；
- 原方法是否使用地图、行人真值、跟踪轨迹或其他本项目部署时不可用的信息。

若全文方法能够映射到当前部署输入，则按原文做最接近复现；若不能，使用以下保守替代协议并
显式标注为 literature-inspired baseline：只用 navigation-train 中 5A 成功轨迹的本机可部署
观测拟合 flow，阈值只由训练数据的 nominal likelihood 分位数冻结，低似然时调用 epoch16，
否则调用 5A。不得使用 Dense validation、G25 sealed 或 G26 测试结果选择 flow、阈值或
checkpoint，也不得使用 2 m 真值标签调节阈值。

### 4.3 同场评测

E1 使用冻结的 Dense test 后续原始顺序场景，不进入 G25 已冻结的前 256 场确认性主表。
在首次读取场景和运行方法前，固定连续 `128` 场、记录切片边界和 manifest SHA-256。评测
seed 固定为 `20260921`、`20260922`，同场运行：

```text
E1-N  5A always-on
E1-F  5A + epoch16 + normalizing-flow switch
E1-P  5A + epoch16 + B2/PIRoute
```

预算为 `128 scenes x 2 repeats x 3 methods = 768 episodes`。比较重点为 E1-P 与 E1-F；
5A 用于校验该补充测试片的难度和两种切换方法的绝对收益。指标与 G25 保持一致，包括 full
success、agent success、collision、timeout、三种 steps、interaction share 和 switches。

E1 只能说明当前实现的通用 OOD 切换是否能解释 PIRoute 的收益，不能据此否定所有 normalizing
flow 方法，也不能声称全面超过 IROS 2024 原系统。

## 5. 不采用的外部基线

- 旧 ORCA/NH-ORCA：全向速度到差速动作的换算失稳，且 `0.24 m` 静态路径净空与约
  `0.40 m` 有效半径不一致；旧结果不能进入正式比较。
- CADRL/SARL：通常需要可辨识的邻居状态并采用不同动力学，和当前本机 LiDAR 部署约束不等价。
- DWA/TEB：需要额外地图、代价地图和规划器配置，比较的是完整导航栈而不是冻结 Actor 的
  Router；在当前投稿时间内优先级低于 E1。
- Learning When to Switch：原方法学习目标策略能否成功接管，并要求策略间共同可切换区域；
  当前两个冻结 Actor 没有同口径的接管成功标签，直接套用会引入新的训练问题。
- Shared Trajectory-Based Multi-Policy：生成和评价共享候选轨迹，不是对冻结 Actor 做选择，
  不能作为只替换 B2 的同模块对照。

## 6. 执行顺序与停止规则

1. 等待 G25 sealed 七方法、三 repeat 全部完成、归档并完成冻结统计；
2. 审计 3/7 车 launch、场景生成器和逐场结果格式，冻结 Q1 manifests 后运行 Q1；
3. 获取并核对 IROS 2024 全文，冻结 E1 的 flow 实现、训练 split 和测试切片；
4. 完成 flow 训练审计后运行 E1 三方法同场评测；
5. 统一生成补充表格，不把 G26 探索性结果并入 G25 确认性 p 值或主假设。

任一阶段发现实现依赖真值、场景错序、车辆数配置不一致或测试结果参与阈值选择时立即停止并
修复协议，不保留择优结果。本文件不授权任何 Actor 训练。
