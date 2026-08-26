# G26 数量泛化与外部切换基线

状态：`Q1 已完成并冻结统计；E1 manifest 已冻结，三方法同场评估进行中`

登记日期：`2026-08-19`

更新时间：`2026-08-26`

## 2026-08-21 Q1 完成与证据边界

Q1 已按冻结协议完成：3车和7车分别使用128个独立生成且通过Gazebo reset的场景，5A与B2
各运行两个repeat（`20260911/20260912`），合计`1024 episodes`。全部8个结果文件均通过
`(128,17)`形状、manifest顺序和每车终止记账检查；期间未训练或更新Actor、B2，也未使用
2 m特权距离规则。统计文件为`local_data/q1/results/q1_statistics.json`，SHA-256为
`c02c17fa49509d5a31fd70f0862508fc7d1d04855f0047794c5019bdd0b835fb`。

Q1的scene-cluster BCa区间与sign-flip沿用G25计算口径，但Q1登记时未单独冻结显著性门槛，
因此统一标为补充性探索推断，不进入G25确认性主检验。主要结果如下：

| 车辆数 | 方法 | full success | agent success | collision | timeout | raw steps | I占比 | switches |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 5A | 0.7617 | 0.8776 | 0.1211 | 0.0039 | 22.54 | 0 | 0 |
| 3 | B2 | 0.7930 | 0.9010 | 0.0951 | 0.0117 | 36.81 | 0.6820 | 4.54 |
| 7 | 5A | 0.0586 | 0.5603 | 0.4397 | 0 | 16.54 | 0 | 0 |
| 7 | B2 | 0.0742 | 0.6557 | 0.3432 | 0.0078 | 33.96 | 0.6516 | 11.17 |

3车B2相对5A的full-success差值为`+0.0313`，BCa 95% CI
`[-0.0273,+0.0938]`，探索性sign-flip `p=0.3853`；7车差值为`+0.0156`，CI
`[-0.0273,+0.0547]`，`p=0.5749`。两者都只能称正向点估计，不能称已证明整队成功率提高。
7车的agent-success差值为`+0.0954`（CI `[+0.0614,+0.1272]`），collision差值为
`-0.0965`（CI `[-0.1283,-0.0625]`），支持更高并发下的每车完成与安全收益；3车对应区间
均跨0，只支持方向性结果。

代价与G25一致：3/7车raw steps分别增加`14.27/17.42`，I占比为`68.2%/65.2%`。
3车双方共同成功的173个scene-repeat pair中，B2平均增加`11.88`步（CI
`[7.10,16.54]`）；7车只有2个双方共同成功pair，paired-success steps不具稳定解释价值。
penalized completion steps在3/7车的差值区间都跨0。timeout点估计各增加`0.0078`，不能
声称timeout改善。

场景分布用于限定解释：3车冲突边`0/1/2/3+ = 61/58/8/1`，B2收益主要出现在1-edge；
7车有`123/128`场属于3+ edge，冲突边均值`5.3984`。因此7车极低的绝对full success不能与
五车G25直接横向比较，也不能单独解释成数量泛化失败。Q1最终支持的表述是：冻结系统从五车
零更新部署到3/7车后，B2相对5A保持了full-success正向点估计，并在7车高冲突条件下明确改善
每车成功率和碰撞，但没有证明3车或7车的整队成功率提升。

3车manifest 0-edge的61场需要单独披露：5A/B2 full success为`0.9344/0.9262`，agent
success均为`0.9699`，collision为`0.0273/0.0246`；任务结果基本保持，但B2的I占比仍为
`68.8%`，raw steps增加`13.82`。manifest 0-edge只表示名义同步路径没有预测冲突，不保证实际
闭环轨迹完全无交互；即便如此，这一高调用率仍不支持“低负载下Router稀疏调用或几乎不误触发”
的表述。Q1数量外推主要保持了结果方向，没有保持调用效率或证明Gate具有良好的数量校准。

## 2026-08-21 E1 文献审计与实现冻结

已核对DOI `10.1109/IROS58592.2024.10802676`的Crossref、OpenAlex、Semantic Scholar、
Unpaywall、arXiv、DBLP、作者ORCID、IEEE入口和公开代码索引。可访问证据确认原工作以当前状态
likelihood为依据，用normalizing flow判断是否处于学习方法预期分布，并在learning-based与
rule-based方法之间切换；参考文献包含Graph Normalizing Flows、RealNVP和GATv2。IEEE全文
不是开放获取，公开索引没有作者全文或官方代码，因此无法核对其精确图状态、网络层数、阈值、
切换保持机制和rule-based控制器。结构化审计见`local_data/e1/literature_audit.json`。

因此E1不再称严格复现，固定名称为`normalizing-flow-inspired switching baseline`。它只复现
“用nominal-state likelihood决定何时离开默认策略”的切换思想，并在本项目中保持5A与epoch16
两个Actor完全相同，只替换Router。原论文切向rule-based方法，而本基线切向冻结epoch16；这项
差异必须在正文和补充材料中披露，E1不能代表原论文系统的绝对性能。

在读取E1测试场景前，冻结以下实现，不按后续闭环结果修改：

1. 训练源只使用G11-A1的640场navigation-train冻结5A轨迹。train manifest SHA-256为
   `a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026`，640场逐场结局日志
   SHA-256为`8c58e0b15e69db1913a636742ee451d26774ed64293d2bb251de686144c39c51`，A1 shard语义
   digest为`7b50f36611629332f89515b8035ce5576c100217a80536b7fe13601ce839fa4e`；
2. 只保留5A整队full-success的`442/640`场。按`scenario_pool + interaction_band`四层，使用
   `SHA256("20260821:" + scenario_id)`排序后，各层前80%作为fit、其余作为calibration；固定为
   `352/90`场、`15011/4059`帧；
3. 输入只取shard中每车24维`frame_actor_states`，即部署时已有的本机Actor观测。不使用Oracle
   标签、其他机器人真值、ego pose、场景名、冲突图、G0检测标签或测试数据；
4. flow固定为24维RealNVP：6个交替mask affine coupling block，每个coupling的scale/shift
   MLP为`24 -> 128 -> 128 -> 48`，ReLU，log-scale经`tanh`限制到`[-2,2]`；标准正态base；
5. fit特征均值/标准差只由fit split计算。优化固定为CPU、seed `20260821`、Adam
   `lr=1e-3`、weight decay `1e-6`、50个full-batch epoch；每场总权重相同，长轨迹不因帧多
   获得更大总权重，不做epoch、层数或学习率扫描；
6. checkpoint固定使用第50 epoch，不读取calibration选择模型。唯一阈值为calibration NLL的
   95分位数；每个环境步若`NLL > threshold`则调用epoch16，否则调用5A。无滞回、无最短保持、
   无测试后阈值调整；日志中的Gate score定义为calibration NLL empirical CDF；
7. flow checkpoint、阈值、训练摘要和SHA-256全部冻结后，才允许首次生成Dense test原始顺序
   `256:384`的128场E1 manifest。测试seed仍为`20260921/20260922`，同场运行
   `5A / NF-inspired / B2`共`768 episodes`。

E1训练本身只更新flow，不更新Actor、G0或B2。上述检查已通过，manifest已固定并开始读取
E1 test；从此不再修改上述任一实现参数。

## 2026-08-26 E1 评估进行中

E1评估按单Gazebo实例串行运行`2 seeds x 3 methods x 128 scenes = 768 episodes`。当前首个
组合`g26_e1_5a_s20260921`正在增量执行，其余组合尚未开始。运行入口为
`scripts/start_g26_e1_evaluation.sh`，实时总日志为
`logs/active/g26-e1-evaluation/runner.log`，单次方法日志为同目录下的
`g26_e1_<method>_s<seed>_attempt*.log`；完成后日志归档至
`logs/archive/test/g26_e1_evaluation/`。结果临时写入
`local_data/e1/results/`，完成后再运行`scripts/analyze_g26_e1_results.py`生成统计。
评估期间不读取中间结果来调整任何冻结组件。

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

1. G25 sealed 七方法、三 repeat 已完成、归档并完成冻结统计；
2. 3/7 车场景审计、Q1 manifests、零更新评测和探索性统计已完成；
3. IROS 2024 文献审计、E1 flow 实现、训练 split 和测试切片已冻结；
4. 当前运行 E1 三方法同场评测，完成后执行逐场格式校验和统计分析；
5. 统一生成补充表格，不把 G26 探索性结果并入 G25 确认性 p 值或主假设。

任一阶段发现实现依赖真值、场景错序、车辆数配置不一致或测试结果参与阈值选择时立即停止并
修复协议，不保留择优结果。本文件不授权任何 Actor 训练。
