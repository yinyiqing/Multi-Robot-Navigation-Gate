# G25 最终闭环消融与 Sealed 评测预注册

状态：`方案冻结，V1-V6 validation 消融、部署成本审计与 G25 sealed 已完成`
登记日期：`2026-08-17`，统计细节修订：`2026-08-18`，sealed 归档：`2026-08-20`

## 1. 目的与边界

本阶段只补齐论文的闭环因果消融、部署成本和一次性 sealed test。主方法继续冻结为
`generalist-5a + avoidance-epoch16 + router-b2`。**禁止训练或更新任何 Actor**；允许为
结构消融从同一冻结数据重新训练 Router，但消融结果不得反向修改 B2 的结构、阈值或 checkpoint。

Dense256 和 G17 已参与模型选择，全部现有结果均定义为 development validation evidence。
它们可以回答“为什么选择当前方法”，不能承担最终确认性结论。只有本协议冻结后的一次性
sealed test 可以形成投稿主结论；该测试已完成并承担当前投稿主结论。

G25 sealed 使用 dense test 原始顺序前 256 个场景、三个 repeat（`20260901/20260902/20260903`）
和七种冻结方法，共 `5376 episodes`。PIRoute 相对 5A 的 full-success 差值为 `+0.1393`
（`0.2461 -> 0.3854`），scene-cluster BCa 95% CI `[+0.1016,+0.1784]`，双侧 sign-flip
`p=0.00001`；collision 差值为 `-0.0924`，paired-success steps 增加 `13.51`。完整统计与
SHA-256 记录见 `local_data/sealed/sealed_statistics.json`。

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

### 3.1 已复用证据与 2026-08-18 队列

V1 `epoch16 always-on` 已在完全相同的 Dense256 validation 协议下完成，因此不重复运行：

- 结果：`full success=0.3281`、`agent success=0.745`、`collision=0.151`、
  `unresolved=0.104`、`timeout=0.406`、`raw mean steps=154.09`；
- seed：`20260810`，结果形状：`(256, 17)`，场景为冻结 manifest 原始前 256 个且顺序一致；
- 结果 SHA-256：`296897f0d22f1bd0cb961a36f8ed47d82ec0dbea6882293ce35238b4e101299b`；
- 原始文件：`12_参数匹配单Actor容量对照/local_data/dense_first256_pilot/results/g12_dense256_epoch16_r1_s20260810.npy`。

当晚只串行补跑 V2 和 V6，共 `512 episodes`。V2 固定使用 G11-D2 已登记的
`2.0/2.2 m + hold 3`，不在 Dense256 上调参；V6 保持 B2 checkpoint 与 stride 2 不变，唯一
变化为 on/off 均设为 `0.43` 且 hold 设为 0。V3 尚无冻结的 TTC 规则实现，V4/V5 需要严格重训
Router，均不在当晚仓促启动；sealed test 在全部前置项和 dry-run 完成前不得读取或运行。

### 3.2 V2/V6 结果

两组均于 `2026-08-18` 第一次尝试完整跑完，无异常重启。结果形状均为 `(256, 17)`，场景
ID及顺序与冻结 manifest 一致，且每组 `1280` 个机器人终止状态完整。

| 方法 | full success | agent success | collision | unresolved | timeout | raw steps | I占比 | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5A（冻结参考） | `0.2695` | `0.6992` | `0.3000` | `0.0008` | `0.0039` | `17.46` | `0` | `0` |
| B2/PIRoute | **`0.4258`** | **`0.7883`** | `0.2102` | `0.0016` | `0.0078` | `34.40` | `0.7002` | `7.54` |
| V2 min-LiDAR | `0.3008` | `0.7227` | **`0.1797`** | `0.0977` | `0.3984` | `154.34` | `0.9995` | `5.04` |
| V6 B2 no-hysteresis/hold | `0.3984` | `0.7797` | `0.2203` | **`0`** | **`0`** | `29.92` | `0.6358` | `8.39` |

V2 相对冻结 5A 为 37 场改善、29 场退化，McNemar exact `p=0.3891`，full success 没有
显著提升。它把静态障碍也当作机器人，导致交互 Actor 占比达到 `0.9995`，结果接近
epoch16 always-on 的高 timeout/长步数行为。因此 B2 的收益不能由简单最小 LiDAR 阈值解释。

V6 相对 5A 为 49 场改善、16 场退化，`p=5.08e-5`，说明去掉滞回和保持后 Router 仍然有效；
但相对完整 B2 为 38 场改善、45 场退化，`p=0.5104`，差异不显著。完整 B2 的 full success
点估计高 `0.0274`，V6 切换次数也更多；据此只能把滞回/保持描述为减少抖动的工程机制，不能
声称它显著提高成功率。

结果 SHA-256：

- V2：`670e5939e9904f0aaab5d55386a4e5b8ed905a4764364a1328c4dbf4d6000388`；
- V6：`d1f795dff2076b5fc1fa08c4c9c66538c6f806387f50a6f702c540309ecb55c4`。

冻结 5A 的论文参考来自 G18 复测（full success `0.2695`）。早期 G12 在相同 manifest、模型
和 seed 下得到 `0.2852`，反映 Gazebo 闭环仍有运行波动；不得把 G12 的 5A 数字与冻结主表
混用，sealed test 的多个 repeat 与 scene-cluster 统计正是为处理这一问题。

### 3.3 V4/V5 实现冻结

V4/V5 继续使用 B2 的 A1 640 场加 student-rollout 640 场聚合训练数据、source+scenario
等权、类别权重、FPR 约束、40 epoch、hidden width 64、seed `20260804`和 checkpoint 选择
规则。V4 唯一结构变化是 GRU 序列长度从 8 改为 1，每次前向只接收当前帧；V5 保持 8 帧
GRU，但从输入中删除两个 Actor 动作及其差分，输入维度从 82 变为 76。二者阈值均由原训练
recipe 独立选择，不复用或扫描 Dense256 阈值。训练入口为：

```bash
bash scripts/run_g25_router_ablations.sh
```

两组训练均已完成。V4 best epoch 7，阈值 `0.43`，checkpoint SHA-256 为
`4189e5976b0de06ead49ba66ff9b11f7cf2ef029a8f39709f8a56cc72d4a858a`；V5 best epoch 1，
阈值 `0.51`，checkpoint SHA-256 为
`95ba582ec2c9a35ed60f17a21936c83166267d3b5ec25b302e6d7bfad3c94012`。离线 validation F1
分别为 `0.8333/0.8365`，只用于实现审计，不替代 Dense256 闭环消融。

### 3.4 V3 TTC/CPA 规则冻结

V3 只在 A1 640 场与 student-rollout 640 场 navigation-train 数据上选择阈值，共
`70,981`帧；脚本不得读取 Dense256 validation 或 sealed test。网格选择首先要求 0-edge
oracle-negative 帧误触发率不超过 `0.10`，再最大化 2 m 交互标签召回，最后最小化总体 FPR。

冻结进入条件为：G0/G1 候选 raw/smoothed shape probability 最大值 `>=0.4`、轨迹年龄`>=2`、
距离`<=2.5 m`、closing speed `>=0.05 m/s`、TTC `<=3.0 s`且CPA distance `<=0.75 m`。
退出采用固定宽松边界：shape `>=0.3`、距离`<=3.0 m`、closing speed `>=0.025 m/s`、
TTC `<=3.5 s`、CPA `<=1.0 m`；hold 3、stride 2。训练集 frame recall 为 `0.2525`，总体
FPR为`0.1049`，0-edge负帧FPR为`0.09835`。选择入口为：

```bash
python3 scripts/select_g25_ttc_rule.py
```

V3 随后在冻结 Dense256 validation 上一次性跑满 `256/256`，无重启。full success 为
`0.2969`，agent success 为`0.7234`，collision 为`0.2758`，timeout 为`0.0039`，
raw mean steps 为`21.60`，interaction share 为`0.1277`。相对5A为23场改善、16场退化，
exact `p=0.3368`，没有显著提升；相对完整B2为19场改善、52场退化，exact
`p=1.12e-4`。因此即使加入G0/G1软候选、接近速度、TTC和CPA，固定手工风险规则仍不能
解释B2的闭环收益。该结论限于本次冻结规则，不外推到所有可能的手工控制器。

V3结果SHA-256为
`3e9f3fd42fdaffacb7ffd6348c3c530dff93634f2354ac7ef88233b251fff074`。

### 3.5 V4/V5 闭环结果

V4/V5 均在第一次尝试完整跑满`256/256`，且场景顺序、终止记账与冻结manifest一致：

| 方法 | full success | agent success | collision | timeout | raw steps | I占比 | switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| B2/PIRoute | **`0.4258`** | **`0.7883`** | `0.2102` | `0.0078` | `34.40` | `0.7002` | `7.54` |
| V4 single-frame | `0.3789` | `0.7719` | `0.2211` | `0.0352` | `45.50` | `0.7053` | `10.00` |
| V5 no-action-difference | `0.3945` | `0.7828` | `0.2164` | `0.0039` | `31.02` | `0.6365` | `7.26` |

两项消融都显著优于5A（V4/V5 exact `p=9.14e-4/3.78e-4`），但相对完整B2的下降均未
显著（V4为34改善/46退化，`p=0.2185`；V5为38改善/46退化，`p=0.4452`）。因此只能说
完整B2点估计最好，且V4出现更多切换、timeout和长轨迹；现有单seed validation不能声称
8帧Router历史或Actor动作差异各自是取得成功率提升的必要组件。

V4/V5结果SHA-256分别为
`141bcd4407cd97e7c822d88dc928b502c836bfcc78035a409e6e34d739fe76ae`和
`2a5f8d837a67330334422b7fc6181c708ed4221f1226fd7da0d04c306e3f328c`。统一复核入口为：

```bash
python3 scripts/analyze_g25_validation_ablations.py
```

统一validation汇总文件SHA-256为
`d867d9f7ff67225489a27e586dd2b7cbc82a931258e0b7ac96836a88a3e0da7b`。

## 4. 部署成本审计

在固定硬件、batch size 1、预热 200 次、计时 2000 次的协议下分别测量：

- 5A 单 Actor 前向延迟；
- 两个 Actor 顺序前向和可用时的并行前向延迟；
- G0/G1 前端、Router 和端到端控制延迟；
- 参数量、理论 MACs/FLOPs、Gate stride、控制频率和峰值显存。

报告中不得用“双 Actor 参数量约等于 R2B”替代运行时成本。PIRoute 每个路由帧需要两个 Actor
候选动作，这是相对单 Actor 的明确部署代价。

“不重新训练 Actor”的成本优势也必须量化。由冻结日志和 artifact 审计以下训练成本，不补跑
Actor，也不对缺失记录作估算：

- 5A、epoch16 和 R2B 的训练 agent samples、environment steps、wall-clock、设备和可恢复的
  GPU/CPU hours；
- G0 的场景数、帧数、候选数、训练 epoch 和 wall-clock；
- A1/B2 的原始与 student-rollout 场景数、Gate 帧数、Router 训练 wall-clock 和设备；
- V4/V5 新增 Router 消融的样本量、wall-clock 和设备。

只有在统一口径的记录可恢复时才比较计算成本；无法恢复的历史 wall-clock 标为 `not recorded`，
不得用理论值替代实测值。论文中的“低成本复用”限定为“不产生新的 Actor 更新和 Actor rollout
训练预算”，不能在审计完成前声称总计算成本必然更低。

CPU部署成本审计已按上述协议完成。单Actor、双Actor顺序前向和完整路由帧的平均延迟分别为
`0.037/0.077/1.443 ms`，完整路由帧p95为`1.490 ms`；对应参数量为单Actor `501,802`、
双Actor `1,003,604`、G0 `89,859`、Router `30,529`。测量使用6个候选的代表帧、单线程CPU，
未实现Actor并行前向，GPU峰值为0。该结果说明相对当前`0.2 s`控制周期的延迟余量充足，但
双Actor确实增加参数、Actor前向次数和G0/G1感知计算，论文不得称为零成本。
部署成本结果文件SHA-256为
`6c7d4ed8657a9355eb1a88974a63f1f424c26dacfb21c2a2ce625a4f01dec0dc`。

训练成本审计也已完成。可恢复的A1/B2/V4/V5 CPU Router训练wall-clock分别为
`42.15/94.27/62.70/85.38 s`；A1使用`28,082`训练帧，B2/V4/V5均使用A1与student
聚合后的`70,981`帧。B1 student rollout另包含640场、42,899帧，其wall-clock未记录。
epoch16可恢复为`320,000` agent samples和2,560场，5A所选历史路径只能确认约190k新增
samples的下界；5A完整warm-start血缘、epoch16与G0的wall-clock均未可靠记录。论文只能声称
本文阶段没有新增Actor更新，不能声称完整系统总训练成本已证明更低。审计结果SHA-256为
`9265e57679d0ea0e2887f43abc1404b0960eca296e07b1cd0be9aeb4f554277b`。

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

sealed test 只运行下列 7 个方法：

```text
V0  5A
V1  epoch16 always-on
V2  min-LiDAR rule
V3  TTC rule
V8  B2/PIRoute
V9  2 m privileged distance rule
V10 R2B-best
```

三个 repeat 均对这 7 个方法使用相同场景和 seed，总预算固定为
`7 x 256 x 3 = 5376 episodes`。V4 single-frame、V5 no-action-difference、V6
no-hysteresis/hold 和 V7 A1 只在 Dense256 validation 做闭环消融，不进入 sealed；它们不属于
唯一确认性假设的必要方法。方法顺序采用循环移位，避免某一方法总在仿真进程的相同位置运行。
失败重启只允许从已落盘的最后完整 scene ID 继续，不得重跑后择优。

方法基础顺序冻结为`5A / epoch16 always-on / min-LiDAR / TTC / B2 / 2m privileged / R2B`；
三个repeat分别循环左移`0/1/2`位。dry-run固定只读取Dense validation前2场，seed为
`20260818`，运行同一七方法入口；只有七组结果均通过形状、场景顺序和终止记账检查后，才允许
`prepare_g25_sealed_manifest.py`首次读取dense test并按原始顺序截取前256场。

该dry-run已于`2026-08-18`完成。七种方法均在第一次实际仿真尝试中跑满`2/2`，结果形状、
场景顺序和每组10个机器人终止状态全部通过；期间没有读取dense test。completion记录
SHA-256为`db757448457e3785f6bc8e4adafc40681ba64d620c4fd65cd33667c5b10c895f`。

dry-run提交后已首次读取冻结dense test并生成正式manifest，只取原始顺序前256场，不做任何
策略相关筛选。原始dense test SHA-256为
`03a744048102d7310db026e399e41c4ce664ed31b180438b2a2b519c78133eab`，正式前256场manifest
SHA-256为`1098d13d09af2a4850c2d227eec8fe87f05a7de3b3a8d830de2dd472a059e211`。

### 5.2 假设与指标

唯一主要假设：在 dense sealed test 上，B2/PIRoute 的 episode-level full success 高于 5A。
主要效应量是二者 full-success 差值。collision、timeout、steps、interaction share 和切换次数
均为预注册次要指标，必须与成功率同表报告，不得因方向不利而省略。

统计单位是 scene。三个 repeat 不能当作 768 个相互独立场景：

1. 报告每个 repeat 的点估计和改善/退化/持平数；pooled McNemar 只作参考；
2. 对每个 scene 先计算三个 repeat 的 PIRoute-5A 配对差值均值，再以 scene 为 cluster；
3. 主效应 95% CI 使用 scene-cluster BCa paired bootstrap，重采样 `20,000` 次，随机 seed
   固定为 `20260818`；每次抽取 256 个 scene cluster，并携带各自全部 repeat；bias correction
   由bootstrap分布计算，acceleration使用leave-one-scene-cluster-out jackknife；
4. sign-flip 使用上述 scene-level 配对差值，进行 `100,000` 次 Monte Carlo 随机符号翻转，
   检验统计量`T`为256个scene差值的均值；双侧检验，
   `p=(1 + count(|T*| >= |T_obs|))/(100000 + 1)`，随机 seed `20260818`；
5. 唯一主要检验的显著性水平为双侧 `alpha=0.05`，不做多重校正。只有 BCa 95% CI 下界
   大于 0 且 sign-flip `p<0.05` 时，才称 sealed test 确认核心 full-success 差异；
6. 若点估计为正但 CI 跨 0 或 sign-flip 未达阈值，只写正向趋势；若点估计不为正，则写未复现；
7. collision、timeout、steps、interaction share 使用相同 scene-cluster BCa bootstrap。
   它们是次要/描述性指标；若额外报告 p 值，统一用 Holm 方法校正并明确探索性；
8. 按 0/1/2/3+ 冲突边分层只作预注册异质性分析，不分别宣称新的主要显著性结论。

### 5.3 步数与提前终止

碰撞会提前结束 episode，因此“所有 episode 原始平均步数更少”可能反而代表更差的安全性。
步数固定报告三种口径：

1. `raw termination steps`：所有 episode 到终止的原始步数，只描述仿真占用时间，不称导航效率；
2. `paired-success steps`：仅在 PIRoute 与比较方法对同一 scene/repeat 都 full success 时计算
   配对步数差，作为成功条件下的导航效率；同时报告有效配对数，不跨方法比较不同成功子集，
   并明确该口径受“双方均成功”的条件选择影响，只作描述性结果；
3. `penalized completion steps`：full success 使用实际步数，其他结局统一赋值为该协议配置中的
   最大 horizon `H_max`，用于联合表示完成率和耗时。

三种口径均按 scene 聚类给出 95% CI。不得只选取对方法有利的一种步数定义。

V1-V7、V9-V10 是机制/边界对照。B2 对 A1、single-frame 或 no-action-difference 的 validation
比较即使显著，也不改变主方法或主要假设；B2 对 student-rollout 的现有非显著结论必须如实保留。

## 6. 图表与停止规则

最终输出固定包括：

- success-collision-steps Pareto 图，validation 与 sealed 分面显示；
- full success、collision、timeout、三种 steps、interaction share 的配对差值及 95% CI；
- 至少一个成功路由、一个晚退出和一个误触发案例的时间轴；
- 参数量、前向次数、FLOPs、延迟和控制频率表；
- Actor、G0 和 Router 的训练样本量、wall-clock 与可恢复计算成本表。

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
2. 审计现有 Dense256 的 V0/V7/V8/V9/V10 逐场结果；协议与哈希完全相同则复用，不重复运行；
3. 只补跑缺失的 V1-V6 Dense256 validation，并完成部署/训练成本审计；
4. 冻结脚本、容器/环境、manifest、顺序与统计输出，完成一次 dry-run；
5. 一次性运行 7 个方法、三个 seed 的 dense sealed test；
6. 只运行预注册统计脚本并生成最终表图。

本文件不授权任何 Actor 训练。
