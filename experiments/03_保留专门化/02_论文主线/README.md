# ICRA论文主线：普通导航Actor、条件避障Actor与在线Gate

状态：`legacy route frozen / all E2 interaction-Actor pilots rejected`。
更新时间：`2026-08-13`。

本文件是研究方法、数据边界和实验准入的唯一协议。项目快速状态见
[PROJECT_STATUS](../../../PROJECT_STATUS.md)，历史实验状态见
[实验注册表](../../EXPERIMENT_REGISTRY.md)。

## 2026-08-13避障Actor续训授权

主线恢复为`5A + 避障Actor + 可部署Gate`。epoch-16是原避障Actor，epoch-17是当前冻结
避障Actor；数字只用于标识artifact。原避障Actor在16轮预算边界达到明显新高，
因此授权从其完整320k checkpoint原样恢复，只追加epoch 17-20共80k。原checkpoint不覆盖，
新分支训练内部胜出后仍须做独立matched复测才能替换。协议见
[避障Actor续训](16_避障Actor续训/README.md)。

该续训已于`2026-08-14`完成并在epoch 20停止。epoch 17的internal validation full
success为`0.7429`，高于原epoch 16的`0.7071`，但timeout为`0.0357 vs 0.0143`；
epoch 18-20没有形成更高成功率，并呈现避障占比、timeout和步数增加的过度保守趋势。
独立matched admission随后完成`480/480`场。epoch 17相对epoch 16的full success为
`0.6875 vs 0.6333`、collision为`0.0967 vs 0.1075`，但平均步数为`37.46 vs 32.32`
（`1.159x`），超过预注册的`1.10x`效率上限；逐场exact `p=0.1421`，multi-edge收益也
只有`+0.0125`。因此epoch 17没有通过原预注册的`1.10x`效率硬门槛。

在读取sealed test前，项目于`2026-08-14`显式修订模型选择目标：full success作为主指标，
collision与timeout作为安全约束，平均步数改为必须报告的效率代价。epoch 17在internal和
独立matched validation上full success方向一致，且matched collision下降、timeout不变，
因此冻结为当前避障Actor；epoch 16保留为消融对照。论文和实验记录必须披露该规则修订、
`p=0.1421`及`1.159x`步数代价，不能声称epoch 17通过了原效率准入。

G11-F随后完成epoch-17动作特征下的A1离线重建和`640/640`场student rollout全量审计。
当前只聚合原5A轨迹与新epoch-17 student轨迹训练Gate；旧epoch-16 B1 shard和B2
checkpoint只作历史证据，不进入最终方法。

G11-F-C的`300/300`场闭环pilot中，5A/F-A1/F-B2的full success为
`0.640/0.710/0.680`。F-B2碰撞略低但timeout和平均步数明显更高，且相对F-A1逐场为
9改善、12退化，因此当前冻结F-A1，F-B2作为DAgger消融。

同场R2-10k补充pilot随后完成。R2-10k的full success为`0.760`，高于F-A1的`0.710`；
collision为`0.060 vs 0.088`，平均步数为`28.44 vs 33.41`，但timeout为`0.030 vs 0`。
F-A1相对R2-10k为9场改善、14场退化，exact `p=0.4049`。当前小样本差异未显著，但
差异未显著。但R2-10k经过了从随机初始化开始的`n1 -> n2 -> n3 -> n5`额外课程，
与5A/双Actor的训练流程和预算不匹配。因此这组数据只作cross-protocol诊断并在
补充材料披露，不进入论文公平容量baseline排名，也不用于决定Gate取舍。

## 2026-08-12候选支线授权

旧`5A + epoch-16`继续冻结为fallback和论文基线，不覆盖其artifact。为消除最终方法中
“普通Actor来自E2、避障Actor却基于5A训练”的模型血缘不一致，当前显式授权一个最小
`E2 + I-E2`候选pilot：

1. 在N5冻结validation上补跑seed `20260818`的E2-only matched control；
2. 从E2 Actor warm start新的`I-E2`，窗口外由冻结E2执行，2米交互窗口内由I-E2执行；
3. I-E2训练侧使用87维邻域Critic、动态距离加权reward和interaction-only更新，部署
   Actor仍为24维；前21k agent samples冻结Actor；
4. pilot预算固定`2 x 20k`，不得自动延长到旧epoch-16的320k；
5. 训练后在同seed、同120场manifest上运行`E2 + I-E2 recovery-oracle`。

该pilot只是候选方法准入，不表示E2或I-E2已替换当前冻结组件。完整协议与日志入口见
[E2恢复Actor诊断与训练](15_E2恢复Actor诊断与训练/README.md)。

该pilot现已完成。matched E2、E2+旧epoch-16 recovery、E2+I-E2 recovery的full
success分别为`0.6583/0.7750/0.7333`。I-E2相对E2有正收益，但配对检验`p=0.1360`，
且multi-edge full success为`0.425`，低于旧epoch-16的`0.600`。因此I-E2尚未替换冻结
interaction Actor；当前只授权同场轨迹和动作诊断，不授权直接延长训练或启动新Gate。

### I-E2-M多冲突修订pilot（已拒绝）

为回应上一版只覆盖单冲突边、在multi-edge上停滞的问题，已书面修订Actor I的训练分布，
但不改变论文主方法的两个Actor加在线Gate结构。I-E2-M从冻结E2 Actor warm start，使用
只来自navigation-train且与冻结非train视图互斥的2400场训练清单：edge-1/edge-2/edge-3+
分别为`960/720/720`；内部validation为`140/30/30`。清单哈希和逐项排除记录见
`datasets/fixed_v1/views/ie2_multi_conflict_v1/`。

相对首版，I-E2-M保留24维部署Actor、87维训练侧邻域Critic、interaction-only更新和E2
窗口外执行；修正 `average` 模式未实际调用 interaction stagnation reward 的配置问题，
改为 `average_plus_interaction`，并加入弱 safe-recovery奖励。预算固定为2x20k，Actor
在21k后解冻。完成后只在冻结N5 120场上做matched复测，不读取sealed test，不启动Gate；
该修订取消single-to-multi零样本泛化主张。

I-E2-M现已完成但未通过。matched E2、旧epoch-16 recovery、I-E2-M recovery的full
success为`0.7417/0.7750/0.6833`，multi-edge为`0.450/0.600/0.400`。诊断确认训练确实
执行了edge-1/edge-2/edge-3+，但Actor梯度只覆盖1m内仍在闭合的risk状态，没有覆盖停滞
恢复与release；40k也只经历了422/2400个训练场景。当前拒绝I-E2-M且不追加预算，完整
证据见[I-E2-M诊断](15_E2恢复Actor诊断与训练/I_E2_M_DIAGNOSIS.md)。

### I-E2-F4四阶段pilot（已拒绝）

F4关闭了closing-risk二次筛选，让整个2米交互窗口参与Actor更新，并启用近车恢复奖励；
训练分布继续覆盖multi-edge。Actor冻结的epoch 1与解冻后的epoch 2在同一200场internal
validation上的full success为`0.630/0.605`，collision为`0.094/0.099`，timeout为
`0.090/0.100`，平均步数为`54.64/66.10`。更新后所有主要指标均退化，因此拒绝F4，
不追加训练，不用于Gate。训练器的`best`来自冻结阶段，不是训练成功的避障Actor。

训练前的同seed matched控制中，E2与`E2 + old epoch-16 recovery`的full success为
`0.6917/0.7750`，15场改善、5场退化，exact `p=0.04139`。完整记录见
[I-E2-F4结果](15_E2恢复Actor诊断与训练/I_E2_F4_RESULTS.md)。

## 1. 当前方法

```text
Actor N：冻结5A，负责普通导航、目标推进和静态障碍
Actor I：冻结epoch-17，负责短时局部机器人冲突
Gate：运行时逐机器人、逐时刻选择Actor，并负责进入与退出避障状态
```

执行策略为：

```text
a_t = pi_I(o_t),  if g(h_t) = 1
      pi_N(o_t),  otherwise
```

`o_t`是本车24维Actor观测；`h_t`是Gate可使用的本机激光、导航状态和短时历史。
部署时禁止使用其他机器人真值、通信、场景名称、冲突图或standard/dense标签。

导师已于`2026-08-04`认可“普通导航Actor + 条件避障Actor”的角色划分。避障Actor
不承担全程导航效率，但必须保持目标一致性、解除冲突并允许Gate交还控制。

导师确认的交付是训练一个可部署在线Gate，没有把“single-edge训练到multi-edge零样本
泛化”设为方法要求。当前先用0-edge和single-edge训练，是为了建立干净、可诊断的第一
候选；multi-edge是冻结后的评测维度。自然泛化若成立可作为附加贡献，若不成立则允许
在读取sealed test前书面修订训练集并重训同一个Gate，方法主线仍不改变。

## 2. 为什么需要Gate

冲突强度会在同一条轨迹中出现和消失，因此episode开始时固定选择Actor只能形成场景
分类器，不能完成所需的在线分工。

Gate成立的必要条件是两个Actor存在不同优势区。如果Actor I在无冲突、单冲突和多冲突
上同时以不更差的碰撞、成功、timeout和效率全面支配Actor N，则Gate应被删除并退化为
单Actor。现有证据相反：

- 5A普通推进快，但在机器人冲突中碰撞较多；
- epoch-16降低碰撞，但全程运行会大量保守等待和timeout；
- 在局部冲突窗口调用epoch-16，组合结果明显高于两者全程独立运行。

不得为了证明Gate有用而故意破坏Actor I的导航能力；互补性必须由固定评测得到。

## 3. Actor训练契约

当前方法中的5A和epoch-17 artifact继续冻结，不得覆盖。epoch-16作为消融对照保留。
当前不再授权Actor训练；以下条目记录已经结束的书面例外：

- [避障Actor续训](16_避障Actor续训/README.md)从epoch-16完整checkpoint分叉，只追加
  epoch 17-20；新分支通过独立matched复测前不得替换原避障Actor；
- [G12参数匹配单Actor容量对照](12_参数匹配单Actor容量对照/README.md)训练一个不进入
  主方法的加宽单Actor baseline，用于排除双Actor收益只是参数量翻倍。

以上授权不允许更新或覆盖5A、原epoch-16，也不恢复其他历史Actor路线。

`2026-08-14`新增一个不进入主方法的容量baseline授权：G12-R2B按历史5A流程训练参数匹配
加宽Actor。它从3D2 Actor函数保持扩宽，使用五车standard、individual reward、fresh
24维Critic、20k Actor冻结和30k总预算；完整协议见
[R2B 5A流程对照](12_参数匹配单Actor容量对照/R2B_5A_RECIPE_PROTOCOL.md)。该实验不得
以训练出更差结果为目标，也不得读取G11-F-C或sealed test选择checkpoint。R2B已完成：
Actor解冻后从`0.508` full success降至`0.042`，训练明显退化。按5A相同的validation
模型选择逻辑，自动best一直停在epoch 1；因此冻结R2B-best作为流程/预算匹配的
训练流程输出baseline。论文必须同时报告：best未实际使用新增容量，后续Actor更新坍塌；
不得表述为容量不足。

R2B-best的G11-F-C同场`100`场补评已完成。5A/R2B-best/F-A1的full success为
`0.640/0.680/0.710`，collision为`0.104/0.106/0.088`，timeout均为`0`，平均步数
为`21.91/19.41/33.41`。点估计满足`5A < R2B-best < F-A1`，但两个相邻差异的
McNemar exact检验均未显著（`p=0.4545/0.6636`），因此只能作pilot趋势而非最终结论。

### Actor N：generalist-5a

- 五车共享Actor；
- 负责目标推进、墙和箱子避障以及普通状态；
- 当前冻结，不再微调。

### Actor I：avoidance-epoch17

- 从5A初始化，并从原epoch-16完整训练状态续训一轮；
- 在强交互五车训练集中采用逐机器人、逐时刻oracle分工；
- 最近活动机器人真值中心距离`<=2.0 m`时，由Actor I执行并进入其更新范围；
- 其他状态始终由冻结5A执行；
- 真值距离只用于训练分工，没有进入Actor的24维部署输入；
- 原epoch 16训练`320,000` agent samples并覆盖`2560/2560`场；当前epoch 17从该完整
  状态继续训练一轮`20,000` samples，epoch 18-20因退化拒绝。

完整路径复审发现原2560场中有11场实际为edge-2，占`0.43%`。当前默认论文主张是
“两个冻结Actor由一个可部署Gate在线分工”，不主张整个系统严格只见single-edge，
因此不为零样本拓扑主张重训Actor。只有后续明确升级该附加主张时才需要：

1. 用corrected full-horizon edge-1重新训练同协议Actor I；或
2. 明确收窄论文主张，不把现有模型称为严格零样本冲突拓扑泛化。

无论是否升级该附加主张，所有训练数据和表述都必须在读取sealed test前冻结。

## 4. Gate定义

### 当前oracle

当前有效组合逐机器人、逐时刻计算最近活动机器人的真值距离：

```text
d_nearest <= 2.0 m -> epoch-17
d_nearest >  2.0 m -> 5A
```

这是不可部署上界，不是学习Gate成绩。机器人实际只有本机激光；现有20维扇区最小值
会把机器人、墙和箱子压成同一种障碍。

### 可部署Gate

Gate必须使用：

- 本机激光点或由其得到的连续形状证据；
- 本机里程计和目标状态；
- 自运动补偿后的候选跟踪、闭合速度、CPA/TTC等短时特征；
- 滞回和最短保持时间，避免频繁抖动。

Gate不得把“2米内有障碍物”直接当成机器人标签。`2.0 m` oracle可以用于监督初始化、
诊断和上界；最终选择标准应接近“短期调用Actor I是否比继续Actor N更有利”。

## 5. 已确认结果

### 匹配单冲突validation，140场

| 方法 | agent success | collision | full success | timeout | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5A全程 | `0.8186` | `0.1814` | `0.4214` | `0/140` | `35.2` |
| 5A + epoch-16真值oracle | `0.9157` | `0.0800` | `0.7000` | `2/140` | `54.3` |

这是validation，不是test。组合改善主要位于deep和close层，同时存在更保守、更慢的代价。

### dense validation，1000场

| 方法 | agent success | collision | unresolved | full success | timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5A全程 | `0.7064` | `0.2930` | `0.0006` | `0.3090` | `0.0030` |
| 5A + epoch-16真值oracle | `0.8476` | `0.1456` | `0.0068` | `0.5450` | `0.0180` |

epoch-16全程运行在前256场的full success为`0.2305`、timeout为`0.5156`，因此它是
条件避障Actor，不是独立Dense Actor。

### 历史可部署Gate

第一版G2-A Gate在独立exact-edge-2的200场确认上：

- 5A full success：`0.325`；
- learned Gate：`0.405`；
- McNemar exact：`p=0.06812`；
- oracle增益恢复：`30.2%`。

方向为正但未通过统计和收益恢复门槛，只能作为后续Gate基线。

## 6. 数据边界

### Gate训练可见

- `fixed_v1`导航train内部划分的数据；
- 0-edge和corrected full-horizon single-edge；
- 仿真真值只能生成监督标签，不进入推理输入。

以上是当前G11-A1/B第一候选的冻结数据协议，不是“训练一个Gate”方法本身必须永远只见
single-edge。若multi-edge评测不足，允许在sealed test前登记新实验ID并加入
navigation-train的multi-edge数据；修改后必须重做模型选择和全部准入，且不得再声称
single-to-multi零样本泛化。

### 模型选择

- train和validation的scenario ID必须互斥；
- 允许用固定小validation选择Gate阈值、滞回和保持时间；
- 更大validation用于准入，不能反复调参。

### 最终评测

在模型、特征和阈值冻结后按顺序评估：

1. 0-edge能力保持；
2. single-edge局部收益；
3. exact-edge-2及更复杂拓扑边界；
4. dense/standard sealed test。

同一张方法对比表中的5A、epoch-16 always-on、规则Gate、A1、B2、最终Gate、参数匹配
单Actor和oracle必须重新运行同一个冻结manifest。所有方法必须共享完全相同的scenario
ID与顺序、评测seed列表、重复次数、物理参数、最大步数和终止条件。结果逐场记录
`scenario_id`和evaluation repeat；分析器必须先校验manifest SHA-256以及ID缺失、重复、
错序和repeat不一致，任何一项失败都应终止分析。历史上不同集合或不同运行协议的汇总
成功率只能作背景证据，不能拼入最终方法表。

若Gate或Actor I使用multi-edge训练数据，则不得声称single-to-multi零样本泛化。

## 7. Gate准入

可部署Gate至少同时满足：

1. 0-edge full success相对5A下降不超过`3`个百分点；
2. single-edge full success显著超过5A，并且改善多于退化；
3. timeout不出现系统性增加；
4. 明确超过不区分机器人与静态障碍的min-laser规则Gate；
5. 最终Gate在multi-edge上仍保持正向收益；第一候选若未达到，可加入train内部
   multi-edge重训同一个Gate，但必须取消零样本泛化表述；
6. 报告相对同场oracle的收益恢复比例，不能只报告绝对最好值。

主指标为`full_success_rate`，同时报告`agent_success_rate`、`collision_rate`、
`unresolved_rate`、`timeout_episode_rate`、平均步数、Gate激活比例和切换次数。
结局必须满足：

```text
success + collision + unresolved = agents * episodes
```

## 8. 必须对照

| 对照 | 用途 |
| --- | --- |
| 5A always-on | 普通导航基线 |
| epoch-16 always-on | 条件Actor全程使用的失败边界 |
| min-laser或距离规则Gate | 不区分机器人语义的可部署下界 |
| 历史G2-A Gate | 已有学习Gate基线 |
| 当前learned Gate | 最终方法 |
| `2.0 m`真值oracle | 不可部署上界 |
| corrected edge-1完整Actor pilot | “为什么不用单一完整Actor”的失败对照 |
| G12参数匹配加宽单Actor | 控制两个Actor checkpoint带来的参数容量增加 |

必要消融包括无时序、无机器人形状证据、无滞回和不同最短保持时间。

## 9. 已关闭路线

- standard/dense两个完整场景专家；
- 独立Dense完整Actor及继续扫描reward/Critic；
- epoch-16无边界整网续训；当前只允许已登记的epoch 17-20固定续训；
- 24维单帧Residual融合；
- controlled-ego、pair Actor及复杂Critic支线；
- corrected edge-1完整Actor继续训练。

失败证据保留，但历史目录中的旧命令不构成执行授权。完整分类见
[实验注册表](../../EXPERIMENT_REGISTRY.md)。

## 10. 当前执行顺序

1. 固定本文档、模型哈希和数据边界。
2. 当前不作整个系统严格single-edge训练的主张，不为该附加主张重训Actor I；若后续
   升级主张，必须在sealed test前重新冻结协议。
3. [可部署在线Gate研究](11_可部署在线Gate研究/README.md)的G11-A1时序蒸馏已通过，
   G11-B1的640场student rollout和G11-B2主seed聚合训练已完成；G11-C固定50场、两个
   仿真重复也已完成。B2的full success为`0.78/0.76`，两次均高于A1的`0.70/0.66`，
   因此保留student聚合路线；该结论只通过pilot停止条件，不是最终准入。
4. G11-D2已经完成。B2相对5A的导航收益显著，但因平均步数为`2.058x`、interaction
   占比`0.7756`而未通过效率准入；B2相对A1的配对检验为`p=1.0`。
5. G12-P1在40k处因Actor动作饱和和full success坍塌早停，只作训练稳定性诊断，不作
   参数容量结论。后续按[G12-R公平参数匹配路线](12_参数匹配单Actor容量对照/REVISED_PLAN.md)
   执行原宽度控制、从头课程训练和受约束联合训练。
6. G12-R1原宽度控制已完成：full success从`0.717`降至`0.283`，说明无约束TD3/fresh
   Critic是P1与R1的共同不稳定来源，参数扩宽不是P1坍塌的充分解释。R1只作诊断，不作
   R2 warm start或论文主性能模型。
7. G12正式大Actor不沿用Gate的0-edge/edge-1训练边界。R2-S0已通过；S1固定困难case诊断
   已完成`126/126`场，full success为`72/126`。首段8-case repair-only更新虽正常完成20k，
   但broad n1降至`69/120=0.575`且出现49次timeout，候选按协议拒绝，不进入targeted复测
   或S2。S2从S0 best完成两车完整broad首段，10k/20k full success为`0.9333/0.9250`，
   冻结10k best。S3三车首段的10k/20k full success为`0.8750/0.8833`，冻结20k best；
   S4五车首段的10k/20k full success为`0.6667/0.6917`。20k配对结果显著超过5A，但因
   `3/120` timeout相对5A增加`0.025`而未通过；预先登记的唯一10k fallback随后以
   `0.7000 vs 0.5583` full success、`0` timeout通过全部五项准入。冻结10k作为R2参考，
   R3已经按固定四槽训练调度完成40k pilot，但Actor解冻后full success从
   `0.667`降至`0.575`，因此已停止且不启动R4。具体协议见
   [G12-R2协议](12_参数匹配单Actor容量对照/R2_PROTOCOL.md)和
   [G12完整场景协议](12_参数匹配单Actor容量对照/FULL_SCENE_PROTOCOL.md)及
   [G12-R3协议](12_参数匹配单Actor容量对照/R3_PROTOCOL.md)。
8. I-E2、I-E2-M和I-E2-F4均已拒绝。旧epoch-16继续作为fallback；不追加这些E2配套
   Actor的训练，不扫描reward权重，也不启动基于F4的新Gate。当前唯一避障Actor训练是
   从原epoch-16完整状态固定续训epoch 17-20；F4不得作为其warm start或配置来源。
9. 完成主对照、消融和multi-edge边界评估。G11-E的50场exact-edge-2 pilot与后150场
   confirmation已经完成清单冻结和互斥审计，不得并行启动第二套Gazebo。
10. Gate、参数匹配单Actor和所有阈值冻结后一次性读取sealed test。

任何新长跑必须先在本文件登记实验ID、数据split、模型哈希、seed、准入和停止条件。
