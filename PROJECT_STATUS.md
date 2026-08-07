# 当前项目状态

更新时间：`2026-08-07`。

本文件是进入项目后的第一阅读入口。方法定义、实验准入和数据边界以
[论文主线协议](experiments/03_保留专门化/02_论文主线/README.md)为准；历史 README
中的“当前”和“下一步”只代表当时判断。

## 投稿时间

- ICRA 2027论文截止时间为`2026-09-15 23:59 PST`，以
  [官方Call for Papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
  为准；从`2026-08-06`起约有40天。
- 当前“两周”目标是冻结方法和主要长跑，不是把全部40天都用于试验。之后必须保留统一
  复测、统计、作图、视频和论文写作时间。

## 一句话方法

在无通信、局部观测的五车导航中，冻结普通导航 Actor 和条件避障 Actor，由运行时
Gate 根据本机传感器在两者之间逐机器人、逐时刻切换。

```text
Actor N = generalist-5a
  普通推进、目标导航、墙和箱子避障

Actor I = interaction-epoch16
  局部机器人冲突中的减速、避让和脱困

Gate
  仅用本机可部署观测决定 N -> I 和 I -> N
```

导师已于`2026-08-04`认可“普通导航 Actor + 条件避障 Actor”的角色划分。避障 Actor
不是完整 Dense Actor，也不要求全程独立导航；它仍需保持最低限度的目标趋势和退出
冲突能力。

导师确认的当前任务是训练一个在线Gate，没有要求把“single-edge训练到multi-edge零样本
泛化”作为方法前提或论文硬主张。当前0-edge/single-edge训练协议用于先得到干净的Gate
候选；multi-edge先作为冻结后的评测维度和潜在附加贡献，不得反过来带偏主线。

## 当前冻结组件

| ID | artifact | SHA-256 | 状态 |
| --- | --- | --- | --- |
| `generalist-5a` | `TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth` | `fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5` | 普通导航 Actor |
| `interaction-epoch16` | `interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth` | `6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b` | 条件避障 Actor |

两者的部署 Actor 输入均为本车24维观测。`interaction-epoch16`训练时使用了仿真真值
进行状态分工，但真值距离没有进入 Actor 输入。

## 已确认事实

1. 在140场单冲突 validation 上，5A全程运行的full success为`0.4214`；按`2.0 m`
   真值距离局部调用epoch-16后为`0.7000`。
2. 在dense validation 1000场上，5A为`0.3090`；真值组合为`0.5450`。
3. epoch-16全程运行的前256场中，full success为`0.2305`，timeout为`0.5156`。
   它降低碰撞但不适合承担普通导航。
4. `2.0 m`切换使用其他机器人真值位置，只是不可部署oracle上界，不是已训练Gate。
5. 历史可部署Gate在独立exact-edge-2的200场上从`0.325`提高到`0.405`，但
   `p=0.06812`且只恢复oracle收益的`30.2%`，未通过最终准入。
6. corrected edge-1完整Actor pilot在Actor解冻后，50场monitor的full success从
   `0.42-0.44`降至`0.16/0.10`，已经关闭，只作单Actor失败对照。
7. G11-A0离线诊断中，单帧加入两个Actor动作没有增益；8帧GRU在360度标签下相对
   静态Gate的F1与AP连续5个seed提高，区间IoU提高且切换次数下降。该结果只授权采集
   当前协议的A1数据，不是闭环导航成绩。
8. G11-A1在导航train内部互斥的640/120场上完成正式离线pilot。预注册主seed及4个
   复核seed全部通过：T1相对S0的F1平均提高`0.01470`、区间IoU提高`0.02207`，切换
   次数平均减少`240`；该结果授权student rollout，不是闭环full success。
9. G11-B1已完成navigation-train的`640/640`场student rollout并通过全量审计，共
   `42,899`个Gate帧，数据集SHA-256为
   `bda1a3ebe16eb481da8629b21f8f030fe9f0a6499da6409c90b0c2e936614fba`。采集轨迹的
   full success为`0.750`，只作训练集运行诊断，不是validation或方法成绩。
10. G11-B2主seed已完成来源与场景等权聚合训练，checkpoint SHA-256为
    `fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52`。它满足冻结S0
    FPR上限，但相对A1主seed在同一内部validation上的F1为`-0.00928`、AP为`-0.00541`、
    区间IoU为`-0.01405`；只授权固定50场闭环pilot，不构成聚合有效的结论。
11. G11-C固定50场、两个仿真重复的闭环pilot已经完成，共`300/300`个episode。合并
    结果中5A/A1/B2的full success分别为`0.64/0.68/0.77`，collision分别为
    `0.110/0.076/0.060`；B2在两个重复中均高于A1，因此student-rollout聚合通过
    “是否保留”的pilot判断。B2仍有`0.789`的interaction Actor占比、`51.25`平均步数
    和`0.03` timeout，且相对A1的小样本配对检验未显著，不能写成最终Gate准入通过。
12. G11-D2独立validation的200场七策略评测已完成。5A/A1/B2的full success分别为
    `0.650/0.740/0.745`；B2相对5A配对McNemar exact `p=0.01266`，single-edge子集
    `p=0.00599`。B2通过导航准入，但平均步数为5A的`2.058`倍、interaction占比为
    `0.7756`，未通过预注册效率准入；B2相对A1为22场改善、21场退化，`p=1.0`。
13. G12-P1参数匹配加宽单Actor在函数保持初始化后仍有`0.717` full success，但Actor
    解冻20k samples后降至`0.050`并触发早停。动作漂移到近似固定`[0.992,-0.913]`，
    因此该运行只证明无约束fresh-Critic TD3发生训练坍塌，不能证明大Actor容量不足。
14. G12-R1原宽度控制已完成：同一validation的full success从Actor解冻边界的`0.717`
    降到更新20k后的`0.283`，agent success从`0.915`降到`0.735`。这证明P1坍塌的共同
    根因是fresh-Critic无约束TD3更新，不是参数翻倍；P1更严重只说明扩宽可能放大不稳定。
15. G12正式大Actor路线已纠正：Gate的0-edge/edge-1清单只用于P1/R1诊断；R2从头复现
    完整单车到五车课程，R3/R4在完整standard/dense train上全程控制五车，并对强交互
    子集做预注册重采样。正式大Actor不会排除train中的multi-edge场景。
16. G12-R2-S0已按100k预算完成。五次单车broad validation的full success为
    `0.983/1.000/1.000/0.992/1.000`，epoch 2为冻结best；全程没有P1/R1式坍塌。S0通过
    基础导航准入，但不构成两车、五车或冲突能力结论。
17. G12-R2-S1固定困难case诊断已完成`126/126`场，总full success为`72/126=0.5714`；
    42个stage-case项中`22 pass / 0 borderline / 20 repair`。普通近目标与隔墙导航通过，
    缺口集中在近障恢复、贴墙姿态、离墙推进和反向脱离。该结果授权登记repair-only S1
    补课，但不授权直接跑满80k，也不是论文性能指标。
18. G12-R2-S1首段20k定向补课已完成，但broad n1仅`69/120=0.575` full success，
    collision为`2/120`、timeout为`49/120`，未通过预注册回归门槛。候选已拒绝且不得进入
    targeted复测或S2；结果表明8-case全网络无anchor更新造成严重窄分布遗忘，不否定S0
    已建立的加宽Actor单车导航能力。
19. G12-R2-S2两车完整broad首段已完成。10k/20k固定120场validation的full success分别为
    `0.9333/0.9250`，collision为`0.0583/0.0667`，均无timeout；冻结10k为best并通过进入
    三车课程的准入。n2 validation只有`17/120`个派生冲突场景，且无逐场评测记录，因此
    该结果只证明两车完整导航稳定，不能声称冲突能力提升或相对S0已有训练增益。
20. G12-R2-S3三车完整broad首段已完成。10k/20k full success为`0.8750/0.8833`，agent
    success为`0.9417/0.9472`，collision均为`0.0528`；20k消除了10k的unresolved和timeout，
    平均步数从`22.05`降至`14.00`，因此冻结20k best并进入五车课程。
21. G12-R2-S4五车完整broad首段已完成。10k/20k full success为`0.6667/0.6917`，agent
    success为`0.8867/0.8917`，collision为`0.1117/0.1000`；20k timeout升至`0.0417`且
    平均步数升至`35.45`，但未触发回滚线，因此冻结20k best。进入R3前先做同场5A配对
    分层准入，不能从训练汇总直接声称大Actor已通过普通能力保持。
22. G12-R2五车20k配对准入中，R2相对5A的full success为`0.6917 vs 0.5583`，多冲突为
    `0.425 vs 0.150`，总体18场改善、2场退化，`p=0.000402`；但timeout为`3/120`，相对
    5A增加`0.025`并超过`0.020`上限，故严格判定未通过。当前只登记预先保存的10k作为
    唯一fallback；若仍失败则R2停止，不再扫描checkpoint或seed。

以上都是validation或diagnostic，不是sealed test结果。不同数据集上的数值不得直接
横向比较。

## 当前工作

当前主线仍是可部署Gate；参数匹配单Actor只作论文公平对照：

1. 冻结5A和epoch-16，不再更新两个Actor。
2. Gate使用本机激光雷达、导航状态及必要的短时历史，不能读取其他机器人里程计、
   场景类别或冲突图。
3. `2.0 m` oracle用于监督、诊断和上界；最终Gate需要判断何时调用避障Actor更有利，
   不能把“附近有障碍物”直接等同于“附近有机器人”。
4. G11-D2已经归档：B2导航收益成立，但效率准入失败；A1与B2尚无显著差异，不能把
   student-rollout聚合写成已证明的闭环增益。
5. 下一项Gate实验是multi-edge边界评估。G11-E已经冻结50场
   exact-edge-2 pilot与后150场confirmation，二者与当前Gate训练、G11-C和G11-D2均
   无场景重叠。若自然泛化不足，可以在读取sealed test前
   书面修订协议、加入navigation-train的multi-edge数据并重训同一个Gate；此时不再
   声称single-to-multi零样本泛化。
6. G12-P1已停止并降级为训练稳定性诊断。后续容量对照遵循
   [G12-R公平路线](experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/REVISED_PLAN.md)：
   先做原宽度控制，再让参数匹配Actor从头完成普通导航课程和完整五车分布联合训练。
   任何新长跑必须先冻结具体超参数和manifest，不得用D2或sealed test调参。
7. G12-R1已归档到`logs/archive/diagnostic/g12_r1/`，只作机制诊断，其checkpoint不得
   作为R2 warm start或论文主性能模型。
8. G12完整场景内部validation已经冻结为`g12_full_scene_selection_v1`：120场按来源和
   0-edge/edge-1/multi-edge同时平衡，与全部navigation train及G11-C/D2/E逐一互斥。
   历史课程审计也已完成：2D/3D2/5A的selected checkpoint没有可靠Actor更新证据，且
   初始基础模型预算未知。R2改为随机加宽Actor先做n1 broad导航，再补固定单车case并按
   n2/n3/n5扩展。R2-S0已经完成并通过；S1的8-case repair-only首段因broad full success
   降至`0.575`而拒绝。S2至S4已完成两车、三车和五车完整场景首段；当前冻结S4 20k
   best。20k配对准入因timeout多1场而未通过，当前按原场景、seed和阈值评测唯一10k
   fallback。失败S1不得作为warm start。
9. 最终方法表必须重新在同一个冻结manifest上运行全部方法。每个方法使用完全相同的
   scenario ID、顺序、评测seed列表、重复次数、物理参数和终止条件；分析前逐项审计
   manifest哈希、缺失/重复/错序ID。不同场景或不同评测协议的历史汇总值不得混入同一表。

## 问题与主张边界

| 问题 | 当前处理 |
| --- | --- |
| epoch-16原2560场训练集经完整路径复审有11场实际为edge-2 | 当前默认不作“整个系统只见单冲突”的严格主张，因此不为该主张重训Actor；若后续升级为零样本拓扑泛化贡献，必须先重训或明确披露边界 |
| 本机观测难以区分机器人、墙和箱子 | 复用G0/G1的形状软分数、跟踪和相对运动连续特征；历史手工阈值与旧Gate只作基线 |

## 已关闭路线

- standard/dense两个完整场景专家；
- 独立Dense完整Actor及继续扫描reward/Critic；
- epoch-16整网全程续训；
- `5D + 零初始化Residual`；
- `5A + epoch-16动作差`的24维单帧Residual；
- controlled-ego、pair Actor和复杂Critic支线；
- corrected edge-1完整Actor继续训练。

这些路线可以作为论文失败对照或机制诊断引用，但不得从其旧README、checkpoint的
`best`后缀或局部最好成绩推导当前下一步。

## 导航

| 文档 | 用途 |
| --- | --- |
| [论文主线协议](experiments/03_保留专门化/02_论文主线/README.md) | 当前方法、数据边界、准入和实验矩阵 |
| [实验注册表](experiments/EXPERIMENT_REGISTRY.md) | 历史路线状态和可复用结论 |
| [模型注册表](TD3/MODEL_REGISTRY.md) | 模型ID、artifact和使用限制 |
| [数据集索引](experiments/03_保留专门化/02_论文主线/datasets/README.md) | train/validation/test边界 |
| [结果索引](experiments/03_保留专门化/02_论文主线/results/README.md) | 证据所在目录 |
| [执行手册](README_执行文档.md) | 环境、进程和运行记录要求 |
