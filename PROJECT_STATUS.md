# 当前项目状态

更新时间：`2026-08-14`。

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

Actor I = avoidance-epoch17
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
| `avoidance-epoch17` | `avoidance_actor_from_5a_balanced_continue_e20_s20260813_best_actor.pth` | `149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5` | 条件避障 Actor |

两者的部署 Actor 输入均为本车24维观测。`avoidance-epoch17`训练时使用了仿真真值
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
22. G12-R2五车20k配对准入中，R2相对5A的full success为`0.6917 vs 0.5583`，但timeout
    为`3/120`并超过`0.020`上限，故严格判定未通过。预先登记的唯一10k fallback随后通过
    全部五项准入：full success为`0.7000 vs 0.5583`，collision为`0.1000 vs 0.1900`，
    timeout为`0/120`，总体20场改善、3场退化，`p=0.000488`。10k与20k之间full success
    无显著差异（11场改善、10场退化，`p=1.0`），选择10k的依据是约束和效率，而非峰值扫描。
23. G12-R3的40k pilot已经完成但未通过：训练清单按`standard/strong/dense/strong`确定性循环，
    R2-10k只加载Actor并固定为参考；fresh geometry Critic在21k阈值后才解冻Actor，以
    保护20k评测边界；之后使用
    `1e-5` Actor学习率、归一化Q、非交互状态`lambda_keep=1.0`和`1.0`梯度裁剪。R3-20k
    Actor冻结时full success为`0.667`，40k解冻更新后降至`0.575`，低于R2-10k参考
    `0.700`，因此不启动R4。当时冻结R2-10k作为大Actor候选；后续公平性复审因其
    额外课程和预算不匹配，已将它降级为cross-protocol诊断。R3不读取D2、E或sealed test，
    也不更新5A和epoch-16。完整记录见`12_参数匹配单Actor容量对照/R3_RESULTS.md`。
24. 当前普通Actor重训N1已按R2-style单车broad协议完成100k。120场validation的full
    success为`0.975/1.000/1.000/1.000/1.000`，collision和timeout从epoch 2起均为`0`；
    best checkpoint由epoch 3更新。该结果证明原宽度Actor能从随机初始化建立单车基础导航，
    不证明两车、五车或冲突能力；下一步进入N2两车broad。
25. 当前普通Actor重训N2两车broad首段已完成20k。10k/20k的full success均为`0.925`，
    agent success均为`0.950`，collision均为`0.046`，timeout均为`0.008`；20k平均步数
    从`13.3`降至`11.9`并由脚本更新为best。N2通过首段准入，说明原宽度Actor能稳定进入
    两车broad；由于n2 validation冲突场景比例低，不能据此声称冲突能力或五车能力。
26. 当前普通Actor重训N3三车broad首段已完成20k。10k/20k的full success为
    `0.833/0.892`，agent success为`0.919/0.953`，collision为`0.075/0.036`，
    timeout为`0.017/0.025`；20k由脚本更新为best并通过准入。该结果说明原宽Actor能稳定
    进入三车broad，且与G12-R2-S3加宽Actor同阶段同级；但timeout/unresolved仍需关注，
    不得单独声称原宽优于加宽或已经解决五车冲突。
27. 当前普通Actor重训N5五车broad首段已完成20k。10k/20k的full success为
    `0.667/0.717`，agent success为`0.890/0.920`，collision为`0.083/0.062`，
    timeout为`0.117/0.092`，平均步数为`57.1/50.6`；20k由脚本更新为best并通过首段
    准入。该结果说明原宽Actor已形成可评估的五车普通Actor候选，但timeout和效率明显
    不足，不能直接声称优于G12-R2或可替换旧5A；下一步必须在同一冻结manifest上做5A、
    N5、B2/最终Gate、R2-10k和oracle的配对分层准入。
28. 当前普通ActorN5-20k同场配对准入已完成。旧5A/R2-10k/N5-20k在同一120场N5
    validation上的full success为`0.558/0.700/0.700`，agent success为
    `0.810/0.900/0.910`，collision为`0.190/0.100/0.075`，timeout为
    `0.000/0.000/0.067`，平均步数为`20.08/20.39/46.15`。N5相对旧5A overall为24场
    改善、7场退化，`p=0.00333`，multi-edge为13场改善、1场退化，`p=0.00183`；但
    timeout超过`+0.02`准入上限，因此严格判定未通过，不能直接替换旧5A。N5相对R2-10k
    为13场改善、13场退化，`p=1.0`，full success无差异但效率明显更差。
29. N5 timeout诊断显示，8个timeout case全部无碰撞，其中7个为`4S/0C/1U`，1个为
    `3S/0C/2U`；N5在这些case中累计`31`个agent成功、`0`碰撞、`9` unresolved。5A/R2-10k
    在同8场中分别有`4/8`和`5/8` full success。这说明N5的主要问题是避障或等待后缺少
    恢复推进，而不是安全不足；当前不启动local critic分支，若修Actor应登记为
    efficiency repair，但主线优先回到Gate。
30. N5 efficiency repair E1已完成但未通过准入：5k/10k的full success为
    `0.675/0.650`，collision为`0.102/0.123`，timeout均为`0.025`，平均步数为
    `29.9/27.3`。它证明timeout和效率可以被推进/恢复奖励快速改善，但当前配置过度催促
    Actor，导致碰撞超过`0.10`上限且full success低于N5同场基线`0.700`；不得用E1
    latest或epoch2替换N5-20k/旧5A。若继续E2，应从E1 epoch1或N5-20k重新出发并降低
    推进强度或加入更明确的碰撞保护。
31. N5 efficiency repair E2已经完成同场120场配对准入。相对旧5A，E2的full success为
    `0.7500 vs 0.5583`、collision为`0.0567 vs 0.1900`，但timeout仍为`0.0667`，
    超过冻结上限`+0.02`，因此严格判定未通过，不能直接替换旧5A。相对旧N5-20k，E2
    只是在full success和collision上更好，timeout没有改善，平均步数还更高。
    训练内的epoch 1 best仍保留，但它只能算更强的普通Actor候选，不是最终普通Actor。
32. 同场把E2的交互窗口替换为epoch-16真值oracle后，overall full success从`0.7500`
    降到`0.7250`，collision从`0.0567`升到`0.0800`，平均步数从`49.28`升到`50.56`，
    timeout仍为`0.0667`。这说明当前瓶颈仍在普通Actor的恢复推进，不是“把避障Actor接
    上去就能把E2抬过去”。
33. E2 recovery-oracle诊断已完成120场：full success为`0.7750`，collision为`0.0550`，
    timeout为`0.0500`，平均步数为`41.98`；旧2米oracle分别为`0.7250/0.0800/0.0667/50.56`。
    recovery规则把epoch-16加权动作占比从`44.6%`降到`19.1%`，收益集中在dense和
    multi-edge。但E2基线来自seed `20260817`，该诊断来自`20260818`，因此`0.775 vs
    0.750`还不是严格同随机重复的因果提升。
34. `2026-08-12`协议显式授权E2候选支线的最小Actor pilot：先补同seed E2-only控制，
    再从E2 warm start训练40k的条件交互Actor `I-E2`。该授权不改变旧5A+epoch-16冻结
    fallback，也不授权直接跑满320k；I-E2只在2米交互窗口执行和更新，训练侧使用87维
    邻域Critic和动态reward，Actor保持24维部署输入，21k前冻结Actor校准fresh Critic。
35. I-E2 40k pilot和同seed 120场复测已经完成。E2、E2+旧epoch-16 recovery、
    E2+I-E2 recovery的full success分别为`0.6583/0.7750/0.7333`，collision为
    `0.090/0.055/0.070`，timeout为`0.100/0.050/0.075`。I-E2相对E2为19场改善、
    10场退化，`p=0.1360`，说明训练有效但尚未通过显著性准入；它在edge-1达到`0.850`，
    但multi-edge只有`0.425`，低于旧epoch-16的`0.600`。当前只授权逐case轨迹与动作诊断，
    不授权直接延长I-E2训练或开始新Gate训练。
36. I-E2离线轨迹诊断已经完成。强交互训练集和验证集的清单指标均为单冲突边，
    而N5评测中的multi-edge包含2至5条冲突边；这解释了I-E2在edge-1达到`0.850`、
    但multi-edge只有`0.425`。I-E2在13个相对旧epoch-16退化case中的交互动作占比为
    `0.396`，交互平均线速度为`0.015`、停滞帧比例为`0.928`；旧epoch-16对应为
    `0.144/0.147/0.775`。当前判定为训练分布缺少multi-edge且恢复推进不足，不是简单的
    训练时长不足；不得直接追加40k，下一实验必须先登记multi-edge训练分布修订。
37. I-E2-M多冲突pilot及matched复测已经完成但未通过。E2、E2+旧epoch-16 recovery、
    E2+I-E2-M的full success为`0.7417/0.7750/0.6833`；multi-edge为
    `0.450/0.600/0.400`。实际训练覆盖edge-1/edge-2/edge-3+为`169/127/126`个episode，
    所以失败不是缺少multi-edge执行。20k冻结Actor到40k解冻Actor的内部full success只从
    `0.610`到`0.615`，timeout从`0.095`升到`0.135`。同观测动作审计显示I-E2-M mean
    linear仅`0.0543`，接近E2的`0.0489`，而旧epoch-16为`0.1571`；根因是Actor梯度只
    覆盖1m内仍闭合的risk状态，排除了停滞恢复和release阶段。不得追加40k；完整诊断见
    `15_E2恢复Actor诊断与训练/I_E2_M_DIAGNOSIS.md`。
38. I-E2-F4四阶段pilot已完成并拒绝。Actor冻结的20k与21k后更新的40k在同一200场
    internal validation上的full success为`0.630/0.605`，collision为`0.094/0.099`，
    timeout为`0.090/0.100`，平均步数为`54.64/66.10`。Actor梯度门持续通过，所以失败
    不是更新未发生；完整2米窗口更新与更强恢复奖励反而使成功、安全和效率同时退化。
    自动保存的`best`来自Actor冻结阶段，不是训练出的避障Actor；不追加预算，不做基于
    F4的Gate。训练前同seed matched控制显示E2+旧epoch-16 recovery相对E2 full success
    为`0.7750 vs 0.6917`，15场改善、5场退化，exact `p=0.04139`。
39. 避障Actor epoch 17-20固定续训已完成。epoch 16/17/18/19/20的internal validation
    full success为`0.7071/0.7429/0.6500/0.6857/0.7071`；epoch 17为唯一新候选，但其
    timeout从`0.0143`升到`0.0357`、平均步数从`54.5`升到`57.9`。后续epoch的避障Actor
    占比和平均步数最高升至`69.6%/87.2`，确认继续更新出现过度保守趋势。停止于epoch 20，
    不进入epoch 21；原epoch 16继续冻结，epoch 17只授权独立matched admission。
40. epoch 16与epoch 17的独立matched admission已完成`480/480`场并通过数据审计。
    合并两个seed后，epoch 17相对epoch 16的full success为`0.6875 vs 0.6333`、collision
    为`0.0967 vs 0.1075`、timeout均为`0.0125`，但平均步数为`37.46 vs 32.32`，达到
    `1.159x`并超过预注册的`1.10x`上限。逐场为40改善、27退化，exact `p=0.1421`；收益
    主要来自edge-1，multi-edge仅增加`0.0125`。它未通过原预注册的效率硬门槛。
41. 在读取sealed test前，项目于`2026-08-14`修订避障Actor选择规则：full success为
    主选择指标，collision和timeout为安全约束，平均步数作为必须报告的效率代价而非单独
    否决项。依据internal与独立matched validation方向一致，冻结epoch 17进入最终Gate
    重训；epoch 16保留为消融对照。该修订和epoch 17的`1.159x`步数代价必须完整披露，
    不得表述为通过原效率准入。
42. G11-F已用epoch-17动作特征离线重建A1 Gate并通过全部离线门槛；随后完成并审计
    `640/640`场epoch-17 student rollout，共`43,827`个Gate帧，dataset SHA-256为
    `5037144924ceb5e433a5e02a17cdffa5a4338f016f08208dc7a64854548887e8`。采集训练集
    full success为`0.7422`，只作行为诊断。下一步是聚合原5A shard与新student shard
    训练G11-F-B2，不得混入旧epoch-16 student数据。
43. G11-F-B2聚合Gate已完成。它通过冻结S0的FPR上限，但相对新F-A1在同一内部
    validation上的F1/AP/区间IoU分别为`-0.00729/-0.00292/-0.01106`，切换少6次。
    该离线现象不能决定闭环优劣；下一步只做固定50场、两个重复的`5A/F-A1/F-B2`
    闭环pilot，再决定最终Gate候选。
44. G11-F-C固定闭环pilot已完成`300/300`场。5A/F-A1/F-B2合并full success分别为
    `0.640/0.710/0.680`，collision分别为`0.104/0.088/0.080`，timeout分别为
    `0/0/0.030`，平均步数分别为`21.91/33.41/44.87`。F-B2相对F-A1为9场改善、12场
    退化，exact `p=0.6636`，且成功率低3个百分点、timeout和步数更高；因此拒绝F-B2，
    冻结F-A1作为当前Gate候选。F-A1与R2-10k大Actor尚未在同一manifest上比较，历史
    R2/B2数字不得替代该实验。
45. G11-F-C的R2-10k同场补充pilot已完成`100/100`场。5A/F-A1/F-B2/R2-10k的full
    success为`0.640/0.710/0.680/0.760`，collision为`0.104/0.088/0.080/0.060`，timeout
    为`0/0/0.030/0.030`，平均步数为`21.91/33.41/44.87/28.44`。F-A1相对R2-10k为
    9场改善、14场退化，exact `p=0.4049`。但R2-10k使用了从随机初始化开始的
    `n1 -> n2 -> n3 -> n5`额外课程，与5A/双Actor的训练流程和预算不匹配。因此该结果
    降级为历史cross-protocol诊断，保留披露但不进入论文公平容量baseline排名，也不再
    用它决定Gate是否继续。
46. 为控制R2与历史5A训练血缘不同这一因素，已登记G12-R2B参数匹配Actor：从3D2 Actor
    函数保持扩宽，复制5A的五车standard、individual reward、fresh 24维Critic、20k
    Actor冻结、学习率和30k总预算。R2B是当前的流程匹配公平性对照；R2的数据保留
    为历史诊断，但不能替代R2B进入正式比较。R2B也不能以“必须低于Gate”为选择目标。
47. G12-R2B已完成30k。10k/20k/30k internal validation的full success为
    `0.533/0.508/0.042`，collision为`0.195/0.200/0.418`，timeout为`0/0/0.150`。
    10k仍与扩宽前3D2函数等价；约10k Actor更新后的30k明显坍塌。因此R2B没有形成可用
    的稳定大Actor，只证明5A recipe不适合稳定训练扩宽Actor。项目接受退化作为对照
    结果，因此冻结真正经过Actor更新的R2B-30k作为流程/预算匹配大Actor baseline；
    不使用冻结期自动best。正式数值须在G11-F-C同一manifest上补评后确定。

以上都是validation或diagnostic，不是sealed test结果。不同数据集上的数值不得直接
横向比较。

## 当前工作

当前主线仍是可部署Gate；参数匹配单Actor只作论文公平对照：

1. 主线冻结旧5A与epoch-17避障Actor。epoch 17未通过原`1.10x`效率硬门槛，但在读取
   sealed test前按修订后的“full success主指标、collision/timeout安全约束、步数报告
   代价”规则入选；epoch 16保留为消融对照，不再继续Actor训练。
2. Gate使用本机激光雷达、导航状态及必要的短时历史，不能读取其他机器人里程计、
   场景类别或冲突图。
3. `2.0 m` oracle用于监督、诊断和上界；最终Gate需要判断何时调用避障Actor更有利，
   不能把“附近有障碍物”直接等同于“附近有机器人”。
4. 旧epoch-16的G11-D2已经归档：B2导航收益成立但效率准入失败，只作历史证据。当前
   G11-F已完成epoch-17 A1重建、student rollout和闭环pilot，F-A1冻结为Gate候选；
   F-B2只作DAgger消融，不得直接进入最终方法。
5. 下一项Gate实验是multi-edge边界评估。G11-E已经冻结50场
   exact-edge-2 pilot与后150场confirmation，二者与当前Gate训练、G11-C和G11-D2均
   无场景重叠。若自然泛化不足，可以在读取sealed test前
   书面修订协议、加入navigation-train的multi-edge数据并重训同一个Gate；此时不再
   声称single-to-multi零样本泛化。
6. G12-P1已停止并降级为训练稳定性诊断。G12-R3也已因Actor解冻后退化而停止，不再扩展
   R4。后续容量对照遵循
   [G12-R公平路线](experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/REVISED_PLAN.md)：
   先做原宽度控制，再让参数匹配Actor从头完成普通导航课程和完整五车分布联合训练。
   任何新长跑必须先冻结具体超参数和manifest，不得用D2或sealed test调参。R2因
   额外课程和预算不匹配已降级为cross-protocol诊断；流程匹配的R2B-30k即使
   退化也保留为公平大Actor对照，当前补做G11-F-C同场评测。
7. G12-R1已归档到`logs/archive/diagnostic/g12_r1/`，只作机制诊断，其checkpoint不得
   作为R2 warm start或论文主性能模型。
8. G12完整场景内部validation已经冻结为`g12_full_scene_selection_v1`：120场按来源和
   0-edge/edge-1/multi-edge同时平衡，与全部navigation train及G11-C/D2/E逐一互斥。
   历史课程审计也已完成：2D/3D2/5A的selected checkpoint没有可靠Actor更新证据，且
   初始基础模型预算未知。R2改为随机加宽Actor先做n1 broad导航，再补固定单车case并按
   n2/n3/n5扩展。R2-S0已经完成并通过；S1的8-case repair-only首段因broad full success
   降至`0.575`而拒绝。S2至S4已完成两车、三车和五车完整场景首段；S4训练内best为20k，
   但20k配对准入因timeout多1场未通过，唯一10k fallback通过全部五项准入，因此冻结
   S4 10k作为R2参考。R3 40k的混合清单、数值配置、21k Actor解冻阈值和停止条件已经
   冻结并执行完成，但Actor解冻后退化。R2-10k与B2的同场数据仅保留为历史
   cross-protocol诊断，不再作为候选容量baseline。失败S1不得作为warm start。
9. 新的普通Actor重训 baseline 已改为课程式路线，目标不是替代当前Gate主线，而是为
   `generalist-5a` 的角色重建一个更强的普通Actor N。`2026-08-09`直接五车 fresh local
   critic pilot 已中止归档；正式路线从原宽度随机Actor/Critic的单车 broad N1 开始，
   再按 N2/N3/N5 扩展，local critic 不再在五车阶段突然 fresh 接入。N1已经完成并通过，
   N2、N3和N5首段也已经完成并通过；N5同场配对准入因timeout失败，不能直接替换旧5A。
   `epoch-16`已经是使用邻域
   critic和交互reward训练出的避障专家，因此普通Actor N的clean课程继续保持原24维critic。
   当前登记的N5 efficiency repair E1只处理N5-20k的timeout/效率问题，不改变普通Actor的
   观测结构，也不把它训练成第二个避障Actor。
10. 最终方法表必须重新在同一个冻结manifest上运行全部方法。每个方法使用完全相同的
   scenario ID、顺序、评测seed列表、重复次数、物理参数和终止条件；分析前逐项审计
    manifest哈希、缺失/重复/错序ID。不同场景或不同评测协议的历史汇总值不得混入同一表。
11. 当前三个E2配套Actor pilot均已完成并拒绝。首版I-E2因单冲突训练分布导致multi-edge不足；
    I-E2-M修复了数据覆盖，但训练目标仍只覆盖逼近风险状态，没有学到停滞恢复和交还。
    I-E2-F4进一步覆盖完整交互窗口和四阶段恢复信号，但Actor更新后全面退化。三者均不得
    替换旧epoch-16；当前不授权继续Actor长跑或启动基于新Actor的Gate训练。历史协议位于
    [E2恢复Actor诊断与训练](experiments/03_保留专门化/02_论文主线/15_E2恢复Actor诊断与训练/README.md)
    。

12. 不再沿用当前TD3目标做第五次E2配套Actor权重微调。若未来重启，必须先提出与F4有
    本质区别的训练形式，并在离线或短窗反事实上证明有效后重新书面修订协议。
13. `I-E2-F4`已归档为失败诊断；完整结果见
    `15_E2恢复Actor诊断与训练/I_E2_F4_RESULTS.md`。

## 问题与主张边界

| 问题 | 当前处理 |
| --- | --- |
| epoch-16原2560场训练集经完整路径复审有11场实际为edge-2 | 当前默认不作“整个系统只见单冲突”的严格主张，因此不为该主张重训Actor；若后续升级为零样本拓扑泛化贡献，必须先重训或明确披露边界 |
| 本机观测难以区分机器人、墙和箱子 | 复用G0/G1的形状软分数、跟踪和相对运动连续特征；历史手工阈值与旧Gate只作基线 |

## 已关闭路线

- standard/dense两个完整场景专家；
- 独立Dense完整Actor及继续扫描reward/Critic；
- epoch-16无边界整网续训；当前唯一例外是已登记的避障Actor epoch 17-20固定续训，
  该分支不得覆盖原artifact或自动进入epoch 21；
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
