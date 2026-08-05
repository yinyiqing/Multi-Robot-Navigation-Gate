# 当前项目状态

更新时间：`2026-08-05`。

本文件是进入项目后的第一阅读入口。方法定义、实验准入和数据边界以
[论文主线协议](experiments/03_保留专门化/02_论文主线/README.md)为准；历史 README
中的“当前”和“下一步”只代表当时判断。

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

以上都是validation或diagnostic，不是sealed test结果。不同数据集上的数值不得直接
横向比较。

## 当前工作

当前只开发可部署Gate：

1. 冻结5A和epoch-16，不再更新两个Actor。
2. Gate使用本机激光雷达、导航状态及必要的短时历史，不能读取其他机器人里程计、
   场景类别或冲突图。
3. `2.0 m` oracle用于监督、诊断和上界；最终Gate需要判断何时调用避障Actor更有利，
   不能把“附近有障碍物”直接等同于“附近有机器人”。
4. G11-C已完成并归档。G11-D1的4个CPU训练复核seed也已全部通过，离线F1均值为
   `0.84487 +/- 0.00132`；主seed仍为`20260804`，没有从复核seed中挑峰值。
5. 来自导航validation、排除旧G3场景的D2独立200场manifest已经冻结，SHA-256为
   `6250b941f127d550641a621d4253e17ea0770ff3c0cb94e6254e1f26b9f4978a`；D2运行器已冻结，
   将比较5A、epoch-16 always-on、min-LiDAR规则Gate、旧G2-A、A1、B2和oracle，
   并检查B2的过度激活、效率和timeout代价。
6. 独立准入通过后再评估multi-edge。G11-E已经在不启动Gazebo的情况下冻结50场
   exact-edge-2 pilot与后150场confirmation，二者与当前Gate训练、G11-C和G11-D2均
   无场景重叠；实际运行必须等待G11-D2完成。若自然泛化不足，可以在读取sealed test前
   书面修订协议、加入navigation-train的multi-edge数据并重训同一个Gate；此时不再
   声称single-to-multi零样本泛化。

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
