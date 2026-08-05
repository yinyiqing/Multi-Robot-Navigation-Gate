# 可部署在线 Gate 研究

状态：`G11-A1 passed / G11-B2 trained / closed-loop pilot pending`。更新日期：`2026-08-05`。

本目录只研究两个冻结 Actor 之间的在线切换：

- Actor N：`generalist-5a`，负责普通导航；
- Actor I：`interaction-epoch16`，负责局部机器人交互；
- Gate：逐机器人、逐时刻选择 Actor，并负责进入和退出交互模式。

两个 Actor 不再更新。Gate 部署时只能读取本机传感器、本机导航状态、两个 Actor
对同一观测给出的候选动作以及本机短时历史。其他机器人真值、场景标签和冲突图只能
在训练期生成监督或训练 Critic，不能进入部署 Gate。

## 研究主张边界

当前导师确认的是训练一个Gate完成两个冻结Actor的在线分工，并未要求single-edge到
multi-edge零样本泛化。G11-A1/B保留0-edge与single-edge训练协议，是为了先得到干净的
第一候选；multi-edge先作为泛化评测和潜在附加贡献。若第一候选在multi-edge上不足，
可以在读取sealed test前加入navigation-train的multi-edge数据，仍然只训练同一个Gate，
但必须删除零样本泛化表述并重做全部准入。不得为了追求附加主张而延误可用Gate主线。

## 1. 当前问题不是“检测到机器人”

最终问题是：

> 在当前局部历史下，是否应该暂时把控制权从 5A 交给 epoch-16，以及何时交还？

旧 G2-A 把它近似成“前方 2 m 内是否存在机器人”。这能产生正收益，但不是完整的
Actor 选择目标：

| 固定200场结果 | full success |
| --- | ---: |
| 5A | `0.300` |
| 旧 learned Gate | `0.415` |
| 真值距离 Oracle | `0.555` |

旧 Gate 的提升具有方向性和统计证据，但只恢复了 `45.1%` 的同场 Oracle 增益，低于
预设的 `60%` 准入线；收益主要集中在 edge-1/2，在 edge>=3 上没有净提升。

## 2. 旧 Gate 的主要缺口

1. **训练分布不一致**：G2-A 只在 5A 轨迹上训练，部署后 Gate 会改变轨迹分布。
2. **标签与执行契约不一致**：在线上界使用 360 度最近机器人真值，实际采用的 G2-A
   checkpoint 学的是前方 180 度标签。
3. **逐帧目标不等于切换序列**：旧 MLP 独立分类每一帧，短时历史只经过手工 tracker
   汇总，没有直接学习进入、保持和退出的时序状态。
4. **没有利用两个 Actor 的分歧**：两者对当前观测的候选动作和动作差是完全可部署的
   信息，旧 Gate 没有读取。
5. **目标是交互存在性而非结果收益**：失败场的 epoch-16 激活率反而更高，说明继续
   降阈值、延长保持时间不能补足差距。
6. **反事实路线不可复现**：从同一 Gazebo 锚点做单次8步分叉会放大 LiDAR 噪声，
   不能生成可靠硬标签。

## 3. 首选路线

当前优先执行 [`METHOD_CANDIDATES.md`](METHOD_CANDIDATES.md) 中的方法 A：

> 特权 Oracle 时序蒸馏 + student-rollout 数据聚合。

它仍利用已经证明有效的 `2.0 m` Oracle，但不再做不可重复的状态分叉：Gate 在真实
走过的每个状态上直接读取训练期真值标签；部署网络只看本机信息。先在已有 G2-A
shard 上完成低成本离线复核，只有时序模型明确超过旧静态 MLP 才采集新轨迹。

### G11-A0：已有数据兼容性诊断

- 复用现有 train/validation shard，sealed test 不读取；
- 旧 shard 的 positive-edge 层包含 multi-edge，因此本阶段只比较表示和训练流程，
  产生的 checkpoint 不进入当前方法或拓扑泛化实验；
- 输入增加 5A 动作、epoch-16 动作和动作差，并直接学习短时序；A0不读取上一真值
  mode，避免teacher forcing泄漏；
- 比较静态 MLP、GRU/TCN 时序 Gate；
- 同时报 360 度标签和前方标签，不用其中一个冒充另一个；
- 除 frame precision/recall/FPR 外，增加切换事件召回、进入延迟、持续区间 IoU、
  0-edge 误激活和切换次数。

G11-A0已完成5个seed复核。单帧动作特征S1没有超过S0；8帧GRU T1在360度标签下的
F1、AP、区间IoU和事件precision均为`5/5` seed提高，切换次数均减少。即使逐seed
约束T1的总体FPR和standard/weak FPR不高于S0，F1仍提高`1.17-2.41`个百分点，
因此授权A1采集。旧混合shard、A0 checkpoint和该离线分类指标不得写成闭环结果。

### G11-A1：当前协议正式离线 pilot

- 仅使用导航 train 内部重新划分的 0-edge 与 corrected full-horizon edge-1；
- train/validation scenario ID 互斥；
- 只有 A0 证明时序或 Actor 分歧有明确增益后才采集；
- A1 checkpoint 才有资格进入 student-rollout 和闭环准入。

G11-A1已在当前协议的640场train与120场内部validation上完成。预注册主seed通过全部
离线准入，4个复核seed也全部保持方向：T1相对S0的F1、AP和区间IoU均提高，总体及
0-edge FPR均下降，event recall下降不超过限制，切换次数均减少。因此固定主seed T1
进入G11-B；该结论不是闭环导航成绩。

### G11-B：Student-rollout 数据聚合

- 用 G11-A Gate 运行固定 train 场景；
- 在它实际访问的每个状态查询真值 Oracle 标签，不回退仿真、不做反事实分叉；
- 聚合A1的5A轨迹与student轨迹，并统一使用Oracle监督标签重新训练同一个时序Gate；
  第一轮不额外采集Oracle行为轨迹，因为DAgger所需的是student访问状态上的teacher查询；
- 使用模型集成或 MC dropout 估计不确定性，不确定时默认 5A；
- validation 只用于冻结阈值、滞回和最短保持时间。

G11-B在线T1控制器与1场student smoke已经通过：8帧历史和2步评估间隔与A1采集尺度
一致，student shard的manifest、时序、Oracle标签和内嵌运行元数据审计通过。正式
G11-B1也已完成同一navigation-train manifest的`640/640`场采集并通过全量审计；当前
G11-B2来源平衡聚合重训已经完成，尚未得到闭环validation结论。

### G11-C：端到端准入

第一步见[`G11_C_50场闭环pilot`](G11_C_50场闭环pilot/README.md)：在A1内部validation
固定50场、两个重复上配对比较5A、A1与B2。它只决定是否保留student聚合；通过后才进入
以下更大独立准入。

沿用旧 G3 的固定协议，至少比较 5A、epoch-16 always-on、最小 LiDAR/TTC 规则、
旧 G2-A、新 Gate 和真值 Oracle。200场准入仍要求：

- full success 达到 `>=0.45`，或恢复同场 Oracle 增益 `>=60%`；
- 配对改善明确多于退化；
- timeout 不系统增加；
- standard/0-edge full success 下降不超过3个百分点；
- edge>=2 和 max-degree>=2 保持正收益。

## 4. 何时升级到强化学习 Gate

只有出现以下情况才进入“冻结 Options + Gate RL”：

- 时序 Gate 已能较好复现 Oracle，但闭环收益仍明显不足，说明距离标签与实际收益不等价；
- 或 360 度 Oracle 在本机历史下存在明显不可观测部分，继续做分类蒸馏已经到达上限。

升级时只训练低频二值 Gate；两个 Actor 是固定 options。优先使用局部 recurrent policy
和训练期 privileged centralized critic，并加入切换成本。该方法直接优化导航结果，
但仿真样本成本和训练风险更高，因此不是第一轮实验。

## 5. 当前可执行性

本机已有：

- G2-A pilot train/validation 各100个互斥场景 shard；
- 冻结5A和epoch-16权重；
- 冻结G0 detector和旧G2-A Gate checkpoint；
- 可恢复逐机器人时间顺序的frame index、ego index和timestamp。

G11-A0只回答时序表示是否值得继续；G11-A1已经单独采集当前协议数据并通过离线准入。
G11-B1由预注册主seed `20260804`的T1 epoch 2 checkpoint生成，正式student数据已经
冻结。G11-B2仍只使用导航train聚合训练；完成聚合重训和小validation阈值冻结前，不
读取sealed test。

## 6. 明确不重复

- 不继续调最小 LiDAR、停车、靠右或单一 TTC 硬阈值作为最终方法；
- 不再要求 G0 输出机器人/墙的单帧硬分类；
- 不重跑 `20-bin + GRU` 或小样本逐帧风险分类；
- 不恢复单次8步反事实分叉或 Gazebo 锚点恢复；
- 不用两个历史 TD3 Critic 的绝对值直接比较 Actor；
- 不做动作软融合、Residual Actor 或完整 Dense Actor；
- 不通过反复扫描阈值掩盖标签或表示问题。

相关论文及其对本项目的直接约束见 [`RELATED_WORK.md`](RELATED_WORK.md)。
