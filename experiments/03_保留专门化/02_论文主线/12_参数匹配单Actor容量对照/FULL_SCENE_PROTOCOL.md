# G12 R2-R5完整场景单Actor协议

状态：`selection manifest frozen / R2 10k admission passed / R3 40k registered`。
日期：`2026-08-08`。

## 1. “完整场景”的定义

完整场景不是“只截取一个冲突窗口”，也不是“先知道本场是强或弱交互”。每个episode从
五车起点开始，直到五车全部到达、碰撞或超时，始终由同一个加宽Actor逐车输出动作。
训练清单不依据0-edge、edge-1或multi-edge删除episode。

冲突拓扑只用于构造重采样、训练期loss mask和结果分层，不进入Actor的24维输入。最终
部署不使用场景名称、其他机器人真值、冲突图或Gate。

## 2. 训练数据

### R2课程

历史审计发现2D、3D2和5A的selected checkpoint均位于Actor解冻前或解冻边界，且最初
`TD3_velodyne_multi_v4`的训练预算未登记，因此不能机械声称从随机初始化复刻了历史5A。
R2保留有效课程思想：先建立单车broad导航，再按n2/n3/n5增加车数。固定单车case用于
诊断能力边界，不再作为进入多车的硬前置。A阶段保持原始24维Critic并完整warm start，
不插入fresh-Critic D支路。完整审计、
预算和停止条件见[R2协议](R2_PROTOCOL.md)与[R2历史课程审计](R2_CURRICULUM_AUDIT.md)。

### R3/R4完整五车分布

| 数据 | 数量 | SHA-256 | 用途 |
| --- | ---: | --- | --- |
| `fixed_v1/standard/train.json.gz` | `3000` | `1cb612513f11fa1a38750fc59b1474c80f4746607d59ef54a73485e2141ff394` | broad stream |
| `fixed_v1/dense/train.json.gz` | `6000` | `d2a09cf8d51b89a366d3661487471d2383ef6ef4490133ab0efd6c59772f9a23` | broad stream |
| `strong_interaction_curriculum_v1/full_train.json.gz` | `2560` | `d5b9b1fb968c8752e54e66f1ea3f25e7c2bf45eae3f012a686008704964da142` | interaction replay stream |

strong pool确定性派生自前两个train split，因此不是额外独立数据。论文必须写成“完整
train分布 + 强交互重采样”，不能把总场景数错误相加为11560。

冻结采样：

```text
50% broad stream
  50% standard/train
  50% dense/train

50% interaction replay stream
  strong full_train，按deep/close/margin = 40/40/20 balanced cycle
```

该设计让大Actor看到完整standard/dense五车分布和自然出现的multi-edge，同时保证强交互
不会因9000场broad pool而被稀释。`g11_a1_gate_v1`不再是正式大Actor训练清单，仅保留
给P1/R1做同协议诊断。

## 3. 训练信息与loss

- Actor：`24 -> 1137 -> 855 -> 2`，始终只读本车24维观测；
- 一个共享Actor全程控制五车，不调用5A、epoch-16或Gate；
- R2结束checkpoint冻结为普通导航参考`pi_ref`；
- R3/R4允许使用与epoch-16同级的训练期几何local Critic和`0.8 self + 0.2 neighbor`
  reward，但这些特权信息不得进入Actor输入；
- 非交互状态通过行为保持项约束，交互状态允许学习冲突处理；
- Actor Q项必须做尺度归一化并裁剪梯度。

目标形式：

```text
L_actor = -normalized_Q(s, pi(s))
          + lambda_keep * I[d_nearest > 2.0 m]
            * ||pi(s) - pi_ref(s)||^2
```

`2.0 m`真值只生成训练期mask，与interaction-epoch16的训练信息边界一致；推理时不存在该
变量。R3已经冻结`lambda_keep=1.0`、Q归一化alpha `1.0`、Actor梯度范数裁剪`1.0`、
Actor/Critic学习率`1e-5/8e-5`和21k Actor解冻阈值（20k warm-up加episode guard）。只
允许无闭环模型选择的数值smoke，不能扫描validation成功率。完整数值见
[R3协议](R3_PROTOCOL.md)。

## 4. 内部validation

R2启动前新建`g12_full_scene_selection_v1`：

- 只从`fixed_v1`原始validation构建；
- 共120场，standard/dense各60场，同时0-edge/edge-1/multi-edge各40场；
- 固定分配为standard的`35/20/5`场与dense的`5/20/35`场，顺序分别对应
  `0-edge/edge-1/multi-edge`；这既保持来源等权，也避免伪造dense中的大量0-edge；
- 与G11-C、G11-D2、G11-E以及所有train scenario ID互斥；
- 按原始manifest中的策略无关静态路径冲突指标`metrics.conflict_edge_count`分层，不根据
  任何模型成功与否筛选；该字段使用生成器冻结的`8 s` horizon，不称为full-horizon；
- manifest、生成脚本、SHA-256和互斥审计在R2启动前提交。

可行性审计：排除G11-C/D2/E共450个不同scenario ID后，standard validation剩余
`141/162/82`场，dense validation剩余`7/161/547`场，顺序均为
`0-edge/edge-1/multi-edge`。上述120场配额均有足够候选。

selection已经冻结：

```text
fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz
SHA-256 52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635
```

六层实际候选与上述可行性审计一致；120个ID与全部navigation train、G11-C、G11-D2和
G11-E逐一交集为0。构建与审计细节见该view的README。

该120场只用于R2-R4 checkpoint选择。G11-D2和G11-E在大Actor冻结前不可读取或用于
调参；sealed test只在所有方法冻结后运行一次。

## 5. 最终配对比较契约

R5不是把不同实验的汇总数放到一张表。最终比较必须满足：

1. 5A、epoch-16 always-on、规则Gate、A1、B2、最终Gate、参数匹配单Actor和oracle读取
   同一个冻结manifest，并使用完全相同的scenario ID及顺序；
2. 所有方法共享同一组evaluation rollout seed、相同仿真重复次数、物理参数、最大步数、
   success/collision/timeout定义和仿真启动方式；训练seed与evaluation seed分别报告；
3. 每个结果逐行保存`scenario_id`和repeat ID。分析前校验manifest SHA-256、结果行数、
   ID顺序、重复/缺失ID和repeat覆盖；不一致时直接失败，不生成比较指标；
4. 同场波动通过相同repeat列表、逐场配对统计和至少一次复核运行处理。相同scenario ID但
   不同评测协议的历史结果不能替代最终复测；
5. full success使用同场McNemar exact和paired bootstrap，同时报告改善/退化场数；连续
   指标按同一个`scenario_id + repeat`键配对。

R5执行脚本必须在启动前固定上述manifest哈希和evaluation seed列表，并在完成后生成机器
可读审计结果。审计未通过的运行不得进入论文主表。

## 6. 预算与报告

- R2逐阶段报告agent samples、env steps、Actor updates和validation次数；
- R3固定40k pilot；R4在不改变协议的前提下扩展到320k；
- 最终报告“R2课程总预算 + R4 320k”，不能只写最后阶段预算；
- 参数量、Actor MACs、Gate开销、平均步数、full success、agent success、collision、
  unresolved和timeout全部报告；
- R5至少3 seed，报告均值、标准差和同场配对检验。

## 7. 公平性解释

该baseline在完整`fixed_v1` train上训练，数据覆盖不窄于当前双Actor组件，并允许使用与
interaction Actor同级的训练期几何信息。因此它是对单Actor较为有利的对照。若它仍低于
双Actor+Gate，结果才能支持“收益不只是参数量增加”；若它达到或超过当前方法，则必须
如实收窄论文主张。
