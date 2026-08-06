# G12 R2-R5完整场景单Actor协议

状态：`design frozen in principle / manifests and numeric optimizer settings pending preregistration`。
日期：`2026-08-06`。

## 1. “完整场景”的定义

完整场景不是“只截取一个冲突窗口”，也不是“先知道本场是强或弱交互”。每个episode从
五车起点开始，直到五车全部到达、碰撞或超时，始终由同一个加宽Actor逐车输出动作。
训练清单不依据0-edge、edge-1或multi-edge删除episode。

冲突拓扑只用于构造重采样、训练期loss mask和结果分层，不进入Actor的24维输入。最终
部署不使用场景名称、其他机器人真值、冲突图或Gate。

## 2. 训练数据

### R2课程

R2复现形成5A的历史有效课程链路：单车局部导航、两车/三车回接、3D2和五车5A。历史
README和脚本只作证据；启动前必须重新登记每一阶段真正采用的case manifest、上游
checkpoint关系、样本预算与停止条件，不能直接执行历史README中的“下一步”。

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
变量。`lambda_keep`、Q归一化、梯度阈值、学习率和20k warm-up必须在R3启动前写入运行
manifest。只允许无闭环模型选择的数值smoke，不能扫描validation成功率。

## 4. 内部validation

R2启动前新建`g12_full_scene_selection_v1`：

- 只从`fixed_v1`原始validation构建；
- 共120场，standard/dense各60场，同时0-edge/edge-1/multi-edge各40场；
- 固定分配为standard的`35/20/5`场与dense的`5/20/35`场，顺序分别对应
  `0-edge/edge-1/multi-edge`；这既保持来源等权，也避免伪造dense中的大量0-edge；
- 与G11-C、G11-D2、G11-E以及所有train scenario ID互斥；
- 按策略无关的完整静态路径拓扑分层，不根据任何模型成功与否筛选；
- manifest、生成脚本、SHA-256和互斥审计在R2启动前提交。

可行性审计：排除G11-C/D2/E共450个不同scenario ID后，standard validation剩余
`141/162/82`场，dense validation剩余`7/161/547`场，顺序均为
`0-edge/edge-1/multi-edge`。上述120场配额均有足够候选。

该120场只用于R2-R4 checkpoint选择。G11-D2和G11-E在大Actor冻结前不可读取或用于
调参；sealed test只在所有方法冻结后运行一次。

## 5. 预算与报告

- R2逐阶段报告agent samples、env steps、Actor updates和validation次数；
- R3固定40k pilot；R4在不改变协议的前提下扩展到320k；
- 最终报告“R2课程总预算 + R4 320k”，不能只写最后阶段预算；
- 参数量、Actor MACs、Gate开销、平均步数、full success、agent success、collision、
  unresolved和timeout全部报告；
- R5至少3 seed，报告均值、标准差和同场配对检验。

## 6. 公平性解释

该baseline在完整`fixed_v1` train上训练，数据覆盖不窄于当前双Actor组件，并允许使用与
interaction Actor同级的训练期几何信息。因此它是对单Actor较为有利的对照。若它仍低于
双Actor+Gate，结果才能支持“收益不只是参数量增加”；若它达到或超过当前方法，则必须
如实收窄论文主张。
