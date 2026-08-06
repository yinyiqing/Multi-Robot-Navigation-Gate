# 实验注册表

更新时间：`2026-08-06`。当前方法以[PROJECT_STATUS](../PROJECT_STATUS.md)和
[论文主线协议](03_保留专门化/02_论文主线/README.md)为准。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| `current` | 当前方法组成或正在开发的组件 |
| `frozen evidence` | 已冻结、可引用的基线或方法证据 |
| `diagnostic` | 只回答机制问题，不能作为最终方法成绩 |
| `rejected` | 已完成且不再继续的路线 |
| `invalid` | 协议、实现或运行不完整，数值不得用于趋势和模型选择 |
| `historical` | 早期基础工作，只用于追溯来源 |

`best`只表示某次运行内部保存规则选出的checkpoint，不表示它是当前方法或跨实验最优。

## 当前组件

| 实验/组件 | 状态 | 可用结论 | 不得误读为 |
| --- | --- | --- | --- |
| `generalist-5a` | `frozen evidence / current component` | 普通导航Actor；0-edge validation full success `0.875` | 对所有dense冲突都已解决 |
| `interaction-epoch16` | `frozen evidence / current component` | 按局部oracle调用可将单冲突full success `0.421 -> 0.700` | 可全程独立导航的Dense Actor |
| `2.0 m interaction oracle` | `diagnostic upper bound` | dense validation full success `0.309 -> 0.545` | 已训练Gate或可部署方法 |
| `G0/G1 perception frontend` | `frozen diagnostic` | 原始点云覆盖足够；保留形状和相对运动连续证据 | 已可靠完成机器人硬分类 |
| `G2-A learned Gate` | `frozen baseline / admission failed` | exact-edge-2有`+0.080`方向性收益 | 最终Gate或统计显著结果 |
| `deployable interaction Gate` | `current / G11-D2 navigation passed, efficiency failed` | B2在独立200场相对5A导航收益显著，但interaction占比和步数过高 | 已通过最终效率准入或可以读取部署期真值 |
| `G12-R2-S0` | `current registered baseline` | 随机加宽Actor的n1 broad导航阶段，100k samples | 参数匹配单Actor已完成或容量不足 |

## 路线级分类

| 路线 | 主要位置 | 状态 | 最终决定 |
| --- | --- | --- | --- |
| 单车到五车课程学习 | [`02_课程学习/`](02_课程学习/) | `historical / frozen source` | 形成5A和5D；不从旧stage继续训练 |
| standard expert微调 | [`results/02_普通Actor_失败对照`](03_保留专门化/02_论文主线/results/02_普通Actor_失败对照/README.md) | `rejected` | 未稳定超过冻结模型 |
| 独立Dense完整Actor | [`results/03_强交互Actor_研发记录`](03_保留专门化/02_论文主线/results/03_强交互Actor_研发记录/README.md) | `rejected` | 在危险加速和保守timeout之间退化，不继续扫reward/Critic |
| 条件交互Actor | [`results/05_当前冻结方案`](03_保留专门化/02_论文主线/results/05_当前冻结方案/README.md) | `frozen evidence` | 保留epoch-16作为当前条件避障Actor |
| pair Actor / controlled ego | 历史诊断及已隔离归档 | `rejected` | 不进入当前Actor或Gate训练；默认不检索已隔离目录 |
| 24维Residual融合 | [`09_单冲突Residual组合主线`](03_保留专门化/02_论文主线/09_单冲突Residual组合主线/README.md) | `rejected` | R0未通过；不启动R1/R2 |
| epoch-16整网全程续训 | 同上 | `rejected` | 200场复核full下降、collision上升 |
| corrected edge-1完整Actor | [`10_纯单冲突完整Actor_pilot`](03_保留专门化/02_论文主线/10_纯单冲突完整Actor_pilot/README.md) | `rejected pilot` | 解冻后50场monitor明显退化；只作单Actor对照 |
| 统一停车/靠右/手工TTC规则 | [`results/04_Gate前置验证`](03_保留专门化/02_论文主线/results/04_Gate前置验证/README.md) | `rejected heuristic` | 可作规则基线，不能替代学习Gate |
| 第一版学习Gate | [`07_冲突拓扑组合泛化`](03_保留专门化/02_论文主线/07_冲突拓扑组合泛化/README.md) | `frozen baseline / rejected final` | 方向为正但未过统计和oracle恢复准入 |
| 可部署在线Gate新路线 | [`11_可部署在线Gate研究`](03_保留专门化/02_论文主线/11_可部署在线Gate研究/README.md) | `current / G11-D2 navigation passed, efficiency failed` | B2相对5A导航收益显著，但效率未准入；Actor冻结 |
| 参数匹配单Actor容量对照 | [`12_参数匹配单Actor容量对照`](03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/README.md) | `P1/R1 diagnostic complete / R2-S0 registered` | P1不能归因于参数翻倍；当前只授权随机加宽Actor的n1 broad 100k阶段 |
| 中止和协议错误运行 | [`results/90_中止与无效运行`](03_保留专门化/02_论文主线/results/90_中止与无效运行/README.md) | `invalid` | 不进入论文、趋势判断或模型选择 |

## 可以复用的基础设施

- `TD3/scenario_manifests.py`：固定场景、A*名义路径和冲突图；
- `scripts/build_full_horizon_edge1_view.py`：corrected edge-1视图；
- `TD3/robot_perception/`、`TD3/lidar_cluster_tracking.py`：Gate本机感知前端；
- `TD3/interaction_gate.py`、`TD3/learned_gate_controller.py`：历史Gate基线实现；
- `TD3/test_velodyne_td3_multi.py`：单Actor、oracle和learned Gate统一评测；
- fixed-v1互斥manifest、结果分析与配对统计工具。

复用基础设施不等于恢复其最初对应的方法路线。

## 结果使用规则

1. 先确认split：train、validation、test或partial diagnostic。
2. 先确认场景：0-edge、single-edge、dense validation或其他固定集。
3. 不同manifest的full success不得直接比较。
4. partial结果必须带实际episode数；例如epoch-16全程成绩只有前256场。
5. oracle结果必须明确标注privileged，不得写成Gate结果。
6. `results/90_中止与无效运行`中的数值不得用于任何方法判断。
7. sealed test只允许在模型、特征和阈值全部冻结后读取一次。
