# 在线策略切换相关工作

检索日期：`2026-08-04`。范围包括 temporal abstraction、导航策略切换、privileged
learning、imitation data aggregation 和 uncertainty-aware routing。这里只保留能直接
改变本项目 Gate 设计或论文主张的工作。

## 1. Options 与时序切换

| 工作 | 核心内容 | 对本项目的约束 |
| --- | --- | --- |
| [Sutton et al., 1999](https://doi.org/10.1016/S0004-3702(99)00052-1) | Options/SMDP，把策略表示为带启动和终止条件的时序技能 | 5A 和 epoch-16 可以视为固定 options；Gate 同时学习选择和终止 |
| [Option-Critic, AAAI 2017](https://doi.org/10.1609/aaai.v31i1.10916) | 端到端学习 option 内策略和 termination | “Actor + 在线 Gate”不是新结构；本项目只冻结 option policy，学习 selector/termination |
| [Harb et al., AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11831) | 给 option 切换加入 deliberation cost | 支持低频决策、最短保持和显式切换成本，而不是逐帧无代价抖动 |

## 2. 导航中的上下文选择

| 工作 | 核心内容 | 对本项目的约束 |
| --- | --- | --- |
| [APPLD, RA-L 2020](https://doi.org/10.1109/LRA.2020.3002217) | 从示范学习不同导航上下文的 planner 参数 | 证明按上下文选择专门参数/策略是成熟路线 |
| [APPLI, ICRA 2021](https://doi.org/10.1109/ICRA48506.2021.9561311) | 从人工 intervention 学习专门参数集，并按置信度调用 | 与“默认 5A + 局部交互专家”最接近；Gate 架构本身不能作为创新点 |
| [APPLR, ICRA 2021](https://doi.org/10.1109/ICRA48506.2021.9561647) | 用 RL 学习上下文相关的 planner 参数选择 | 支持把直接优化结果的 Gate RL 作为监督蒸馏失败后的升级路线 |
| [多控制策略切换, IECON 2022](https://doi.org/10.1109/IECON49645.2022.9968759) | 根据可通行空间在四种导航策略之间切换 | 直接说明多策略导航切换已有先例，简单密度规则只适合作基线 |
| [Normalizing-Flow Switching, IROS 2024](https://doi.org/10.1109/IROS58592.2024.10802676) | 用 normalizing flow 判断意外场景，在学习和规则方法间切换 | OOD/置信度 Gate 值得作为基线，但“异常”不等于 epoch-16 更优 |
| [Hierarchical Planner Tuning, ICRA 2025](https://doi.org/10.1109/ICRA55743.2025.11128541) | 低频参数选择、中频规划、高频控制的分层 RL | 支持 Gate 低频切换、Actor 高频执行的时间尺度分离 |

结论：双策略、上下文 Gate、置信度切换和分层控制都不是空白。论文不能声称首次提出
“双 Actor + Gate”。

## 3. Privileged Teacher 与数据聚合

| 工作 | 核心内容 | 对本项目的启发 |
| --- | --- | --- |
| [DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html) | 在 learner 实际访问的状态上查询专家并聚合数据 | 解决旧 G2-A 只在 5A 轨迹训练的分布偏移；不需要反事实回滚 |
| [SafeDAgger, AAAI 2017](https://doi.org/10.1609/aaai.v31i1.10857) | 学习何时让 learner 或专家接管 | 与 Gate 的“何时 defer 给交互 Actor”形式相近 |
| [EnsembleDAgger, IROS 2019](https://doi.org/10.1109/IROS40897.2019.8968287) | 用 ensemble 方差估计 learner 不确定性并控制接管 | 支持不确定时默认 5A，并把 ensemble 作为安全 guard |
| [Asymmetric Actor-Critic, RSS 2018](https://doi.org/10.15607/RSS.2018.XIV.008) | 训练 Critic 使用特权状态，部署 Actor 只用传感器 | 支持后备 RL Gate 使用全局训练 Critic、局部部署 Gate |
| [Learning by Cheating, CoRL 2019](https://proceedings.mlr.press/v100/chen20a.html) | 将拥有特权环境信息的 teacher 蒸馏到传感器 student | 直接支持用真值距离 Oracle 监督可部署 Gate |
| [RMA, RSS 2021](https://doi.org/10.15607/RSS.2021.XVII.011) | 用观测历史估计训练期可见、部署期隐藏的环境因素 | 支持用 recurrent history 近似当前不可直接观测的交互状态 |

这些工作共同支持当前首选方案：privileged teacher 可以存在，但最终 student 必须只读
部署观测；数据必须覆盖 student 自己造成的状态分布。

## 4. 多机器人导航边界

| 工作 | 核心内容 | 与本项目的差异 |
| --- | --- | --- |
| [CADRL, RA-L 2017](https://doi.org/10.1109/LRA.2017.2651371) | 去中心化、无通信的深度强化学习避碰 | 已证明本机观测驱动多智能体避碰，不支持“首次无通信避碰”的主张 |
| [Single-Robot EPS for Multi-Robot Navigation, ICRA 2022](https://doi.org/10.1109/ICRA46639.2022.9812341) | 将单机器人技能搜索结果注入多机器人策略训练 | 已有简单技能向多机器人训练迁移，但不是局部交互 option 的在线重复调用 |

## 5. 可以成立的创新边界

文献检索后，可能成立的贡献不是 Gate 结构，而是以下组合是否被实验完整证明：

1. 普通导航和局部交互两个冻结角色明确的策略；
2. 用特权局部交互 Oracle 训练、但只靠本机历史部署的 recurrent Gate；
3. Gate 在 learner 自己访问的状态上进行数据聚合，避免单次反事实分叉；
4. 在不读取冲突图和其他机器人状态的前提下，反复调用局部技能；
5. 按冲突边数、最大度和同时冲突数验证组合泛化，并同时报告 0-edge 能力保持。

是否足够创新最终取决于可部署 Gate 的结果。只有 Oracle 上界、没有 learned Gate 的
跨拓扑收益，不能支撑上述主张。
