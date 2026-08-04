# Gate 候选方法与优先级

## 总表

| 优先级 | 方法 | 预计成本 | 主要回答 | 决策 |
| --- | --- | --- | --- | --- |
| P0 | A. 特权 Oracle 时序蒸馏 + 数据聚合 | 低到中 | 能否更完整恢复现有 Oracle 收益 | 立即做 |
| P1 | B. 冻结 Options + recurrent RL Gate | 高 | 距离标签不等于最终收益时，能否直接学结果 | A 到顶后再做 |
| P2 | C. 共享 Option-Value 离线 Gate | 中 | 能否从已有混合轨迹估计两种 option 的长期价值 | 只做小型可行性检查 |
| Baseline | D. OOD/不确定性 Gate | 低 | 异常或低置信状态是否适合交给专家 | 必须比较，不作主方法 |
| Baseline | E. TTC/CPA + 滞回规则 | 低 | 学习 Gate 是否超过可部署手工风险规则 | 必须比较 |

## A. 特权 Oracle 时序蒸馏 + 数据聚合

### 定义

教师仍是现有有效契约：训练时根据其他机器人真值距离生成 Gate 标签。Student 输入：

- 当前24维 Actor 状态；
- G0/G1 的候选形状分数、相对速度、closing speed、CPA、TTC 和轨迹年龄；
- 5A 与 epoch-16 对当前观测输出的两个候选动作及动作差；
- 上一步 Gate mode、mode 持续时间；
- 最近若干步的 recurrent hidden state。

网络优先用小型 GRU 或 TCN，不引入大 Transformer。部署输出仍是二值 mode 概率，
再经过 switch-on/off 双阈值和最短保持时间。

### 为什么值得先做

- 不修改 Actor；
- 复用现有 shard、detector、tracker 和 controller；
- 不需要恢复 Gazebo 锚点；
- 直接修复旧 G2-A 的轨迹分布、时序和 Actor 分歧缺失；
- 即使失败，也能快速区分“监督流程问题”和“本机不可观测上限”。

### 与旧 GRU 的区别

旧实验只在很小数据上用 `20-bin` 或逐帧高分辨率 scan 预测风险，输入已丢失目标身份
结构，且不包含两个 Actor 的候选动作。新方法在 G0/G1 目标级连续特征上学习完整 Gate
序列，并使用 student-rollout 聚合，不是重跑已拒绝表示。

### 最小实验

1. 在已有 shard 上重现旧 MLP 指标，确认数据读取和 split 完全一致。
2. 只增加 Actor 候选动作，做静态消融。
3. 增加 GRU/TCN 和前一 mode，做时序消融。
4. 分别训练 360 度与前方标签，报告两者，不凭 frame F1 偷换执行契约。
5. 只有 validation 明确改善才进入一轮 student rollout 聚合。
6. 通过离线准入后运行固定50场，再冻结参数运行固定200场。

### 风险

- 360 度标签可能含本机历史也无法推断的信息；
- Oracle 的距离规则不一定等价于 epoch-16 的真实优势；
- G0/G1 tracker 的候选排序最多保留4条轨迹，复杂冲突可能丢失关键对象。

对应诊断：比较前方/全向 teacher 上限，统计 top-k proposal 覆盖率，并检查“Oracle
成功但 student 失败”是否主要来自未观测目标或错误退出时机。

## B. 冻结 Options + Recurrent RL Gate

### 定义

把 5A 和 epoch-16 作为两个不可更新的 options。共享 Gate policy 对每辆机器人输出
option，连续执行若干步后才能重新选择。Gate actor 只读取本机历史；训练 Critic 可以
读取全局机器人状态、碰撞和团队完成情况，部署时丢弃 Critic。

训练目标直接包含：到达、碰撞、timeout、目标进展和切换成本。使用成熟的 recurrent
PPO/离散 SMDP 实现，不手写新的 RL 核心算法。

### 优点

- 直接优化最终导航结果，不依赖“2 m 等于 Actor I 更优”的假设；
- centralized privileged critic 能在训练期处理团队信用分配；
- option duration 和切换成本自然约束抖动。

### 风险与启动条件

- Gazebo 样本吞吐低，在线 RL 可能超过两周期限；
- 奖励容易重新引入碰撞和 timeout 之间的权衡退化；
- 五车共享环境中的 Gate 决策彼此影响，训练方差高。

仅当方法 A 已证明感知和时序足够、但 Oracle 标签与结果收益不一致时启动。先用固定
小场景池验证 Gate reward 确实学习，禁止直接长跑。

## C. 共享 Option-Value 离线 Gate

### 定义

收集 5A、epoch-16、Oracle、旧 Gate 和探索 Gate 的完整轨迹，训练同一个 recurrent
`Q(h_t, option)` 估计固定执行窗口的 n-step return。部署时选择较高 option value，
并减去切换成本。

### 为什么不同于旧 Critic 路线

旧路线比较不同训练过程产生的 TD3 Critic，尺度和分布不可比。本方法要求：

- 一个共享 evaluator；
- 同一回报定义和同一混合数据集；
- option 级时间窗口，而不是单步连续动作值；
- conservative/offline regularization，低支持区域默认 5A。

### 判定

它不需要 Gazebo 状态分叉，但仍有离线外推误差。只做 held-out trajectory ranking 和
小规模闭环 pilot；如果 value calibration 不稳定，立即停止，不用它生成伪标签。

## D. OOD/不确定性 Gate

在 5A 正常成功轨迹上训练 normalizing flow 或轻量 ensemble。低似然、高不确定状态
触发 epoch-16，高置信正常状态保留 5A。

该方法实现简单，也有 IROS 2024 的直接导航先例，适合作为基线或方法 A 的 guard。
但“状态异常”不等于“epoch-16 更优”，因此不能单独作为最终 Gate 逻辑。

## E. TTC/CPA + 滞回规则

使用 G1 连续特征构建一个预先冻结的规则，例如机器人形状软分数、closing speed、
CPA distance 和 TTC 的联合条件，再加双阈值和最短保持时间。

历史手工特征已经证明不能作为最终方法，但该规则仍是必要下界：新 learned Gate 必须
证明自己学到的不只是“前方有快速接近物体”。阈值只能在小 validation 上冻结。

## 推荐顺序

1. 完成 A 的已有数据离线 pilot；这是当前最高信息增益、最低成本步骤。
2. A 离线通过后做一轮 student-rollout 聚合和固定50/200场闭环准入。
3. 同时补 D/E 两个低成本基线。
4. A 若达到准入，停止方法扩张，进入完整 validation 和论文实验。
5. A 若分类已到顶但闭环未过，才在 B 与收缩论文主张之间做一次明确决策。
6. C 只在已有混合轨迹覆盖充分时尝试，不进入默认排期。
