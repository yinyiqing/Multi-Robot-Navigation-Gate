# ICRA Paper Protocol: Preserve-and-Specialize

状态：`D0 方法与实验协议冻结`。

在本协议的 `D1-D3` 完成前，不启动新的 Actor 或 gate 训练。后续若改变方法主张、dense 定义、数据划分或主指标，先修改本协议，再改代码和脚本。

## 1. 一句话主线

在无通信、局部观测的多机器人导航中，完整策略继续适应高交互场景会破坏已有普通导航能力；因此冻结 generalist，用受限 residual 学习 interaction-dense specialist，再用本地时序 gate 按状态调用 specialist。

最终执行策略：

```text
a_t = clip(pi_G(o_t) + g(h_t) * delta_pi_D(h_t), action_bounds)
```

- `pi_G`：冻结的普通导航 generalist，当前候选为 `5D`。
- `delta_pi_D`：dense residual specialist。
- `h_t = o_{t-H+1:t}`：本车最近 `H` 帧观测。
- `g(h_t) in [0, 1]`：本地时序 interaction gate。
- 执行阶段不读取其他机器人真值，不要求通信。
- 训练阶段允许 Critic 使用局部邻居几何，属于 CTDE。

## 2. 论文问题与假设

### RQ1: 什么是 dense

机器人数量或单位面积密度不足以描述导航难度。真正影响协作的是具有时间重叠的路径冲突，即 interaction density。

### RQ2: 为什么不训练一个全能 Actor

历史 full fine-tune、PAIR 和 THREE 实验反复出现能力覆盖与退化。假设普通导航与强交互修正共享基础能力，但具有冲突的更新方向。

### RQ3: 为什么需要 specialist 和 gate

只有部分局部状态需要强交互修正。假设冻结 generalist 并限制 specialist 的动作残差，能够形成互补能力；时序 gate 能在保留普通能力的同时调用该修正。

### 可证伪假设

- `H1`：在相同训练预算和修复后的 Critic 下，residual specialist 的 high-interaction full success 高于 full fine-tune 和 head-only。
- `H2`：paired episodes 中存在足够的 `specialist-only success`，使 oracle union 明显高于任一单专家。
- `H3`：temporal gate 接近 oracle union，并保持 generalist 在 standard benchmark 上的能力。
- `H4`：模型收益随 interaction density 增大，而不是只对五个手工 case 有效。

任何一个假设均允许被实验否定。若 `H2` 不成立，停止 gate 路线。

## 3. Dense 的操作化定义

### 3.1 Spatial density

```text
rho_S = N / A_free
```

其中 `N` 为机器人数量，`A_free` 为可通行自由空间面积。它只描述空间占用，不代表机器人路径必然冲突。

### 3.2 Synchronized nominal conflict graph

对每个机器人，从起点到目标构造恒定名义速度 `v0` 的参考轨迹 `p_i(t)`。第一版使用不穿越静态障碍的最短折线路径；若暂时没有全局路径工具，可使用经过地图可行性检查的直线路径近似。

对机器人对 `(i, j)` 定义：

```text
d_ij = min_t ||p_i(t) - p_j(t)||
t_ij = argmin_t ||p_i(t) - p_j(t)||
```

当以下条件同时成立时，在冲突图中加入边：

```text
d_ij < d_conflict
t_ij < T_horizon
```

初始建议值：

- `v0 = 0.5 m/s`
- `d_conflict = 0.9 m`
- `T_horizon = 8 s`

这些阈值必须在 pilot 中做 `+-20%` 敏感性检查，不能只报告单个阈值。

Interaction density：

```text
rho_I = 2 * |E| / (N * (N - 1))
```

同时保存：

- `conflict_edge_count`
- `max_conflict_degree`
- `mean_conflict_degree`
- `earliest_conflict_time`
- `simultaneous_conflict_count`
- `min_start_clearance`
- `min_goal_clearance`
- `min_path_separation`
- `bottleneck_width / robot_diameter`

### 3.3 五车难度层级

第一版按冲突边数划分，不按策略成功率反向定义：

| Level | 五车冲突边数 | 用途 |
| --- | ---: | --- |
| low | `0-1` | generalist retention / standard |
| medium | `2-4` | specialist 主训练分布 |
| high | `>=5` | specialist 强交互训练与测试 |

完成至少 1000 个生成样本的分布检查后，才能冻结最终区间。

## 4. 场景生成与数据划分

### 4.1 程序化生成约束

每个场景必须满足：

- 起点和目标位于有效自由空间。
- 初始机器人中心距离满足安全约束，不允许 reset 后立即碰撞。
- 目标之间不重叠，目标不贴障碍。
- 每个机器人独立存在可行路径。
- 起终点距离处于固定范围，避免通过缩短任务制造高成功率。
- 按 `rho_I`、最大冲突度和时间重叠接受或拒绝样本。
- 保存完整 manifest，禁止仅保存 `tight1` 之类人工标签。

### 4.2 三类数据

1. `procedural-low`：低 interaction density，用于 generalist retention 和 gate 训练。
2. `procedural-medium/high`：dense specialist 的训练与分布内测试。
3. `canonical-heldout`：crossing、merge、corridor reverse、weaving、bottleneck 等人工 archetype，只用于测试。

现有五个 moderate case 归入 `canonical-heldout`，不再作为唯一训练分布。

### 4.3 严格划分

- 训练、验证、测试使用不重叠的生成 seed。
- canonical-heldout 的几何模板不得出现在训练集。
- 至少保留一张未参与训练的地图。
- 测试扩展到未训练的机器人数量，目标为 `3/5/8`，若 Gazebo 容量不允许 8 车则明确报告限制。

建议初始规模：

| Split | 每个 density level | Seed |
| --- | ---: | --- |
| train | 在线生成 | `0, 1, 2` 分别训练 |
| validation | 200 episodes | `100-299` |
| test | 300 episodes | `1000-1299` |
| canonical-heldout | 每类 60 episodes | 独立固定 seed |

## 5. 方法训练阶段

### S0: Generalist 冻结

- 固定 `5D` Actor，不更新参数。
- 用修复后的互斥指标重测 standard、density sweep 和 held-out cases。
- 旧口径结果只能作为历史诊断。

### S1: Dense specialist

- 初始化 `delta_pi_D` 为零，初始动作与 `5D` 完全一致。
- gate 固定为 `g=1`。
- 只训练 residual 和 Critic，generalist 全冻结。
- 主训练分布为 procedural medium/high。
- 使用 individual reward；第一版不加入 cooperative reward shaping。
- residual action 默认限制为 `0.15`，并对 residual magnitude 加正则。
- best 只根据 validation medium/high 选择，不能看 test。

### S2: 专家互补性审计

在完全相同的 test seeds 上分别运行 generalist 和 specialist，记录 episode 配对结果：

| Generalist | Specialist | 统计名称 |
| --- | --- | --- |
| success | success | both-success |
| success | fail | generalist-only |
| fail | success | specialist-only |
| fail | fail | both-fail |

定义：

```text
oracle_union = both_success + generalist_only + specialist_only
specialist_gain = specialist_only - generalist_only
```

Gate 准入条件：

- high-interaction full success 相对 generalist 至少 `+10 percentage points`。
- `specialist-only` 至少占 paired episodes 的 `10%`。
- oracle union 相对最佳单专家至少 `+8 percentage points`。

未达到任一条件则停止 gate，回到 specialist 的观测、场景生成或优化问题。

### S3: Gate

- 冻结 generalist 和 specialist。
- 在 low/medium/high 混合分布上训练 gate。
- gate 输入只使用本车最近 `H` 帧 24 维观测。
- 第一版比较 `H=1` MLP 与 `H=4/8` GRU 或 temporal attention。
- 输出 soft gate，并加入时间平滑约束，避免动作模式频繁抖动。
- 不使用 case 名称或全局 density label作为执行输入。

最终 gate 准入条件：

- standard full success 相对 generalist 下降不超过 `3 percentage points`。
- high-interaction full success 距 oracle union 不超过 `5 percentage points`。
- gate activation 随 `rho_I` 单调增加，但在 low density 不应长期激活。

## 6. 必须对照与消融

### 主基线

- Frozen `5D` generalist。
- 修复后 Critic 下的 full Actor fine-tune。
- 修复后 Critic 下的 head-only。
- 独立 dense Actor。
- ORCA/RVO 或项目环境中可接入的经典去中心化避碰基线。
- 单一 recurrent/attention policy，回应“为何不训练一个全能策略”。

### 方法消融

- residual always off：generalist。
- residual always on：dense specialist。
- random/fixed residual：排除结构容量本身的作用。
- heuristic gate：按最小激光距离或邻居数量切换。
- single-frame learned gate。
- temporal learned gate。
- oracle episode union：不可执行上界。
- 无 residual bound / 不同 residual scale。
- shared Critic 与 geometry-local Critic。

## 7. 指标与统计

### 主指标

- `full_success_rate`：首要指标。
- `agent_success_rate`
- `collision_rate`
- `unresolved_rate`

三类终止必须互斥，且满足：

```text
success + collision + unresolved = N * episodes
```

### 次指标

- episode steps / completion time
- path length 与 success-weighted path efficiency
- minimum robot distance
- angular action variation / smoothness
- residual magnitude
- gate activation ratio 与切换次数
- 按 `rho_I` 分桶的性能曲线

当前代码尚未完整记录 path length、最小机器人距离和 gate smoothness；在正式实验前补齐。

### 统计要求

- 至少 3 个独立训练 seed。
- 所有模型使用 paired test seeds。
- 报告均值、95% bootstrap confidence interval。
- paired success 使用 McNemar test 或 paired bootstrap。
- 不以单个 best seed 作为论文主结果。

## 8. 论文实验矩阵

| 阶段 | 模型 | low | medium | high | held-out | standard | 目的 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Frozen 5D | yes | yes | yes | yes | yes | 基线曲线 |
| E1 | Full fine-tune | yes | yes | yes | yes | yes | 退化对照 |
| E2 | Head-only | yes | yes | yes | yes | yes | 容量对照 |
| E3 | Residual always-on | yes | yes | yes | yes | yes | specialist |
| E4 | Independent dense Actor | yes | yes | yes | yes | yes | 双完整 Actor 对照 |
| E5 | Heuristic gate | yes | yes | yes | yes | yes | 非学习切换 |
| E6 | Single-frame gate | yes | yes | yes | yes | yes | 时序消融 |
| E7 | Temporal gate | yes | yes | yes | yes | yes | 完整方法 |
| E8 | Oracle union | yes | yes | yes | yes | yes | 上界 |

## 9. 旧实验如何进入论文

| 历史实验 | 新角色 |
| --- | --- |
| A/B/C/D/D2 | reward 与 local Critic 预研，放消融或附录 |
| 5A/5D | generalist 候选与 warm-start 来源 |
| PAIR/THREE | 课程覆盖不能稳定产生 specialist 的证据 |
| full fine-tune 退化 | preserve-and-specialize 的核心动机 |
| head-only | 限制更新范围但表达力不足 |
| 5A + 5D gate/oracle 失败 | 两个模型不等于两个互补专家 |
| random vs fixed dense | spatial density 与 interaction density 的区别 |

旧实验若使用旧终止口径，只用于动机和定性分析；论文主表必须按新口径重跑。

## 10. 决策门

按顺序执行，不允许跳过：

- `D0`：本协议确认。
- `D1`：scenario manifest、冲突图计算和程序化生成器完成并单测。
- `D2`：固定数据划分生成，检查 density 分布和场景有效性。
- `D3`：修复口径下的 generalist baseline 完成。
- `D4`：specialist 训练与三 seed 复现。
- `D5`：互补性达到 gate 准入条件。
- `D6`：gate 训练。
- `D7`：完整基线、消融、泛化和统计检验。
- `D8`：论文图表冻结。

当前只允许推进到 `D1`，不启动 residual 或 gate 训练。

## 11. 预期贡献表述

如果实验支持假设，贡献收敛为三点：

1. 区分 spatial density 与 interaction density，并提出基于同步名义冲突图的程序化评测协议。
2. 提出 preserve-and-specialize residual experts，在不覆盖 generalist 的情况下学习 dense 交互能力。
3. 提出仅依赖本地观测历史的 temporal gate，并在 density sweep、held-out archetypes 和不同机器人数量上验证能力保持与专家调用。

如果 interaction-density 定义没有形成稳定难度曲线，不把它作为独立方法贡献，只作为实验协议；如果 gate 未接近 oracle，不宣称自适应专家选择成功。
