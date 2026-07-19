# ICRA Paper Protocol: Preserve-and-Specialize

状态：`5D 低交互能力冻结；edge-1 interaction residual pilot 已进入执行阶段`。

在本协议的 `D1-D3` 完成前，不启动新的 Actor 或 gate 训练。后续若改变方法主张、dense 定义、数据划分或主指标，先修改本协议，再改代码和脚本。

执行进度（2026-07-18）：`generalist-5d` 已完成 fixed-v1 standard 1000 场和 dense 2000 场。standard full success `0.5750`，dense `0.2795`；完整归档见 [D3 fixed-v1 baseline](results/D3_fixed_v1_generalist_baseline/README.md)。

validation 交互分层（2026-07-19）：5D 在 standard low-interaction、standard interaction、dense overall 上的 agent/full success 分别为 `0.9680/0.8544`、`0.8143/0.4252`、`0.7122/0.3140`。dense 中 0-edge 的 full success 为 `0.9524`，edges>0 时为 `0.2860`，确认主要难度来自交互冲突而非空间缩小；见 [D3 validation](results/D3_fixed_v1_generalist_validation/README.md)。

现有 standard、random dense 和五个 fixed moderate case 的参数与结果汇总见 [场景对照](SCENARIO_COMPARISON.md)。

## 1. 一句话主线

在无通信、局部观测的多机器人导航中，单一策略难以同时覆盖普通与高交互场景；因此从同一个 5D baseline 分别训练 standard expert 和 dense expert，再用本地时序 gate 按状态选择两者。

最终执行策略：

```text
a_t = clip((1 - g(h_t)) * pi_S(o_t) + g(h_t) * pi_D(o_t), action_bounds)
```

- `pi_S`：只用 fixed standard 数据训练的完整 Actor。
- `pi_D`：只用 fixed dense 数据训练的完整 Actor。
- `5D`：两个 expert 的共同初始化和固定 baseline，不作为最终 standard expert。
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

两个场景池需要不同的行为偏好，但执行阶段不能读取场景标签。假设独立训练能形成互补专家，时序 gate 能仅凭本车局部历史选择更合适的 Actor。

### 可证伪假设

- `H1`：standard/dense expert 在各自主场 validation 上都显著高于共同的 5D baseline。
- `H2`：交叉评估呈现对角优势，paired episodes 中存在足够的单专家成功，使 oracle union 明显高于任一专家。
- `H3`：temporal gate 接近 oracle union，同时保持 standard expert 的普通场景能力。
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

### 3.3 指标的用途

冲突边数和 interaction density 只用于描述、分桶和分析结果，不参与场景的接受或拒绝。这样 dense 仍然是随机分布，不会因为人为挑选“难例”而变成另一组特殊 case。

## 4. 场景生成与数据划分

### 4.1 程序化生成约束

每个场景必须满足：

- 起点和目标位于有效自由空间。
- 初始机器人中心距离满足安全约束，不允许 reset 后立即碰撞。
- 目标之间不重叠，目标不贴障碍。
- 每个机器人独立存在可行路径。
- 起终点距离处于固定范围，避免通过缩短任务制造高成功率。
- Gazebo reset 后传感器正常、无初始碰撞，实际位置与 manifest 一致。
- 保存完整 manifest，禁止仅保存 `tight1` 之类人工标签。

筛选只能依据上述策略无关的有效性条件。禁止依据 `5D`、dense Actor 或本文方法的成功/失败删除 test 场景。

### 4.2 两个场景池

只生成两类环境：

1. `standard`：普通五车随机场景，训练/验证普通 Actor，并检查 gate 是否保留普通能力。
2. `dense`：在 tight1 与 tight2 之间连续随机采样的五车场景，训练/验证 dense Actor。

Gate 不需要第三种环境。它直接混合读取 `standard/train` 和 `dense/train`。validation 和 test 只是两个场景池各自互不重叠的数据划分，不是新环境。

Dense 固定参数：起点方形半宽在 `1.65-1.75 m` 连续采样，起点间距至少 `1.2 m`，任务距离 `0.9-2.3 m`，五车、四个随机箱子。越界 goal 直接重采样，不做 clip。

### 4.3 严格划分

- 训练、验证、测试使用不重叠的生成 seed。
- 场景在离线生成和有效性筛选后冻结；训练、验证和测试全过程只按 manifest 回放。
- 训练集随机采样；每次 validation 固定从 manifest 首个场景按顺序回放，保证跨 epoch 可比。
- 先多生成候选，再只删除无效 reset，最后按目标数量截取，不能运行方法后再清理 test。

建议初始规模：

| Pool | train | validation | test |
| --- | ---: | ---: | ---: |
| standard | 3000 | 500 | 1000 |
| dense | 6000 | 1000 | 2000 |

正式数量可根据 Gazebo 筛选成本调整，但 test 必须在训练前冻结，并保留所有有效场景。当前仓库中的 `datasets/pilot` 只用于检查生成与回放管线，不是正式论文数据。

## 5. 方法训练阶段

### S0: 5D baseline 冻结

- 固定 `5D` Actor 作为共同初始化和对照，不把它称为最终 standard expert。
- fixed-v1 standard/dense baseline 已完成。
- validation 分层确认 5D 的低交互导航能力较强，性能下降主要集中在 standard/dense 的冲突子集。
- 旧口径结果只能作为历史诊断。

### S1: Standard expert

- 从 5D Actor-only warm-start，完整 Actor 可更新，Critic 重新初始化。
- 只使用 `standard/train`，best 只根据 `standard/validation` 的 full success 选择。
- 第一轮保持 5D 的网络、reward 和 geometry-local Critic 配置，只改变固定数据与训练目标。
- checkpoint 记录 validation manifest 哈希、场景数和采样协议；协议变化时旧曲线归入 history，不参与新 best 比较。
- 每个 validation epoch 保存独立 Actor/Critic 快照，完整 latest 用于续训，完整 best 只在同一 validation 协议内选择。
- v1 前 6 epoch 只有两轮 Actor 更新，续训到 epoch 12 后确认该配置会发生后期退化。
- v1 actor-only warm-start 在 epoch 10 达到 `0.850` agent success 后退化，结果归档于 `results/D4_standard_expert_actor_only_v1`。
- v2 保持原 `0.8/0.2` cooperative reward，完整加载形状兼容的 5D Actor/Critic，并用弱 Actor anchor 控制策略漂移。
- 训练 transition 将 timeout 视为 terminal；Critic 更新次数按有效 agent samples 归一化到 collective environment steps，避免跨 reset bootstrap 和 timeout 过度更新。
- v3 的 100 场正向信号未在完整 500 场 validation 复现：5D agent/full `0.8776/0.6020`，v3 epoch 2 为 `0.8712/0.5920`，且平均多用 `8.378` 步；拒绝该候选，不读取 test。
- `standard/test` 只在模型和超参数冻结后运行一次。

完整 validation 分层后，5D 在 standard low-interaction 上达到 agent/full `0.9680/0.8544`。因此暂停继续微调 standard Actor，将 5D 冻结为普通专家候选；只有最终 Gate 在完整 standard 上仍无法达到目标时才重新打开该路线。

### S2: Dense expert

- 从同一个 5D Actor-only warm-start，使用与 standard expert 相同的 Actor 结构。
- 只使用 `dense/train`，best 只根据 `dense/validation` 选择。
- 不读取 standard/dense case 标签作为 Actor 输入。
- `dense/test` 只在模型和超参数冻结后运行一次。

首个受控 pilot 不直接覆盖全部 dense：使用 standard/dense 各 256 个 edge-1 train 场景，validation 使用两池全部 423 个 edge-1 场景。5D 在该 validation 上的 full-success 基线约为 `0.525`。5D 主体冻结，residual scale 为 `0.10`；前 `41000` agent samples 只预热 Critic，每 `40000` samples 完整验证一次，共两轮。只有 full success 达到至少 `0.60`、碰撞下降且 timeout 不增加，才扩展到 edge 1-2。

### S3: 专家互补性审计

在完全相同的 test scenario ID 上分别运行两个 expert，记录 episode 配对结果：

| Standard expert | Dense expert | 统计名称 |
| --- | --- | --- |
| success | success | both-success |
| success | fail | standard-only |
| fail | success | dense-only |
| fail | fail | both-fail |

定义：

```text
oracle_union = both_success + standard_only + dense_only
dense_gain = dense_only - standard_only
```

Gate 准入条件：

- 两个 expert 在各自主场相对 5D 至少 `+10 percentage points` full success。
- dense-only success 至少占 dense paired episodes 的 `10%`。
- oracle union 相对最佳单专家至少 `+8 percentage points`。

未达到任一条件则停止 gate，回到 expert 的观测、场景生成或优化问题。

### S4: Gate

- 冻结 standard expert 和 dense expert。
- 混合固定的 `standard/train` 与 `dense/train` 训练 gate。
- gate 输入只使用本车最近 `H` 帧 24 维观测。
- 第一版比较 `H=1` MLP 与 `H=4/8` GRU 或 temporal attention。
- 输出 soft gate，并加入时间平滑约束，避免动作模式频繁抖动。
- 不使用 case 名称或全局 density label作为执行输入。

最终 gate 准入条件：

- standard full success 相对 standard expert 下降不超过 `3 percentage points`。
- high-interaction full success 距 oracle union 不超过 `5 percentage points`。
- gate activation 随 `rho_I` 单调增加，但在 low density 不应长期激活。

## 6. 必须对照与消融

### 主基线

- Frozen `5D` baseline。
- 独立 standard expert。
- 修复后 Critic 下的 full Actor fine-tune。
- 修复后 Critic 下的 head-only。
- 独立 dense Actor。
- ORCA/RVO 或项目环境中可接入的经典去中心化避碰基线。
- 单一 recurrent/attention policy，回应“为何不训练一个全能策略”。

### 方法消融

- standard expert always on。
- dense expert always on。
- random/fixed gate：排除选择机制本身的作用。
- heuristic gate：按最小激光距离或邻居数量切换。
- single-frame learned gate。
- temporal learned gate。
- oracle episode union：不可执行上界。
- 不同 gate 平滑强度和切换滞回。
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
- gate confidence / expert usage ratio
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
- `D3`：修复口径下的 fixed-v1 standard/dense generalist baseline 完成。
- `D4`：specialist 训练与三 seed 复现。
- `D5`：互补性达到 gate 准入条件。
- `D6`：gate 训练。
- `D7`：完整基线、消融、泛化和统计检验。
- `D8`：论文图表冻结。

当前允许进入 `D4` specialist 训练；gate 仍需等待 D5 互补性审计通过。

## 11. 预期贡献表述

如果实验支持假设，贡献收敛为三点：

1. 区分 spatial density 与 interaction density，并提出基于同步名义冲突图的程序化评测协议。
2. 从共同 baseline 独立训练 standard/dense experts，通过交叉评估验证领域专门化与互补性。
3. 提出仅依赖本地观测历史的 temporal gate，并在 density sweep、held-out archetypes 和不同机器人数量上验证能力保持与专家调用。

如果 interaction-density 定义没有形成稳定难度曲线，不把它作为独立方法贡献，只作为实验协议；如果 gate 未接近 oracle，不宣称自适应专家选择成功。
