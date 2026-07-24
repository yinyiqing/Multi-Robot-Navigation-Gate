# ICRA Paper Protocol: Preserve-and-Specialize

状态：`5D 冻结为弱交互 Actor；强交互课程 Stage 1 已拒绝；20-bin时序风险编码已拒绝`。

后续若改变方法主张、交互强度定义、数据划分或主指标，先修改本协议，再改代码和脚本。

执行进度（2026-07-18）：`generalist-5d` 已完成 fixed-v1 standard 1000 场和 dense 2000 场。standard full success `0.5750`，dense `0.2795`；完整归档见 [D3 fixed-v1 baseline](results/D3_fixed_v1_generalist_baseline/README.md)。

validation 交互分层（2026-07-19）：5D 在 standard low-interaction、standard interaction、dense overall 上的 agent/full success 分别为 `0.9680/0.8544`、`0.8143/0.4252`、`0.7122/0.3140`。dense 中 0-edge 的 full success 为 `0.9524`，edges>0 时为 `0.2860`，确认主要难度来自交互冲突而非空间缩小；见 [D3 validation](results/D3_fixed_v1_generalist_validation/README.md)。

现有 standard、random dense 和五个 fixed moderate case 的参数与结果汇总见 [场景对照](SCENARIO_COMPARISON.md)。

## 1. 一句话主线

在无通信、局部观测的多机器人导航中，单一策略难以同时覆盖弱交互导航与紧迫冲突；因此保留 5D 作为弱交互 Actor，训练一个强交互 Actor，再用本地时序 Gate 按当前交互风险选择两者。

最终执行策略：

```text
a_t = clip((1 - g(h_t)) * pi_W(o_t) + g(h_t) * pi_I(h_t), action_bounds)
```

- `pi_W`：冻结的 5D，作为弱交互 Actor。
- `pi_I`：复制5D Actor/Critic初始化、通过close→deep课程独立训练的强交互 Actor。
- `standard/dense`：两种场景生成分布，不再表示两个 Actor 的身份。
- `h_t = o_{t-H+1:t}`：本车最近 `H` 帧观测。
- `g(h_t) in [0, 1]`：本地时序 interaction gate。
- 执行阶段不读取其他机器人真值，不要求通信。
- 训练阶段允许 Critic 使用局部邻居几何，属于 CTDE。

## 2. 论文问题与假设

### RQ1: 什么是 dense

机器人数量或单位面积密度不足以描述导航难度。真正影响协作的是具有时间重叠的路径冲突，即 interaction density。

### RQ2: 为什么保留弱交互 Actor

5D 在无冲突 standard/dense validation 上的 full success 为 `0.8544/0.9524`，但在有冲突子集降至 `0.4252/0.2860`。历史 full fine-tune 又反复出现能力覆盖与退化，因此冻结已经可靠的弱交互能力，只单独学习强交互修正。

### RQ3: 为什么需要 specialist 和 gate

弱交互与紧迫冲突需要不同的行为偏好，但执行阶段不能读取离线风险标签。假设渐进交互课程能形成与5D互补的强交互Actor，Gate最终仅凭本车局部历史在状态级选择更合适的Actor。

### 可证伪假设

- `H1`：强交互 Actor 在 deep validation 上显著高于 5D，同时不明显损害 close/margin。
- `H2`：paired episodes 中存在足够的单专家成功，使 oracle union 明显高于任一 Actor。
- `H3`：temporal Gate 接近 oracle union，同时保持 5D 的弱交互能力。
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

- 固定 `5D` Actor 作为弱交互 Actor、强交互 Actor 的基础策略和论文基线。
- fixed-v1 standard/dense baseline 已完成。
- validation 分层确认 5D 的低交互导航能力较强，性能下降主要集中在 standard/dense 的冲突子集。
- 旧口径结果只能作为历史诊断。

### S1: 弱交互 Actor

- 直接冻结 `generalist-5d`，不再训练独立的 standard expert。
- standard/dense 的无冲突子集用于确认弱交互能力，不作为运行时 Gate 标签。
- checkpoint 记录 validation manifest 哈希、场景数和采样协议；协议变化时旧曲线归入 history，不参与新 best 比较。
- 每个 validation epoch 保存独立 Actor/Critic 快照，完整 latest 用于续训，完整 best 只在同一 validation 协议内选择。
- v1 前 6 epoch 只有两轮 Actor 更新，续训到 epoch 12 后确认该配置会发生后期退化。
- v1 actor-only warm-start 在 epoch 10 达到 `0.850` agent success 后退化，结果归档于 `results/D4_standard_expert_actor_only_v1`。
- v2 保持原 `0.8/0.2` cooperative reward，完整加载形状兼容的 5D Actor/Critic，并用弱 Actor anchor 控制策略漂移。
- 训练 transition 将 timeout 视为 terminal；Critic 更新次数按有效 agent samples 归一化到 collective environment steps，避免跨 reset bootstrap 和 timeout 过度更新。
- v3 的 100 场正向信号未在完整 500 场 validation 复现：5D agent/full `0.8776/0.6020`，v3 epoch 2 为 `0.8712/0.5920`，且平均多用 `8.378` 步；拒绝该候选，不读取 test。
- `standard/test` 只在模型和超参数冻结后运行一次。

完整 validation 分层后，5D 在 standard low-interaction 上达到 agent/full `0.9680/0.8544`，在 dense 0-edge 上 full success `0.9524`。因此 5D 已冻结为弱交互 Actor，旧 standard expert v1-v3 只保留为“直接微调全能策略会退化”的失败对照。

### S2: 强交互 Actor

- 保留原始5D作为独立弱交互Actor；强交互Actor完整复制5D Actor/Critic warm-start，复制后的Actor全部参数参与训练。
- 第一轮保持5D的单帧24维输入、`24 -> 800 -> 600 -> 2` Actor和`0.8/0.2` reward，只改变课程数据。
- Stage 1为`256 close + 128 margin`；Stage 2为`256 deep + 256 close + 128 margin`；Stage 3为`512 deep + 128 close + 128 margin`。
- 三阶段固定使用`60 deep + 40 close + 40 margin` validation并输出分层指标。
- Actor 和后续 Gate 都不读取离线 risk、standard 或 dense 标签。
- reward严格保留`0.8 self + 0.2`距离加权邻居项，不增加时序结构或额外reward，避免同时改变多个变量。
- Stage 1只有close提升且margin保持才进入Stage 2；最终Stage 3要求deep full success相对同协议5D至少`+15 points`且碰撞下降，close/margin下降不超过`5/3 points`。

Stage 1 同协议 epoch 1/2 的 overall full success 为 `0.4929/0.3357`；close、deep、margin 分别从 `0.6250/0.2167/0.7750` 降至 `0.2750/0.1333/0.7000`，未通过准入条件。离线审计进一步发现，训练后 Actor 在全部 `40001` 个 replay state 上只增加、不降低线速度；对动作变化超过 `0.05` 的状态，Critic 有 `76.80%` 错误偏好真实性能更差的新动作。因此停止 Stage 2，不用更多 epoch 或 seed 重复当前配置。完整结果见 `results/D4_strong_interaction_curriculum_stage1_s20260723`。

可部署相对运动观测的30场sensor probe显示：危险目标原始点云覆盖率为`97.92%`，说明Velodyne信息本身足够；但二维点簇质心跟踪的最佳precision/recall/FPR权衡未同时达到`0.70/0.80/0.10`准入线。主要误报来自静态环境点簇的质心抖动和错误关联，因此拒绝该具体特征，不接Actor；下一候选应加入三维高度轮廓或其他目标身份一致性。完整结果见 `results/D4_lidar_cluster_motion_probe_s20260724`。

后续三维XYZ shape probe在18个校准scenario和12个独立scenario上审计。8维形状逻辑回归在独立集达到precision/recall/FPR `0.651/0.912/0.307`，仍无法排除大量静态点簇。因此停止继续组合手工形状/速度阈值；若继续相对运动主线，下一候选必须是由仿真privileged CPA/TTC标签监督、部署时仅使用本机连续激光帧的时序编码器。完整结果见 `results/D4_lidar_cluster_shape_probe_s20260724`。

20-bin时序风险编码按scenario划分为18 train / 6 validation / 6 test，在完全相同输入下比较单帧MLP与8帧GRU。两者test precision均约`0.159`，FPR为`0.525/0.486`，GRU未恢复可用的CPA/TTC风险信号。因此拒绝“5D原20-bin输入 + 时序编码”，不接Actor或Gate；下一候选必须保留更高角分辨率，并使用新的scenario-level test。完整结果见 `results/D4_temporal_risk_encoder_20bin_s20260724`。

历史 edge-1 residual v1/v2 将 deep、close、margin 混在一起且只使用单帧观测，因此不视为当前强交互 Actor 的正式训练；结果仅用于说明价值外推和静态观测问题。

pilot 实测 epoch 1/2 full success 为 `0.5130/0.4704`，碰撞率为 `0.1546/0.1721`。更新后的 residual 几乎恒定饱和到 `[+0.10, -0.10]`，同时 Critic Q 上升而真实性能下降；拒绝当前 residual TD3 配置，不启动更多 seed 或 edge 1-2，完整归档见 `results/D4_interaction_edge1_residual_pilot_s20260720`。

后续只允许一个直接针对该机制的 v2：复用 epoch 1 已预热 Critic，将 Actor Q 项按 batch mean absolute Q 归一化，并以权重 `2.5` 约束到冻结 5D 动作；训练一轮 40000 agent samples 后在相同 423 场 validation 上判定。除该 objective 外不扩大网络、不增加交互难度。

v2 full success 为 `0.5177`，仅比现场冻结基线 `0.5130` 多 2 场，同时碰撞增加，且低于历史 5D 的 `0.5248`。Residual 边界饱和已消失，说明价值外推约束有效，但仍未形成可用的状态相关避让行为；停止继续调整 Actor objective，后续先补充相对速度/TTC 等交互观测或独立交互监督信号。完整归档见 `results/D4_interaction_edge1_conservative_residual_v2_s20260720`。

60 场冻结 5D 风险 probe 进一步确认：按同步路径最小间距分层后，deep/close/margin full success 为 `0.15/0.55/0.85`；失败组在进入约 `1.2 m` 接近区时的闭合速度更高、TTC 更短，但 5D 仍保持高线速度。当前单帧 Actor 不显式观测这些动态量，因此暂停 specialist 续训，先运行固定优先级让行 oracle 验证可解上限。完整归档见 `results/D4_interaction_risk_probe_5d_s20260721`。

固定优先级让行 oracle 在全部 edge-1 上将碰撞率从 `0.170` 降到 `0.147`，但 full success 从 `0.517` 降到 `0.450`。分层后，deep full success 从 `0.15` 升到 `0.35`，close/margin 却从 `0.55/0.85` 降到 `0.30/0.70`。因此拒绝“只要有交互就停车”，但保留“仅在紧迫冲突切换到交互专家”的主线。下一阶段先定义可部署的时序闭合速度/TTC 观测，不直接继续训练。完整归档见 `results/D4_interaction_risk_yield_oracle_s20260721`。

基于扇区最小距离的自运动补偿时序差分已在 20-bin 和独立 180-bin 输入上审计。两者 frame recall 为 `0.808/0.909`，但 false-positive rate 高达 `0.636/0.738`，且所有 episode 都被激活。根因是扇区最小值没有稳定物体关联；该特征族已拒绝，不接入 Actor/Gate，不继续调阈值。下一步只做原始二维点的自运动补偿移动簇可行性验证。完整归档见 `results/D4_temporal_interaction_scan_diff_s20260721`。

### S3: Actor 互补性审计

在完全相同的 test scenario ID 上分别运行两个 expert，记录 episode 配对结果：

| 5D 弱交互 Actor | 强交互 Actor | 统计名称 |
| --- | --- | --- |
| success | success | both-success |
| success | fail | weak-only |
| fail | success | interaction-only |
| fail | fail | both-fail |

定义：

```text
oracle_union = both_success + weak_only + interaction_only
interaction_gain = interaction_only - weak_only
```

Gate 准入条件：

- 强交互 Actor 在 deep 上相对 5D 至少 `+15 percentage points` full success。
- interaction-only success 至少占强交互 paired episodes 的 `10%`。
- oracle union 相对最佳单专家至少 `+8 percentage points`。

未达到任一条件则停止 gate，回到 expert 的观测、场景生成或优化问题。

### S4: Gate

- 冻结 5D 和通过准入条件的强交互 Actor。
- 混合 strong/weak 状态轨迹训练 Gate；standard/dense 只用于覆盖不同空间分布。
- gate 输入只使用本车最近 `H` 帧 24 维观测。
- 第一版比较 `H=1` MLP 与 `H=4/8` GRU 或 temporal attention。
- 输出 soft gate，并加入时间平滑约束，避免动作模式频繁抖动。
- 不使用 case 名称或全局 density label作为执行输入。

最终 gate 准入条件：

- 弱交互 full success 相对 5D 下降不超过 `3 percentage points`。
- high-interaction full success 距 oracle union 不超过 `5 percentage points`。
- gate activation 随 `rho_I` 单调增加，但在 low density 不应长期激活。

## 6. 必须对照与消融

### 主基线

- Frozen `5D` baseline。
- 冻结 5D 弱交互 Actor。
- 修复后 Critic 下的 full Actor fine-tune。
- 修复后 Critic 下的 head-only。
- 强交互课程 Actor。
- ORCA/RVO 或项目环境中可接入的经典去中心化避碰基线。
- 单一 recurrent/attention policy，回应“为何不训练一个全能策略”。

### 方法消融

- 5D always on。
- 强交互 Actor always on。
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
| E4 | Temporal interaction Actor | yes | yes | yes | yes | yes | 强交互 Actor |
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

当前仍处于`D4`。5D已冻结为弱交互Actor；close→mixed→deep课程已经固定。当前只执行Stage 1：前20000 agent samples锁定Actor得到同协议5D基线，随后训练完整Actor到40000 samples。只有close提升且margin保持才进入Stage 2；Gate仍需等待D5互补性审计通过。

## 11. 预期贡献表述

如果实验支持假设，贡献收敛为三点：

1. 区分 spatial density 与 interaction density，并提出基于同步名义冲突图的程序化评测协议。
2. 保留可靠的弱交互导航能力，并通过渐进交互课程训练强交互Actor，以配对评估验证互补性。
3. 提出仅依赖本地观测历史的 temporal gate，并在 density sweep、held-out archetypes 和不同机器人数量上验证能力保持与专家调用。

如果 interaction-density 定义没有形成稳定难度曲线，不把它作为独立方法贡献，只作为实验协议；如果 gate 未接近 oracle，不宣称自适应专家选择成功。
