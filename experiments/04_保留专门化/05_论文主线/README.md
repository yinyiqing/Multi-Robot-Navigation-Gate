# ICRA Paper Protocol: Preserve-and-Specialize

状态：`5D 冻结为弱交互 Actor；正式均衡强交互训练已产生候选；等待重复验证与单 Actor 配对审计，尚未进入 Gate`。

后续若改变方法主张、交互强度定义、数据划分或主指标，先修改本协议，再改代码和脚本。

执行进度（2026-07-18）：`generalist-5d` 已完成 fixed-v1 standard 1000 场和 dense 2000 场。standard full success `0.5750`，dense `0.2795`；完整归档见 [D3 fixed-v1 baseline](results/D3_fixed_v1_generalist_baseline/README.md)。

validation 交互分层（2026-07-19）：5D 在 standard low-interaction、standard interaction、dense overall 上的 agent/full success 分别为 `0.9680/0.8544`、`0.8143/0.4252`、`0.7122/0.3140`。dense 中 0-edge 的 full success 为 `0.9524`，edges>0 时为 `0.2860`，确认主要难度来自交互冲突而非空间缩小；见 [D3 validation](results/D3_fixed_v1_generalist_validation/README.md)。

现有 standard、random dense 和五个 fixed moderate case 的参数与结果汇总见 [场景对照](SCENARIO_COMPARISON.md)。

## 1. 一句话主线

在无通信、局部观测的多机器人导航中，单一策略难以同时覆盖弱交互导航与紧迫冲突；因此保留 5D 作为弱交互 Actor，训练一个强交互 Actor，再用本地时序 Gate 按当前交互风险选择两者。

最终执行策略：

```text
a_t = clip((1 - g(h_t)) * pi_W(o_t) + g(h_t) * pi_I(o_t), action_bounds)
```

- `pi_W`：冻结的 5D，作为弱交互 Actor。
- `pi_I`：复制5D Actor/Critic初始化、先通过结构化双车交互课程独立训练的强交互 Actor。
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
- 当前 pilot 保持5D的单帧24维输入、`24 -> 800 -> 600 -> 2` Actor和原 TD3，不使用 CNN、GRU、PointNet 或 residual。
- 第一阶段只使用基本双车冲突：`head_on`、`crossing`、`lane_swap`。每类场景内部随机化距离、偏移、中心、旋转和朝向，而不是反复训练少数手工坐标。
- 固定 train `90` 场、validation `30` 场，每类分别为 `30/10`；两个 split 的 scenario ID 和 seed 互斥，并全部通过策略无关 Gazebo reset 检查。
- Actor 和后续 Gate 都不读取离线 risk、standard 或 dense 标签。
- reward保留`0.8 self + 0.2`距离加权邻居项；强交互 pilot 仅关闭原 reward 中鼓励持续前进的 `+0.5*v` 和低速停滞 `-0.03`，避免把必要让行直接视为坏行为，goal/collision与进度项不变。
- epoch 1在 Actor 解锁前得到同协议冻结5D基线；epoch 2训练完整Actor。validation按三种 topology 分层输出。
- pilot 只有在 overall full success 提升、碰撞下降，且提升不是单一 topology 偶然贡献时才进入三车课程；否则先分析 replay 和分层失败，不增加 epoch 或 seed 盲试。

双车诊断数据与复现信息保留在 [pair interaction curriculum v1](datasets/pair_interaction_curriculum_v1/README.md)，不再作为当前训练入口。

双车 pilot 只改善 head-on，crossing 退化且 lane-swap 未改善，碰撞率从 `0.45` 升至 `0.60`；该模型已拒绝，结果仅用于证明按episode均衡不能保证replay均衡，见 `results/D4_pair_interaction_pilot_s20260724`。后续不继续双车研究，也不重新运行 PAIR→THREE。

五车旧 Stage 1 replay 审计发现，强Actor在无可见邻居状态上的变化反而最大，并在 `<=0.8 m` 危险状态继续增加线速度。根因是此前所谓 specialist 仍控制并更新于整个episode，本质上仍在训练generalist。当前候选改为训练期oracle分工：邻居真值距离 `<=2.0 m` 时由强Actor执行并进入其Actor更新集，其余状态始终由冻结5D执行；Critic仍学习完整五车轨迹。Actor结构、TD3目标和部署输入不变。先评估oracle组合上限，通过后才训练可部署Gate。

oracle-specialist pilot 在同一批 `60 deep + 40 close + 40 margin` validation 上，epoch 1/2 的 overall agent success 为 `0.827/0.834`，collision 为 `0.173/0.166`，但 full success 从 `0.500` 降到 `0.471`。deep/close/margin full success 分别从 `0.233/0.575/0.825` 变为 `0.200/0.575/0.775`，未形成强交互专门化。replay 审计显示候选Actor在所有距离层都学到约 `+0.13--+0.16` 的同向角速度偏置，并在 `<=0.8 m` 状态增加线速度 `+0.090`；Critic对这些明显变化有 `79.38%` 偏好。该Critic偏好只说明Actor与Critic自洽，不能单独证明动作在真实环境中更差；它与危险加速、同向转弯和分层评估一起构成训练信号偏移的警示。oracle 分流已修复“普通状态也更新强Actor”的问题，但当前候选仍未学出deep优势。该配置已停止，不增加epoch或seed重复。完整结果见 `results/D4_interaction_oracle_specialist_pilot_s20260724`。

两轮额外配对评估证明单次 full success 波动较大，因此不再仅依据 `0.500 -> 0.471` 判断。三次合计后，5D/候选的 overall agent success 为 `0.8238/0.8229`，full success 为 `0.4571/0.4452`，基本持平且候选略低；但目标 deep 层 full success 在三次中均下降，合计为 `35/180 -> 27/180`。因此拒绝的是当前epoch 2 checkpoint和未修改训练配置，不是双Actor + Gate主线。

Stage 1 同协议 epoch 1/2 的 overall full success 为 `0.4929/0.3357`；close、deep、margin 分别从 `0.6250/0.2167/0.7750` 降至 `0.2750/0.1333/0.7000`，未通过准入条件。离线审计进一步发现，训练后 Actor 在全部 `40001` 个 replay state 上只增加、不降低线速度；对动作变化超过 `0.05` 的状态，Critic 有 `76.80%` 错误偏好真实性能更差的新动作。因此停止 Stage 2，不用更多 epoch 或 seed 重复当前配置。完整结果见 `results/D4_strong_interaction_curriculum_stage1_s20260723`。

可部署相对运动观测的30场sensor probe显示：危险目标原始点云覆盖率为`97.92%`，说明Velodyne信息本身足够；但二维点簇质心跟踪的最佳precision/recall/FPR权衡未同时达到`0.70/0.80/0.10`准入线。主要误报来自静态环境点簇的质心抖动和错误关联，因此拒绝该具体特征，不接Actor；下一候选应加入三维高度轮廓或其他目标身份一致性。完整结果见 `results/D4_lidar_cluster_motion_probe_s20260724`。

后续三维XYZ shape probe在18个校准scenario和12个独立scenario上审计。8维形状逻辑回归在独立集达到precision/recall/FPR `0.651/0.912/0.307`，仍无法排除大量静态点簇。因此停止继续组合手工形状/速度阈值；若继续相对运动主线，下一候选必须是由仿真privileged CPA/TTC标签监督、部署时仅使用本机连续激光帧的时序编码器。完整结果见 `results/D4_lidar_cluster_shape_probe_s20260724`。

20-bin时序风险编码按scenario划分为18 train / 6 validation / 6 test，在完全相同输入下比较单帧MLP与8帧GRU。两者test precision均约`0.159`，FPR为`0.525/0.486`，GRU未恢复可用的CPA/TTC风险信号。因此拒绝“5D原20-bin输入 + 时序编码”，不接Actor或Gate；下一候选必须保留更高角分辨率，并使用新的scenario-level test。完整结果见 `results/D4_temporal_risk_encoder_20bin_s20260724`。

180-bin候选使用旧30场做24/6 train/validation，新的互斥30场只做最终test。GRU相对单帧有改善，但test precision/recall/FPR仅为`0.217/0.781/0.511`，仍远低于准入线。因此不在该holdout上继续调参，不接Actor或Gate。若继续，必须重新建立更大的互斥数据划分，并事先固定能利用角度局部结构的轻量时空编码器。完整结果见 `results/D4_highres_temporal_risk_encoder_s20260724`。

历史 edge-1 residual v1/v2 将 deep、close、margin 混在一起且只使用单帧观测，因此不视为当前强交互 Actor 的正式训练；结果仅用于说明价值外推和静态观测问题。

pilot 实测 epoch 1/2 full success 为 `0.5130/0.4704`，碰撞率为 `0.1546/0.1721`。更新后的 residual 几乎恒定饱和到 `[+0.10, -0.10]`，同时 Critic Q 上升而真实性能下降；拒绝当前 residual TD3 配置，不启动更多 seed 或 edge 1-2，完整归档见 `results/D4_interaction_edge1_residual_pilot_s20260720`。

后续只允许一个直接针对该机制的 v2：复用 epoch 1 已预热 Critic，将 Actor Q 项按 batch mean absolute Q 归一化，并以权重 `2.5` 约束到冻结 5D 动作；训练一轮 40000 agent samples 后在相同 423 场 validation 上判定。除该 objective 外不扩大网络、不增加交互难度。

v2 full success 为 `0.5177`，仅比现场冻结基线 `0.5130` 多 2 场，同时碰撞增加，且低于历史 5D 的 `0.5248`。Residual 边界饱和已消失，说明价值外推约束有效，但仍未形成可用的状态相关避让行为；停止继续调整 Actor objective，后续先补充相对速度/TTC 等交互观测或独立交互监督信号。完整归档见 `results/D4_interaction_edge1_conservative_residual_v2_s20260720`。

60 场冻结 5D 风险 probe 进一步确认：按同步路径最小间距分层后，deep/close/margin full success 为 `0.15/0.55/0.85`；失败组在进入约 `1.2 m` 接近区时的闭合速度更高、TTC 更短，但 5D 仍保持高线速度。当前单帧 Actor 不显式观测这些动态量，因此暂停 specialist 续训，先运行固定优先级让行 oracle 验证可解上限。完整归档见 `results/D4_interaction_risk_probe_5d_s20260721`。

固定优先级让行 oracle 在全部 edge-1 上将碰撞率从 `0.170` 降到 `0.147`，但 full success 从 `0.517` 降到 `0.450`。分层后，deep full success 从 `0.15` 升到 `0.35`，close/margin 却从 `0.55/0.85` 降到 `0.30/0.70`。因此拒绝“只要有交互就停车”，但保留“仅在紧迫冲突切换到交互专家”的主线。下一阶段先定义可部署的时序闭合速度/TTC 观测，不直接继续训练。完整归档见 `results/D4_interaction_risk_yield_oracle_s20260721`。

基于扇区最小距离的自运动补偿时序差分已在 20-bin 和独立 180-bin 输入上审计。两者 frame recall 为 `0.808/0.909`，但 false-positive rate 高达 `0.636/0.738`，且所有 episode 都被激活。根因是扇区最小值没有稳定物体关联；该特征族已拒绝，不接入 Actor/Gate，不继续调阈值。下一步只做原始二维点的自运动补偿移动簇可行性验证。完整归档见 `results/D4_temporal_interaction_scan_diff_s20260721`。

### S3: Actor 互补性审计

强交互 Actor 训练完成后，先在完全相同的固定 validation scenario ID 上让
5D 与强交互 Actor **分别单独运行**，禁止用 oracle 切换结果代替单 Actor
对照。分层至少包含 `edge=0`、margin、close 和 deep，并记录 episode 配对
结果：

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

- 强交互 Actor 若在弱交互、margin、close 和 deep 上均不差于 5D，则取消
  双 Actor + Gate，直接采用单一强交互 Actor；Gate 只有在两个 Actor 存在
  明确互补时才有方法意义。
- `edge=0` 存在成功率天花板，弱交互对照还必须报告平均完成步数、timeout
  和路径效率，并纳入无路径冲突但任务距离较长或静态障碍更复杂的固定场景。
- 强交互 Actor 在 deep 上相对 5D 至少 `+15 percentage points` full success。
- interaction-only success 至少占强交互 paired episodes 的 `10%`。
- oracle union 相对最佳单专家至少 `+8 percentage points`。

未达到任一条件则停止 gate，回到 expert 的观测、场景生成或优化问题。只有
validation 完成上述判断并冻结模型与协议后，才允许在 test 上运行最终一次
对照。

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

当前仍处于`D4`。5D已冻结为弱交互Actor；旧 close→mixed→deep、PAIR→THREE 和双车路线均已停止。五车 oracle-specialist pilot 也已按预先准入线判定未通过：分流机制正常，但强Actor学到统一转向偏置，deep full success 不升反降。下一步先审计固定场景的左右对称性以及Actor可见状态与Critic privileged context的对齐，不直接启动新训练。Gate仍需等待D5互补性审计通过。

后续完成了5A warm-start下的Critic context严格对照：将旧世界坐标几何context修正为本车坐标系相对位置与相对速度后，epoch 2 full success仍从`0.500`降至`0.421`，deep从`0.200`降至`0.133`。Replay审计显示`<=0.8m`危险状态只占`2.84%`，且训练后Actor在该层增加线速度`+0.128`，Critic对明显变化动作的错误偏好率为`76.77%`。因此context修复保留，但不再增加epoch；下一次严格对照只修改安全reward：在`1.2m`内加入随线速度增长的近车处罚，并奖励“机器人间距增大且自身仍向目标前进”的有效避让结果。完整结果见`results/D4_interaction_ego_motion_from_5a_s20260725`。

安全reward对照仍未解决退化：epoch 2 full success从`0.457`降至`0.429`，deep从`0.150`降至`0.100`。进一步审计确认，fresh Critic在训练前就因随机初始化产生几乎单向的动作梯度；冻结Actor训练一个epoch后，该偏置被放大到危险状态中线速度和角速度正梯度均接近`100%`。同时Critic均匀采样全部Replay，而危险样本仅约`2.6%`；安全reward还曾计入Critic不可见或已结束的邻车。当前修复统一reward与Critic的可见active邻车口径、令Critic batch中`75%`来自交互样本，并在Actor解冻前检查危险样本动作梯度；样本不足或梯度近乎单向时继续冻结Actor，不再让错误Critic直接更新Actor。

上述Critic采样与解冻保护实验中，梯度门连续拒绝`226`次，Actor参数始终与5A完全一致。角速度单向偏置从约`96%`改善到约`50%-63%`，但危险状态线速度正梯度到结束仍为`96%-100%`，说明交互过采样只能修正左右不平衡，不能解决未见动作上的线速度外推。下一次对照保持TD3结构不变，在Critic loss中加入局部反事实排序：仅对`<=1.0m`且闭合速度`>=0.1m/s`的状态，要求减速`0.2m/s`的动作不低于原动作。离线联合TD回放中，ranking权重`5`将危险状态线速度正梯度从`98.1%`降到`88.9%`；权重`10+`出现明显过度保守，因此正式pilot固定为`5`并继续使用解冻门。

Critic反事实排序正式pilot将危险线速度正梯度稳定降到约`20%-35%`，但Actor解冻后full success仍从`0.500`降至`0.464`，deep从`0.217`降至`0.133`。Replay对照显示Actor在`<=1.2m`状态仍增加线速度`+0.012`，并产生全局角速度偏移`+0.093`。根因是Actor仍从全部`<=2.0m`交互样本更新，而Critic安全约束只覆盖近距离接近样本。下一次对照保留Critic修复，只让Actor从`<=1.0m`且闭合速度`>=0.1m/s`的候选池更新，并仅对角速度锚定5A；不约束线速度，以保留学习减速的空间。

安全聚焦Actor pilot首次在同一轮冻结基线对照中全面提升：agent success从`0.820`升至`0.840`，collision从`0.179`降至`0.160`，full success从`0.436`升至`0.500`；deep/close/margin full success分别从`0.150/0.500/0.800`升至`0.217/0.600/0.825`。Replay行为审计确认`<=0.8m`线速度下降`0.079`，危险加速问题已消除；仍存在`-0.055`全局角速度漂移。该epoch 2暂列强交互Actor候选，但考虑固定Gazebo验证仍有波动，下一步先重复140场固定验证，再决定是否运行独立训练seed和进入D5互补性审计。完整归档见`results/D4_interaction_focused_actor_from_5a_s20260725`。

同配置的全套5D对照中，epoch 2相对冻结epoch 1仅将agent/full success从`0.824/0.471`提高到`0.830/0.479`，即full仅多成功`1/140`场；deep/close有所提高，但margin full从`0.875`降至`0.800`。其改善明显弱于全套5A配置的`0.436 -> 0.500`。由于当时启动脚本把强Actor warm-start和oracle弱Actor绑定为同一模型，该结果比较的是“全套5A”与“全套5D”，不是纯初始化消融。下一次正式长跑将二者解耦：强Actor从5A初始化，非强交互状态固定由5D执行。完整归档见`results/D4_interaction_focused_actor_from_5d_s20260725`。

解耦后的`5A强Actor初始化 + 5D弱Actor`对照没有通过：冻结epoch 1的agent/full success为`0.839/0.514`，Actor更新后的epoch 2降至`0.811/0.407`；deep/close full分别从`0.217/0.650`降至`0.100/0.475`。恢复过程、Replay、Critic和epoch计数正常，退化发生在Actor解冻后。该结果说明不能在改变弱Actor轨迹分布的同时直接外推此前全套5A训练结论；停止该checkpoint。接下来回到唯一产生明确正向结果的全套5A实验，从其epoch 2 checkpoint原样续训，再独立评估最佳强Actor与5D的oracle配对。完整归档见`results/D4_interaction_focused_actor_5a_init_5d_weak_s20260726`。

后续续训到旧epoch 8进一步确认了安全聚焦Actor配置能够产生正向候选，但旧640场训练使用随机有放回抽样，实际场景覆盖和deep/close/margin短窗口比例均不受保证，因此旧epoch 1-8只作为配置筛选依据，不作为正式长跑曲线。正式长跑从原始5A重新开始，保持已验证有效的全套5A rollout、TD3、reward、Critic ranking、Actor安全聚焦更新和解冻条件不变；唯一实验变量是改用不含validation/test的2560场`full_train`以及修复后的`balanced_cycle`。上限设为16个固定样本epoch，预计足以完成至少一次全池遍历，并在每20,000 agent samples后继续使用同一140场validation选择best。旧epoch 7 + 新Critic的短暂重热实验在Actor解冻前停止，见`results/D4_aborted_e7_rewarm_balanced_preunlock_s20260726`。

正式均衡长跑已正常完成16个epoch和`320,000` agent samples，覆盖全部`2560/2560`个训练场景。训练期oracle组合的固定validation full success从epoch 1冻结5A的`0.436`提高到epoch 16的`0.707`，deep/close/margin分别从`0.183/0.475/0.775`提高到`0.617/0.675/0.875`，碰撞率从`0.174`降至`0.079`；同时平均步数从`33.1`升至`54.5`，表明策略更慢、更保守。epoch 16暂存为候选，但该validation同时用于选best，且结果包含privileged oracle分工，不能当作强Actor单独成绩或Gate成绩。下一步只做重复固定validation，以及5D与epoch 16强Actor在相同scenario ID上的单独配对评估；完成前不读取test、不训练Gate。完整归档见`results/D4_interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726`。

## 11. 预期贡献表述

如果实验支持假设，贡献收敛为三点：

1. 区分 spatial density 与 interaction density，并提出基于同步名义冲突图的程序化评测协议。
2. 保留可靠的弱交互导航能力，并通过渐进交互课程训练强交互Actor，以配对评估验证互补性。
3. 提出仅依赖本地观测历史的 temporal gate，并在 density sweep、held-out archetypes 和不同机器人数量上验证能力保持与专家调用。

如果 interaction-density 定义没有形成稳定难度曲线，不把它作为独立方法贡献，只作为实验协议；如果 gate 未接近 oracle，不宣称自适应专家选择成功。
