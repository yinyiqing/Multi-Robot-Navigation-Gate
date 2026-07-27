# ICRA Paper Protocol: Preserve-and-Specialize

状态：`冻结5A普通导航Actor和epoch-16条件交互Actor；G0/G1 pilot完成，下一步用privileged交互标签训练只读本机感知的G2 Gate`。

后续若改变方法主张、交互强度定义、数据划分或主指标，先修改本协议，再改代码和脚本。

目录入口：[固定数据](datasets/README.md) | [结果分类](results/README.md) | [场景对照](SCENARIO_COMPARISON.md)

## 0. 当前决策快照

当前方法不是“standard场景使用一个完整Actor、dense场景使用另一个完整Actor”。强弱交互会在同一条轨迹中变化，当前方法采用状态级分工：

| 组件 | 固定角色 | 当前状态 |
| --- | --- | --- |
| `generalist-5a` | 普通导航、目标推进和静态避障 | 冻结 |
| `strong-interaction-5a-balanced` | 紧迫机器人交互中的减速与避让 | 冻结；epoch 16 |
| `robot perception` | 从本机激光中区分机器人与静态障碍，并估计相对运动 | 当前工作 |
| `interaction-gate` | 根据可部署交互证据选择两个冻结Actor | 感知通过后训练 |

此前工作的最终结论：

| 工作 | 结果 | 决策 |
| --- | --- | --- |
| fixed-v1场景、冲突图和互斥数据划分 | 完成 | 继续使用；test保持未读 |
| 5A与5D弱交互对照 | full success `0.8750/0.8710`，无显著差异 | 选择与交互Actor训练分布一致的5A |
| standard/full/head-only/residual微调 | 没有稳定超过冻结模型，多次出现退化 | 不再训练独立standard/dense完整Actor |
| 条件交互Actor正式训练 | oracle组合full success `0.421 -> 0.700`，重复验证复现 | 冻结epoch 16候选 |
| 条件交互Actor全程运行 | 不符合训练契约，出现连续timeout | 只作为局部条件策略调用 |
| 可部署交互感知探针 | 原始点云覆盖充分，但现有分类和时序表示误报过高 | 从机器人/静态障碍区分重新开始 |

两个Actor从本决策开始不再更新。独立Actor训练seed仍是论文最终统计要求，但不在Gate可行性确认前继续消耗训练时间。

执行进度（2026-07-18）：`generalist-5d` 已完成 fixed-v1 standard 1000 场和 dense 2000 场。standard full success `0.5750`，dense `0.2795`；完整归档见 [D3 fixed-v1 baseline](results/01_基线评估/D3_fixed_v1_generalist_baseline/README.md)。

validation 交互分层（2026-07-19）：5D 在 standard low-interaction、standard interaction、dense overall 上的 agent/full success 分别为 `0.9680/0.8544`、`0.8143/0.4252`、`0.7122/0.3140`。dense 中 0-edge 的 full success 为 `0.9524`，edges>0 时为 `0.2860`，确认主要难度来自交互冲突而非空间缩小；见 [D3 validation](results/01_基线评估/D3_fixed_v1_generalist_validation/README.md)。

现有 standard、random dense 和五个 fixed moderate case 的参数与结果汇总见 [场景对照](SCENARIO_COMPARISON.md)。

## 1. 一句话主线

在无通信、局部观测的多机器人导航中，冻结可靠的普通导航Actor和已经学会局部避让的条件交互Actor，再通过机器人感知驱动的本地Gate进行状态级选择，避免完整策略微调造成能力覆盖。

最终执行策略：

```text
a_t = pi_I(o_t),  if g(z_t) = 1
      pi_N(o_t),  otherwise
```

- `pi_N`：冻结的5A，作为普通导航Actor。
- `pi_I`：从5A Actor初始化、使用新Critic和安全聚焦TD3更新训练的条件交互Actor。
- `standard/dense`：两种场景生成分布，不再表示两个 Actor 的身份。
- `z_t`：本机激光产生的机器人检测、距离、方位和相对运动证据，可附加Actor原24维观测。
- `g(z_t) in {0, 1}`：状态级interaction gate；第一版采用硬选择和滞回，soft blend只作为后续消融。
- 执行阶段不读取其他机器人真值，不要求通信。
- 训练阶段允许 Critic 使用局部邻居几何，属于 CTDE。

## 2. 论文问题与假设

### RQ1: 什么是 dense

机器人数量或单位面积密度不足以描述导航难度。真正影响协作的是具有时间重叠的路径冲突，即 interaction density。

### RQ2: 为什么保留普通导航 Actor

5A与5D在同一248个无冲突validation场景上的full success为`0.8750/0.8710`，逐场5A-only/5D-only为`12/11`，没有显著差异。选择5A不是因为它显著更强，而是它已与条件交互Actor共同训练并完成匹配验证，可避免额外的5D到5A轨迹分布切换。历史full fine-tune又反复出现能力覆盖与退化，因此冻结可靠的弱交互能力，只单独学习交互修正。

### RQ3: 为什么需要条件交互 Actor 和 Gate

普通导航与紧迫机器人冲突需要不同的行为偏好，但执行阶段不能读取其他机器人真值。条件交互Actor已经在oracle调用下显示净收益；Gate的任务是依据可部署的机器人身份和交互风险证据选择更合适的Actor，而不是简单模仿`2.0 m`真值阈值。

### 可证伪假设

- `H1`：加入条件交互Actor的匹配组合在deep validation上显著高于冻结5A，同时不明显损害close/margin。
- `H2`：匹配配对中combination-only显著多于weak-only，证明调用交互Actor具有净收益。
- `H3`：本机形状与相对运动证据能够识别需要切换Actor的机器人交互，同时在静态障碍附近保持低误激活率。
- `H4`：可部署Gate显著超过调优后的最小激光距离Gate，并接近固定oracle组合，同时保持5A的普通导航能力。
- `H5`：模型收益随 interaction density 增大，而不是只对五个手工 case 有效。

任何一个假设均允许被实验否定。若`H2`不成立，停止双Actor路线；若`H3`不成立，停止Gate路线。H3不要求先得到完美的机器人语义硬标签，而以最终交互召回和静态场景误激活判断。

### 与已有工作的边界

状态级控制器切换本身不是本文创新。[HNRN（ROBIO 2018）](https://doi.org/10.1109/ROBIO.2018.8664803)已经在目标推进与DRL避碰之间切换；[Fan等（IJRR 2020）](https://doi.org/10.1177/0278364920916531)使用距离规则在PID、RL避碰和保守策略之间切换；[All-in-One（ICRA 2022）](https://doi.org/10.1109/ICRA46639.2022.9811797)进一步训练DRL控制器选择器。因此本文只有在以下组合被实验证实时才有方法价值：保留既有多机器人导航能力、只对稀有交互状态训练TD3条件策略、仅用本机机器人感知学习Gate，并在固定随机交互分层上显著超过可部署距离规则。

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

筛选只能依据上述策略无关的有效性条件。禁止依据`5A`、`5D`、条件交互Actor、Gate或本文方法的成功/失败删除test场景。

### 4.2 两个场景池

只生成两类环境：

1. `standard`：普通五车随机空间分布，用于普通导航能力、静态障碍误激活和Gate能力保持评估。
2. `dense`：在tight1与tight2之间连续随机采样的五车空间分布，用于覆盖更小空间中的弱/强交互状态。

两个场景池都不对应某个Actor身份。Gate不需要第三种环境，直接混合读取`standard/train`和`dense/train`；validation和test只是各自互不重叠的数据划分，不是新环境。

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

### S0: Generalist baseline

- 固定5A和5D作为generalist对照；5D继续保留为论文历史基线。
- fixed-v1 standard/dense baseline 已完成。
- validation 分层确认 5D 的低交互导航能力较强，性能下降主要集中在 standard/dense 的冲突子集。
- 旧口径结果只能作为历史诊断。

### S1: 普通导航 Actor

- 直接冻结`generalist-5a`，不再训练独立的standard expert。
- standard/dense 的无冲突子集用于确认弱交互能力，不作为运行时 Gate 标签。
- checkpoint 记录 validation manifest 哈希、场景数和采样协议；协议变化时旧曲线归入 history，不参与新 best 比较。
- 每个 validation epoch 保存独立 Actor/Critic 快照，完整 latest 用于续训，完整 best 只在同一 validation 协议内选择。
- v1 前 6 epoch 只有两轮 Actor 更新，续训到 epoch 12 后确认该配置会发生后期退化。
- v1 actor-only warm-start 在 epoch 10 达到 `0.850` agent success 后退化，结果归档于 `results/02_普通Actor_失败对照/D4_standard_expert_actor_only_v1`。
- v2 保持原 `0.8/0.2` cooperative reward，完整加载形状兼容的 5D Actor/Critic，并用弱 Actor anchor 控制策略漂移。
- 训练 transition 将 timeout 视为 terminal；Critic 更新次数按有效 agent samples 归一化到 collective environment steps，避免跨 reset bootstrap 和 timeout 过度更新。
- v3 的 100 场正向信号未在完整 500 场 validation 复现：5D agent/full `0.8776/0.6020`，v3 epoch 2 为 `0.8712/0.5920`，且平均多用 `8.378` 步；拒绝该候选，不读取 test。
- `standard/test` 只在模型和超参数冻结后运行一次。

完整弱交互对照中，5A/5D在248个固定0-edge场景上的agent success为`0.9726/0.9718`，full success为`0.8750/0.8710`；两者逐场无显著差异。最终选择5A作为普通导航Actor，以匹配已验证的交互Actor训练分布；完整结果见`results/05_当前冻结方案/D4_weak_actor_5a_vs_5d_s20260727`。旧standard expert v1-v3只保留为“直接微调全能策略会退化”的失败对照。

### S2: 条件交互 Actor

- 正式候选从5A Actor warm-start并使用新Critic；Actor仍是原TD3的单帧24维`24 -> 800 -> 600 -> 2`网络。
- 训练轨迹中，冻结5A负责普通状态；其他机器人真值距离`<=2.0 m`时才由条件交互Actor执行。
- Actor只从安全聚焦交互样本更新；Critic使用本车坐标系的邻车相对位置和相对速度，并保留反事实减速排序约束。
- reward保留`0.8 self + 0.2`距离加权邻居项，不改成纯个体reward。
- 正式`full_train`包含2560个五车强交互场景，按deep/close/margin均衡循环采样，16个epoch完整覆盖。
- 当前冻结候选是`interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth`。
- Actor和最终Gate均不得读取离线risk、standard/dense标签或其他机器人真值。

#### 已拒绝的路线与机制诊断

双车诊断数据与复现信息保留在 [pair interaction curriculum v1](datasets/pair_interaction_curriculum_v1/README.md)，不再作为当前训练入口。

双车 pilot 只改善 head-on，crossing 退化且 lane-swap 未改善，碰撞率从 `0.45` 升至 `0.60`；该模型已拒绝，结果仅用于证明按episode均衡不能保证replay均衡，见 `results/03_强交互Actor_研发记录/D4_pair_interaction_pilot_s20260724`。后续不继续双车研究，也不重新运行 PAIR→THREE。

五车旧 Stage 1 replay 审计发现，强Actor在无可见邻居状态上的变化反而最大，并在 `<=0.8 m` 危险状态继续增加线速度。根因是此前所谓 specialist 仍控制并更新于整个episode，本质上仍在训练generalist。当前候选改为训练期oracle分工：邻居真值距离 `<=2.0 m` 时由强Actor执行并进入其Actor更新集，其余状态始终由冻结5D执行；Critic仍学习完整五车轨迹。Actor结构、TD3目标和部署输入不变。先评估oracle组合上限，通过后才训练可部署Gate。

oracle-specialist pilot 在同一批 `60 deep + 40 close + 40 margin` validation 上，epoch 1/2 的 overall agent success 为 `0.827/0.834`，collision 为 `0.173/0.166`，但 full success 从 `0.500` 降到 `0.471`。deep/close/margin full success 分别从 `0.233/0.575/0.825` 变为 `0.200/0.575/0.775`，未形成强交互专门化。replay 审计显示候选Actor在所有距离层都学到约 `+0.13--+0.16` 的同向角速度偏置，并在 `<=0.8 m` 状态增加线速度 `+0.090`；Critic对这些明显变化有 `79.38%` 偏好。该Critic偏好只说明Actor与Critic自洽，不能单独证明动作在真实环境中更差；它与危险加速、同向转弯和分层评估一起构成训练信号偏移的警示。oracle 分流已修复“普通状态也更新强Actor”的问题，但当前候选仍未学出deep优势。该配置已停止，不增加epoch或seed重复。完整结果见 `results/03_强交互Actor_研发记录/D4_interaction_oracle_specialist_pilot_s20260724`。

两轮额外配对评估证明单次 full success 波动较大，因此不再仅依据 `0.500 -> 0.471` 判断。三次合计后，5D/候选的 overall agent success 为 `0.8238/0.8229`，full success 为 `0.4571/0.4452`，基本持平且候选略低；但目标 deep 层 full success 在三次中均下降，合计为 `35/180 -> 27/180`。因此拒绝的是当前epoch 2 checkpoint和未修改训练配置，不是双Actor + Gate主线。

Stage 1 同协议 epoch 1/2 的 overall full success 为 `0.4929/0.3357`；close、deep、margin 分别从 `0.6250/0.2167/0.7750` 降至 `0.2750/0.1333/0.7000`，未通过准入条件。离线审计进一步发现，训练后 Actor 在全部 `40001` 个 replay state 上只增加、不降低线速度；对动作变化超过 `0.05` 的状态，Critic 有 `76.80%` 错误偏好真实性能更差的新动作。因此停止 Stage 2，不用更多 epoch 或 seed 重复当前配置。完整结果见 `results/03_强交互Actor_研发记录/D4_strong_interaction_curriculum_stage1_s20260723`。

可部署相对运动观测的30场sensor probe显示：危险目标原始点云覆盖率为`97.92%`，说明Velodyne信息本身足够；但二维点簇质心跟踪的最佳precision/recall/FPR权衡未同时达到`0.70/0.80/0.10`准入线。主要误报来自静态环境点簇的质心抖动和错误关联，因此拒绝该具体特征，不接Actor；下一候选应加入三维高度轮廓或其他目标身份一致性。完整结果见 `results/04_Gate前置验证/D4_lidar_cluster_motion_probe_s20260724`。

后续三维XYZ shape probe在18个校准scenario和12个独立scenario上审计。8维形状逻辑回归在独立集达到precision/recall/FPR `0.651/0.912/0.307`，仍无法排除大量静态点簇。因此停止继续组合手工形状/速度阈值；若继续相对运动主线，下一候选必须是由仿真privileged CPA/TTC标签监督、部署时仅使用本机连续激光帧的时序编码器。完整结果见 `results/04_Gate前置验证/D4_lidar_cluster_shape_probe_s20260724`。

20-bin时序风险编码按scenario划分为18 train / 6 validation / 6 test，在完全相同输入下比较单帧MLP与8帧GRU。两者test precision均约`0.159`，FPR为`0.525/0.486`，GRU未恢复可用的CPA/TTC风险信号。因此拒绝“5D原20-bin输入 + 时序编码”，不接Actor或Gate；下一候选必须保留更高角分辨率，并使用新的scenario-level test。完整结果见 `results/04_Gate前置验证/D4_temporal_risk_encoder_20bin_s20260724`。

180-bin候选使用旧30场做24/6 train/validation，新的互斥30场只做最终test。GRU相对单帧有改善，但test precision/recall/FPR仅为`0.217/0.781/0.511`，仍远低于准入线。因此不在该holdout上继续调参，不接Actor或Gate。若继续，必须重新建立更大的互斥数据划分，并事先固定能利用角度局部结构的轻量时空编码器。完整结果见 `results/04_Gate前置验证/D4_highres_temporal_risk_encoder_s20260724`。

历史 edge-1 residual v1/v2 将 deep、close、margin 混在一起且只使用单帧观测，因此不视为当前强交互 Actor 的正式训练；结果仅用于说明价值外推和静态观测问题。

pilot 实测 epoch 1/2 full success 为 `0.5130/0.4704`，碰撞率为 `0.1546/0.1721`。更新后的 residual 几乎恒定饱和到 `[+0.10, -0.10]`，同时 Critic Q 上升而真实性能下降；拒绝当前 residual TD3 配置，不启动更多 seed 或 edge 1-2，完整归档见 `results/03_强交互Actor_研发记录/D4_interaction_edge1_residual_pilot_s20260720`。

后续只允许一个直接针对该机制的 v2：复用 epoch 1 已预热 Critic，将 Actor Q 项按 batch mean absolute Q 归一化，并以权重 `2.5` 约束到冻结 5D 动作；训练一轮 40000 agent samples 后在相同 423 场 validation 上判定。除该 objective 外不扩大网络、不增加交互难度。

v2 full success 为 `0.5177`，仅比现场冻结基线 `0.5130` 多 2 场，同时碰撞增加，且低于历史 5D 的 `0.5248`。Residual 边界饱和已消失，说明价值外推约束有效，但仍未形成可用的状态相关避让行为；停止继续调整 Actor objective，后续先补充相对速度/TTC 等交互观测或独立交互监督信号。完整归档见 `results/03_强交互Actor_研发记录/D4_interaction_edge1_conservative_residual_v2_s20260720`。

60 场冻结 5D 风险 probe 进一步确认：按同步路径最小间距分层后，deep/close/margin full success 为 `0.15/0.55/0.85`；失败组在进入约 `1.2 m` 接近区时的闭合速度更高、TTC 更短，但 5D 仍保持高线速度。当前单帧 Actor 不显式观测这些动态量，因此暂停 specialist 续训，先运行固定优先级让行 oracle 验证可解上限。完整归档见 `results/04_Gate前置验证/D4_interaction_risk_probe_5d_s20260721`。

固定优先级让行 oracle 在全部 edge-1 上将碰撞率从 `0.170` 降到 `0.147`，但 full success 从 `0.517` 降到 `0.450`。分层后，deep full success 从 `0.15` 升到 `0.35`，close/margin 却从 `0.55/0.85` 降到 `0.30/0.70`。因此拒绝“只要有交互就停车”，但保留“仅在紧迫冲突切换到交互专家”的主线。下一阶段先定义可部署的时序闭合速度/TTC 观测，不直接继续训练。完整归档见 `results/04_Gate前置验证/D4_interaction_risk_yield_oracle_s20260721`。

基于扇区最小距离的自运动补偿时序差分已在 20-bin 和独立 180-bin 输入上审计。两者 frame recall 为 `0.808/0.909`，但 false-positive rate 高达 `0.636/0.738`，且所有 episode 都被激活。根因是扇区最小值没有稳定物体关联；该特征族已拒绝，不接入 Actor/Gate，不继续调阈值。下一步只做原始二维点的自运动补偿移动簇可行性验证。完整归档见 `results/04_Gate前置验证/D4_temporal_interaction_scan_diff_s20260721`。

### S3: 条件交互Actor有效性审计

当前交互Actor按状态级分工训练：冻结普通导航Actor负责普通状态，交互Actor只在
oracle距离阈值内执行，并只从安全聚焦样本更新。因此首先验证其真实训练契约，
而不是强制它全程独立导航：

1. 冻结普通导航Actor全程运行固定strong-interaction validation。
2. 相同seed和scenario顺序下，按训练时相同的`2.0 m` oracle在普通导航Actor与
   交互Actor之间切换。
3. 按scenario ID记录both、weak-only、combination-only和neither，并按
   deep/close/margin分层。
4. 另用固定`edge=0` validation比较5A与5D，先确定普通导航Actor身份。

匹配重复验证中，5A与`5A + interaction Actor`的full success为
`0.421/0.700`；deep/close/margin从`0.183/0.400/0.800`变为
`0.533/0.825/0.825`。逐场weak-only/combination-only为`8/47`，说明条件
交互Actor在其实际调用方式下有效；同时平均步数`35.2 -> 54.3`和2个timeout
表明其行为偏保守。完整结果见
`results/05_当前冻结方案/D4_interaction_actor_matched_validation_s20260727`。

固定`2.0 m` oracle只用于验证训练上限，不能作为部署方法。强制交互Actor
全程运行的83场partial诊断只证明其不具备训练契约之外的普通导航能力，不用于
否定条件专家，也不进入正式对照。

Actor侧Gate准入条件已经通过：

- 匹配组合在deep上相对普通导航Actor至少`+15 percentage points` full success。
- combination-only显著多于weak-only，且提升不是由单一场景池贡献。
- 普通导航Actor在固定0-edge场景上达到可靠基线，并报告平均步数与timeout。
- test保持未读。

尚未通过的是部署感知准入：Gate必须使用本机传感器区分机器人与静态障碍，不能直接复用oracle真值距离。

### S4: Gate

- 从本阶段开始冻结5A和epoch-16条件交互Actor。所有训练命令只能更新感知模块或Gate参数。
- Gate进行状态级硬选择，不把standard/dense当作Actor标签；切换使用滞回或最短保持时间避免抖动。
- 固定`2.0 m`邻车真值分流只保留为不可部署上界。学习Gate的目标是提高真实导航回报，不是单纯拟合该距离标签。
- Gate训练混合读取strong/weak与standard/dense的train轨迹；validation固定，test保持未读。
- 不使用case名称、全局density标签、其他机器人odom或Critic特权context作为部署输入。

#### G0: 机器人/静态障碍区分（当前）

现有Actor的20维扇区最小距离会把机器人、墙和箱子压成同一种障碍。此前可部署探针已经完成，但都未达到准入线：

| 表示 | precision | recall | FPR | 决策 |
| --- | ---: | ---: | ---: | --- |
| 二维点簇运动 | 0.530 | 0.726 | 0.156 | 拒绝手工质心跟踪 |
| 三维8特征逻辑回归 | 0.651 | 0.912 | 0.307 | 拒绝手工形状阈值 |
| 20-bin 8帧GRU风险分类 | 0.159 | 0.684 | 0.486 | 拒绝压缩激光时序表示 |
| 180-bin 8帧GRU风险分类 | 0.217 | 0.781 | 0.511 | 拒绝小样本逐帧MLP+GRU |

前两项分别审计机器人点簇/运动，后两项审计CPA/TTC风险，标签任务不同，数值不能直接横向比较；共同结论只是它们都未达到各自预先固定的准入线。

原始点云对真值危险机器人的覆盖率为`97.92%`，形成有效点簇的比例为`97.55%`，说明瓶颈在目标分类和关联，不在传感器覆盖。G0使用新的、按scenario互斥的数据划分：

1. 只从train场景采集本机前视三维Velodyne XYZ、本机odometry和时间戳。
2. 其他机器人Gazebo位置只用于离线生成`robot/static`监督标签，不进入推理输入。
3. 先复现8维逻辑回归作为下界，再评估能利用点内结构的轻量点簇分类器；不修改两个TD3 Actor。
4. validation必须覆盖standard/dense、strong/weak、墙角和箱体等困难负样本，并按scenario划分，禁止逐帧随机泄漏。
5. G0至少达到既定validation准入线：precision `>=0.70`、recall `>=0.90`、FPR `<=0.10`；同时报告静态障碍专属FPR。
6. 旧`sensor_probe/sensor_holdout`已经用于方法选择，只作为历史基线；G0重新冻结互斥train/validation/test，新test在感知结构与阈值冻结前不得读取。

#### G1: 机器人相对运动

G0通过后，只对检测为机器人的点簇做跨帧关联，输出置信度、相对距离、方位、闭合速度、CPA和TTC。静止机器人也必须靠形状被保留，不能只把“会动的物体”定义为机器人。G1沿用仿真真值只做训练标签和validation评估的原则。

#### G2: 可部署启发式Gate

在train上固定距离/TTC阈值、滞回和最短保持时间，在validation上比较：5A always-on、最小激光距离Gate、robot-aware Gate和特权`2.0 m` oracle。该阶段只验证感知和切换链路是否可行，不训练Actor。

#### G3: Learned Gate

只有G0-G2证明可部署输入有信息增益后才训练Gate。两个Actor保持冻结，Gate根据机器人检测与相对运动特征选择`pi_N`或`pi_I`；优化目标使用导航成功、碰撞、超时和效率，不把`distance <= 2.0 m`当作唯一监督答案。学习Gate必须与调优后的可部署启发式Gate做配对对照，否则不能形成论文贡献。

最终 gate 准入条件：

- 弱交互full success相对5A下降不超过`3 percentage points`。
- high-interaction full success距固定oracle组合不超过`5 percentage points`。
- learned Gate在固定validation上显著超过可部署启发式Gate。
- 静态障碍引发的错误激活不会形成新的系统性碰撞或timeout。
- gate activation 随 `rho_I` 单调增加，但在 low density 不应长期激活。

## 6. 必须对照与消融

### 主基线

- Frozen `5D` historical baseline。
- 冻结5A普通导航Actor。
- 修复后 Critic 下的 full Actor fine-tune。
- 修复后 Critic 下的 head-only。
- 条件交互 Actor及其oracle组合。
- ORCA/RVO 或项目环境中可接入的经典去中心化避碰基线。
- 单一 recurrent/attention policy，回应“为何不训练一个全能策略”。

### 方法消融

- 5A always on。
- 5D always on。
- 条件交互 Actor always-on诊断。
- random/fixed gate：排除选择机制本身的作用。
- min-laser gate：不区分机器人和静态障碍的可部署下界。
- robot-distance/TTC heuristic gate。
- learned gate without robot identity：验证语义区分是否必要。
- robot-aware learned gate：完整方法。
- privileged `2.0 m` oracle：不可部署上界。
- 不同gate滞回和最短保持时间。
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
- robot detector precision / recall / FPR，另报静态障碍FPR
- relative-distance、closing-speed、CPA与TTC误差
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
| E0 | Frozen 5A / 5D | yes | yes | yes | yes | yes | generalist基线曲线 |
| E1 | Full fine-tune | yes | yes | yes | yes | yes | 退化对照 |
| E2 | Head-only | yes | yes | yes | yes | yes | 容量对照 |
| E3 | 5A + conditional Actor oracle | yes | yes | yes | yes | yes | 条件策略上界 |
| E4 | Min-laser Gate | yes | yes | yes | yes | yes | 无机器人身份的可部署下界 |
| E5 | Robot-aware heuristic Gate | yes | yes | yes | yes | yes | 感知与规则切换基线 |
| E6 | Learned Gate without identity | yes | yes | yes | yes | yes | 机器人区分消融 |
| E7 | Robot-aware learned Gate | yes | yes | yes | yes | yes | 完整方法 |
| E8 | Privileged distance oracle | yes | yes | yes | yes | yes | 不可部署上界 |

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
- `D4`：条件交互Actor训练、匹配重复validation与普通导航Actor选择。
- `D5-G0`：机器人/静态障碍分类达到独立validation准入线。
- `D5-G1`：机器人相对运动估计达到准入线。
- `D5-G2`：可部署启发式Gate完成，确认感知与切换链路有正向上限。
- `D6-G3`：冻结两个Actor，只训练learned Gate。
- `D7`：完整基线、消融、泛化和统计检验。
- `D8`：论文图表冻结。

当前处于`D5-G0`。D4已经得到可供Gate可行性开发的冻结组合：5A普通导航Actor和epoch-16条件交互Actor；独立Actor训练seed留到D7正式统计前补齐，不在当前继续微调。Gate尚未训练，test保持未读。

### D4机制排查与训练记录（历史）

以下段落记录当时每一步为什么继续或停止，只用于追溯，不覆盖上面的当前决策。

后续完成了5A warm-start下的Critic context严格对照：将旧世界坐标几何context修正为本车坐标系相对位置与相对速度后，epoch 2 full success仍从`0.500`降至`0.421`，deep从`0.200`降至`0.133`。Replay审计显示`<=0.8m`危险状态只占`2.84%`，且训练后Actor在该层增加线速度`+0.128`，Critic对明显变化动作的错误偏好率为`76.77%`。因此context修复保留，但不再增加epoch；下一次严格对照只修改安全reward：在`1.2m`内加入随线速度增长的近车处罚，并奖励“机器人间距增大且自身仍向目标前进”的有效避让结果。完整结果见`results/03_强交互Actor_研发记录/D4_interaction_ego_motion_from_5a_s20260725`。

安全reward对照仍未解决退化：epoch 2 full success从`0.457`降至`0.429`，deep从`0.150`降至`0.100`。进一步审计确认，fresh Critic在训练前就因随机初始化产生几乎单向的动作梯度；冻结Actor训练一个epoch后，该偏置被放大到危险状态中线速度和角速度正梯度均接近`100%`。同时Critic均匀采样全部Replay，而危险样本仅约`2.6%`；安全reward还曾计入Critic不可见或已结束的邻车。当前修复统一reward与Critic的可见active邻车口径、令Critic batch中`75%`来自交互样本，并在Actor解冻前检查危险样本动作梯度；样本不足或梯度近乎单向时继续冻结Actor，不再让错误Critic直接更新Actor。

上述Critic采样与解冻保护实验中，梯度门连续拒绝`226`次，Actor参数始终与5A完全一致。角速度单向偏置从约`96%`改善到约`50%-63%`，但危险状态线速度正梯度到结束仍为`96%-100%`，说明交互过采样只能修正左右不平衡，不能解决未见动作上的线速度外推。下一次对照保持TD3结构不变，在Critic loss中加入局部反事实排序：仅对`<=1.0m`且闭合速度`>=0.1m/s`的状态，要求减速`0.2m/s`的动作不低于原动作。离线联合TD回放中，ranking权重`5`将危险状态线速度正梯度从`98.1%`降到`88.9%`；权重`10+`出现明显过度保守，因此正式pilot固定为`5`并继续使用解冻门。

Critic反事实排序正式pilot将危险线速度正梯度稳定降到约`20%-35%`，但Actor解冻后full success仍从`0.500`降至`0.464`，deep从`0.217`降至`0.133`。Replay对照显示Actor在`<=1.2m`状态仍增加线速度`+0.012`，并产生全局角速度偏移`+0.093`。根因是Actor仍从全部`<=2.0m`交互样本更新，而Critic安全约束只覆盖近距离接近样本。下一次对照保留Critic修复，只让Actor从`<=1.0m`且闭合速度`>=0.1m/s`的候选池更新，并仅对角速度锚定5A；不约束线速度，以保留学习减速的空间。

安全聚焦Actor pilot首次在同一轮冻结基线对照中全面提升：agent success从`0.820`升至`0.840`，collision从`0.179`降至`0.160`，full success从`0.436`升至`0.500`；deep/close/margin full success分别从`0.150/0.500/0.800`升至`0.217/0.600/0.825`。Replay行为审计确认`<=0.8m`线速度下降`0.079`，危险加速问题已消除；仍存在`-0.055`全局角速度漂移。该epoch 2暂列强交互Actor候选，但考虑固定Gazebo验证仍有波动，下一步先重复140场固定验证，再决定是否运行独立训练seed和进入D5互补性审计。完整归档见`results/03_强交互Actor_研发记录/D4_interaction_focused_actor_from_5a_s20260725`。

同配置的全套5D对照中，epoch 2相对冻结epoch 1仅将agent/full success从`0.824/0.471`提高到`0.830/0.479`，即full仅多成功`1/140`场；deep/close有所提高，但margin full从`0.875`降至`0.800`。其改善明显弱于全套5A配置的`0.436 -> 0.500`。由于当时启动脚本把强Actor warm-start和oracle弱Actor绑定为同一模型，该结果比较的是“全套5A”与“全套5D”，不是纯初始化消融。下一次正式长跑将二者解耦：强Actor从5A初始化，非强交互状态固定由5D执行。完整归档见`results/03_强交互Actor_研发记录/D4_interaction_focused_actor_from_5d_s20260725`。

解耦后的`5A强Actor初始化 + 5D弱Actor`对照没有通过：冻结epoch 1的agent/full success为`0.839/0.514`，Actor更新后的epoch 2降至`0.811/0.407`；deep/close full分别从`0.217/0.650`降至`0.100/0.475`。恢复过程、Replay、Critic和epoch计数正常，退化发生在Actor解冻后。该结果说明不能在改变弱Actor轨迹分布的同时直接外推此前全套5A训练结论；停止该checkpoint。接下来回到唯一产生明确正向结果的全套5A实验，从其epoch 2 checkpoint原样续训，再独立评估最佳强Actor与5D的oracle配对。完整归档见`results/03_强交互Actor_研发记录/D4_interaction_focused_actor_5a_init_5d_weak_s20260726`。

后续续训到旧epoch 8进一步确认了安全聚焦Actor配置能够产生正向候选，但旧640场训练使用随机有放回抽样，实际场景覆盖和deep/close/margin短窗口比例均不受保证，因此旧epoch 1-8只作为配置筛选依据，不作为正式长跑曲线。正式长跑从原始5A重新开始，保持已验证有效的全套5A rollout、TD3、reward、Critic ranking、Actor安全聚焦更新和解冻条件不变；唯一实验变量是改用不含validation/test的2560场`full_train`以及修复后的`balanced_cycle`。上限设为16个固定样本epoch，预计足以完成至少一次全池遍历，并在每20,000 agent samples后继续使用同一140场validation选择best。旧epoch 7 + 新Critic的短暂重热实验在Actor解冻前停止，见`results/90_中止与无效运行/D4_aborted_e7_rewarm_balanced_preunlock_s20260726`。

正式均衡长跑已正常完成16个epoch和`320,000` agent samples，覆盖全部`2560/2560`个训练场景。训练期oracle组合的固定validation full success从epoch 1冻结5A的`0.436`提高到epoch 16的`0.707`，deep/close/margin分别从`0.183/0.475/0.775`提高到`0.617/0.675/0.875`，碰撞率从`0.174`降至`0.079`；同时平均步数从`33.1`升至`54.5`，表明策略更慢、更保守。epoch 16随后通过匹配重复validation并冻结；该结果包含privileged oracle分工，不能当作条件Actor全程独立成绩或Gate成绩。完整归档见`results/05_当前冻结方案/D4_interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726`。

训练同口径的独立重复validation已完成：5A基线与`5A + epoch16 interaction Actor`的full success为`0.421/0.700`，几乎复现训练时的`0.436/0.707`；agent success、collision和平均步数也分别复现为`0.916/0.080/54.3`。因此训练效果不是单次epoch 16偶然峰值，但结论只适用于条件交互Actor的匹配调用方式。完整归档见`results/05_当前冻结方案/D4_interaction_actor_matched_validation_s20260727`。

弱交互补测进一步显示，5A/5D在相同248个固定0-edge validation场景上的full success为`0.8750/0.8710`，逐场5A-only/5D-only为`12/11`，McNemar exact `p=1.0`。两者能力等价，因此主线选择与交互Actor训练分布一致的5A作为弱Actor，不再引入5D到5A的额外切换层。完整归档见`results/05_当前冻结方案/D4_weak_actor_5a_vs_5d_s20260727`。

## 11. 预期贡献表述

如果实验支持假设，贡献收敛为三点：

1. 区分 spatial density 与 interaction density，并提出基于同步名义冲突图的程序化评测协议。
2. 冻结可靠的普通导航Actor，只在稀有紧迫交互状态中训练条件TD3 Actor，避免完整微调造成能力覆盖。
3. 提出机器人感知驱动的去中心化Gate，并在density sweep、held-out archetypes和不同机器人数量上验证能力保持与条件策略调用。

如果 interaction-density 定义没有形成稳定难度曲线，不把它作为独立方法贡献，只作为实验协议；如果 gate 未接近 oracle，不宣称自适应专家选择成功。
