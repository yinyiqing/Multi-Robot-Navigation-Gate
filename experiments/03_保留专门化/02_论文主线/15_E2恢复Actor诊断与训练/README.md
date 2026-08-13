# E2恢复Actor诊断与训练

状态：`I-E2, I-E2-M and I-E2-F4 completed / all rejected`.

本目录记录从新普通 Actor `E2` 出发，重新定义条件避障 Actor 的路线。它不是对当前
冻结 `5A + epoch-16 + Gate` 主线的静默替换；若后续要进入论文主方法，必须先更新
`PROJECT_STATUS.md` 和论文主线协议。

## 当前判断

`E2` 单独运行已经显著强于旧 `5A`，其主要短板不再是旧 `5A` 那种近车即高碰撞，而是：

```text
局部强交互或密集交互之后，恢复推进、退出僵持和完成全局任务仍不稳定。
```

同场120场validation的现有证据：

| 方法 | full success | collision | timeout | 结论 |
| --- | ---: | ---: | ---: | --- |
| 旧5A | `0.5583` | `0.1900` | `0.0000` | 旧普通基线 |
| E2单独 | `0.7500` | `0.0567` | `0.0667` | 当前更强普通Actor候选 |
| E2 + 旧epoch-16真值2m切换 | `0.7250` | `0.0800` | `0.0667` | 旧避障Actor不适配E2 |
| E2-local-critic当前配置best | `0.7167` | `0.0667` | `0.0667` | 未超过E2 |

因此，不能把 `epoch-16` 视为可直接迁移到 `E2` 的通用避障模块。新的 Actor I 应定义为
`interaction recovery actor`，只解决 `E2` 的局部失败窗口。

## 训练目标

Actor N：

- 固定 `E2`；
- 保持24维Actor输入；
- 负责默认导航、目标推进和静态障碍处理；
- 不再加入邻域critic或动态reward，避免破坏已形成的普通导航能力。

Actor I：

- 从 `E2` warm start；
- Actor推理输入仍保持24维；
- 训练时允许Critic使用邻域/相对运动上下文；
- 训练时允许加入动态交互reward；
- 只在 `E2` 的失败或近失败窗口接管并更新。

Gate / oracle：

- `2.0 m` 不再是充分切换条件；
- `2.0 m` 只作为候选窗口筛选条件之一；
- 真正监督目标应接近：此刻调用 Actor I 是否比继续 E2 更有利。

## 失败窗口定义草案

第一阶段不直接训练，先固定 `E2` 做诊断。候选窗口包括：

1. collision 前若干步；
2. timeout或unresolved episode的后段；
3. 近车距离低于阈值且进度长期接近0；
4. 多车近距离互锁、让行后不恢复；
5. E2成功但耗时明显长的近失败case。

窗口进入不应只看 `d_nearest <= 2.0 m`，而应组合：

```text
near_robot OR dense_local_geometry
AND low_progress_or_stagnation
AND not_near_goal_success
```

## 动态reward与邻域critic

动态reward和邻域critic只用于 Actor I 训练侧，目标是解除冲突而不是鼓励保守等待。

需要同时约束：

- 碰撞惩罚；
- 动态近距离风险惩罚；
- 低进度/僵持惩罚；
- 目标方向保持；
- recovery后推进奖励；
- 接管后状态应更容易交还给E2。

若只惩罚近距离，Actor I 可能再次学成保守等待；该配置不得启动长跑。

## 最小准入

正式训练Gate之前必须先通过oracle短窗验证：

```text
E2 + recovery-oracle > E2 单独
```

同时至少满足：

- full success提高；
- collision不高于E2；
- timeout不系统性高于E2；
- 平均步数不能显著恶化；
- 改善case多于退化case。

如果 recovery oracle 都不能超过 `E2`，不得进入可部署Gate训练。

## 已实现的recovery-oracle pilot

已在 `TD3/test_velodyne_td3_multi.py` 增加 `DRL_MULTI_ACTOR_SELECTION_MODE=recovery_oracle`。
它不是最终可部署Gate，而是训练新 Actor I 前的最小准入诊断。

当前规则：

```text
candidate = nearest active visible robot <= 2.0 m
stagnating = mean recent progress <= 0.003
             OR recent distance decrease <= 0.02 m
switch_to_I = candidate AND stagnating AND goal_distance > 0.45 m
release_to_N = released OR near_goal OR no_longer_stagnating
```

默认滞回：

- 释放距离：`2.4 m`;
- 进度窗口：`5`步;
- 最短保持：`3`步;
- 最长保持：`20`步。

相关环境变量：

| 变量 | 默认值 |
| --- | ---: |
| `DRL_MULTI_RECOVERY_ORACLE_CANDIDATE_DISTANCE` | `2.0` |
| `DRL_MULTI_RECOVERY_ORACLE_RELEASE_DISTANCE` | `2.4` |
| `DRL_MULTI_RECOVERY_ORACLE_PROGRESS_THRESHOLD` | `0.003` |
| `DRL_MULTI_RECOVERY_ORACLE_PROGRESS_WINDOW` | `5` |
| `DRL_MULTI_RECOVERY_ORACLE_DISTANCE_DELTA_THRESHOLD` | `0.02` |
| `DRL_MULTI_RECOVERY_ORACLE_GOAL_DISTANCE` | `0.45` |
| `DRL_MULTI_RECOVERY_ORACLE_MINIMUM_HOLD_STEPS` | `3` |
| `DRL_MULTI_RECOVERY_ORACLE_MAXIMUM_HOLD_STEPS` | `20` |

启动脚本：

```bash
scripts/start_e2_recovery_oracle_epoch16_pilot.sh
```

运行日志：

```text
logs/active/current-generalist-r2style/e2-recovery-oracle-epoch16-pilot/
```

结果与逐步轨迹：

```text
experiments/03_保留专门化/02_论文主线/15_E2恢复Actor诊断与训练/local_data/recovery_oracle_epoch16_pilot/
```

该pilot使用旧 `epoch-16` 作为临时 Actor I，只回答一个问题：

```text
更克制、更接近“恢复失败窗口”的切换，是否已经能避免旧2m oracle的误切退化？
```

若仍不能超过 `E2`，结论不是“Gate没法做”，而是旧 `epoch-16` 与 `E2` 的恢复窗口不适配；
下一步应按本目录定义训练新的 interaction recovery actor。

## 当前下一步

recovery-oracle 120场已经完成：

| 方法 | full success | collision | timeout | 平均步数 | I动作占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2历史同场 | `0.7500` | `0.0567` | `0.0667` | `49.28` | `0` |
| E2 + 旧epoch-16 2米oracle | `0.7250` | `0.0800` | `0.0667` | `50.56` | `44.6%` |
| E2 + 旧epoch-16 recovery-oracle | `0.7750` | `0.0550` | `0.0500` | `41.98` | `19.1%` |

recovery相对E2为14场改善、11场退化，`p=0.6900`；收益集中在multi-edge
（`0.600 vs 0.500`）。由于E2历史结果是seed `20260817`、recovery是`20260818`，必须
先补同seed E2-only控制，不能直接宣称提高2.5个百分点。

## I-E2 40k pilot

为使最终两个Actor都从E2路线产生，当前授权的新Actor为
`interaction_recovery_from_e2_strong40k_s20260820`：

- 初始化与窗口外reference Actor均为E2；
- strong-interaction full train，`balanced_cycle`；
- 2米内由I-E2 rollout，2米外冻结E2；
- Actor只在interaction replay更新；
- 87维ego-motion邻域Critic；
- dynamic distance-weighted reward，`0.8 self + 0.2 neighbors`；
- Actor前21k冻结，之后学习率`1e-6`；
- 两个20k epoch，禁止自动延长；
- 完成后自动运行E2 + I-E2 recovery-oracle 120场。

自动流水线：

```bash
bash scripts/start_e2_ie2_overnight_pipeline.sh
```

日志统一位于：

```text
logs/archive/diagnostic/e2_ie2_recovery/
```

## I-E2 40k pilot结果

自动流水线于`2026-08-12 07:41`完整结束，matched E2控制、40k训练和matched recovery
复测均一次完成。manifest顺序、重复ID、缺失ID和terminal outcome accounting审计通过。

| 方法 | full success | collision | timeout | 平均步数 | I动作占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2 matched | `0.6583` | `0.090` | `0.100` | `53.63` | `0` |
| E2 + 旧epoch-16 recovery | **`0.7750`** | **`0.055`** | **`0.050`** | **`41.98`** | `19.1%` |
| E2 + I-E2 recovery | `0.7333` | `0.070` | `0.075` | `46.92` | `19.4%` |

I-E2相对matched E2为19场改善、10场退化、91场持平，McNemar exact `p=0.1360`。
它已经产生正向导航收益，但没有通过显著性准入，也没有超过旧epoch-16。I-E2与旧
epoch-16的调用比例近似，因此差距不能归因于recovery规则调用次数不同。

按冲突拓扑分层：

| topology | E2 | E2 + 旧epoch-16 | E2 + I-E2 |
| --- | ---: | ---: | ---: |
| zero-edge | `0.925` | `0.950` | `0.925` |
| edge-1 | `0.775` | `0.775` | **`0.850`** |
| multi-edge | `0.275` | **`0.600`** | `0.425` |

训练内部strong validation的full success从epoch 1的`0.579`提高到epoch 2的`0.721`，
说明40k pilot没有发生训练坍塌；当前缺口集中在multi-edge，不应写成“I-E2没有学会避障”。

冻结artifact：

- Actor：`TD3/pytorch_models/interaction_recovery_from_e2_strong40k_s20260820_best_actor.pth`
  - SHA-256：`fd8a97bbae2e8553df3897c3c7119341497f86b09f40a3abff2c205e7514fe02`
- Full checkpoint：`TD3/checkpoints/interaction_recovery_from_e2_strong40k_s20260820_best.pt`
  - SHA-256：`a47b7130844d8a2d2f591298b46512cd2288a088fdf420cbbc7441a97bd7cdc0`

## 当前下一步

只做离线诊断，不立即续训：

1. 对齐同一scenario中E2、旧epoch-16 recovery和I-E2 recovery的episode结果；
2. 优先检查multi-edge中I-E2相对旧epoch-16退化的case；
3. 比较接管时长、线速度、角速度、最近机器人距离、进度和collision前窗口；
4. 判断缺口来自训练分布、Actor更新量，还是动态reward造成多车交互中的错误推进；
5. 诊断完成后再登记唯一下一实验，不直接追加40k。

## 诊断结果：I-E2的multi-edge缺口

离线脚本与输出位于：

```text
analyze_ie2_multi_edge.py
local_data/ie2_multi_edge_diagnosis/diagnosis.json
local_data/ie2_multi_edge_diagnosis/paired_cases.csv
```

审计通过：两份JSONL轨迹均覆盖同一120个scenario ID，顺序和结果数组一致。

### 训练分布审计

`strong_interaction_curriculum_v1/full_train.json.gz`的`2560/2560`场景，以及其
`validation.json.gz`的`140/140`场景，manifest指标均为：

```text
conflict_edge_count = 1
max_conflict_degree = 1
simultaneous_conflict_count = 1
```

这与最终N5 validation按`zero/edge1/multi`划分的multi-edge场景不是同一分布。I-E2
pilot因此只验证了从E2出发的单冲突恢复能力，不能期待它自然具备多冲突恢复能力。

## I-E2-M 多冲突分布修订 pilot

新清单只来自 navigation-train，并与所有已冻结的非train视图互斥：

```text
train:        2400场 = edge-1 960 / edge-2 720 / edge-3+ 720
internal val:  200场 = edge-1 140 / edge-2 30 / edge-3+ 30
```

训练每10场按 `edge1, edge2, edge1, edge3plus, edge1, edge2, edge1, edge3plus,
edge2, edge3plus` 交错。清单构建脚本为 `scripts/build_ie2_multi_conflict_views.py`，
哈希为 train `3c0be8a32513bde471cc5c6731f528397195fdda670be8e3bcf8cc7265ecaba`、
validation `9f44ee3962edb5f0750b8791cb357d0df09d3161552a94fb183a9b193cb6b5e0`。

相对首版仅修改训练分布和恢复奖励，不修改24维Actor输入、E2 warm start、interaction-only
更新或87维邻域Critic。首版配置的 `DRL_MULTI_REWARD_MODE=average` 不会调用 interaction
reward；I-E2-M 改为 `average_plus_interaction`，停滞惩罚为 `0.02`，并启用弱 safe-recovery
（penalty `0.10`、progress bonus/idle penalty 均为 `0.08`）。预算为两个20k agent
samples，Actor在21k后解冻，完成后自动在冻结N5 120场上做 matched 复测。

### 行为诊断

在13个“I-E2退化、旧epoch-16成功”的配对case中，I-E2的交互动作占比为`0.396`，
平均交互线速度为`0.0148`，停滞帧比例为`0.928`，多邻居局部帧比例为`0.0325`；
旧epoch-16分别为`0.144/0.1467/0.775/0.2473`。I-E2并非主要因为高速推进而碰撞，
而是更频繁接管、更近距离地持续等待，导致多车互锁后的恢复推进失败。

9个multi-edge退化case中，7个是碰撞、1个是timeout/unresolved，且冲突边数为2至4。
相对地，I-E2改善的8个case中有5个来自edge-1，只有2个来自multi-edge。这与训练分布
审计完全一致：当前I-E2已经是有单冲突收益的Actor，但不是多冲突恢复Actor。

### I-E2-M最终结论

I-E2-M已按上述multi-edge清单完成40k训练和matched 120场复测，不再是待执行项。
E2、E2+旧epoch-16 recovery、E2+I-E2-M的full success为`0.7417/0.7750/0.6833`，
multi-edge为`0.450/0.600/0.400`。实际训练episode中edge-1/edge-2/edge-3+为
`169/127/126`，所以失败不是“multi-edge清单没有跑到”。

根因是训练单元与恢复任务错位：Actor梯度只覆盖1米内仍在闭合的risk状态，排除了
避让后停滞、分离、恢复推进和release。I-E2-M主要改变了角速度，没有形成旧epoch-16
那样的恢复动作。因此拒绝I-E2-M，不追加40k，不扫描reward权重，也不启动新Gate。
完整证据见[I-E2-M诊断](I_E2_M_DIAGNOSIS.md)。

若未来重启E2配套Actor，必须先定义approach、avoidance、stalled recovery和release四阶段
训练单元，并用离线或短窗反事实准入过滤候选；不得直接启动第三个40k pilot。

## I-E2-F4 四阶段pilot最终结论

F4按登记协议完成40k，但仍未通过。Actor冻结的epoch 1与21k后更新的epoch 2在同一200场
internal validation上的full success为`0.630/0.605`，collision为`0.094/0.099`，
timeout为`0.090/0.100`，平均步数为`54.64/66.10`。Actor梯度门持续通过，说明不是更新
被阻塞；扩大到完整2米交互窗口并增强恢复奖励后，更新反而同时损害成功率、安全和效率。

训练器保留的`best`属于Actor冻结阶段，不是训练出的F4专家；因此没有继续用它做N5末段
复测。F4被拒绝，不追加预算，不进入Gate。完整记录见
[I-E2-F4结果](I_E2_F4_RESULTS.md)。

训练前完成的同seed matched控制显示，E2与`E2 + old epoch-16 recovery`的full success为
`0.6917/0.7750`，后者15场改善、5场退化，exact `p=0.04139`。旧epoch-16虽与E2血缘
不同，仍是当前唯一在matched闭环中证明有收益的条件恢复Actor。
