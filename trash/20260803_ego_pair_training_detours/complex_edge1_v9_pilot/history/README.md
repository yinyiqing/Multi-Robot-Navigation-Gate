# 纯单冲突完整 Actor pilot

## 核心问题

此前完整 Dense Actor 多次失败，但局部避障 Actor 在单冲突附近有效。缺失的对照是：**让一个完整 Actor 控制全程，但只在严格的单冲突场景中训练，它能否学出来？**

本实验用于路线选择，不直接作为最终方法结论。

## 控制变量

沿用稳定的 independent Actor v9 配置：

- 从原始 5A Actor 初始化，不使用 epoch-16 权重；
- 新建 ego-motion local Critic；
- Actor 使用原始 24 维观测并控制完整回合；
- Oracle rollout、Gate 和 target-policy Oracle 全部关闭；
- 保留 v9 奖励、低 Actor LR、Q normalization、5A anchor 和 acceleration cap。

唯一的实质变化是把 Dense 多冲突训练集换为全路径校正后的纯 edge-1 数据。

## 短训练配置

| 项目 | 值 |
|---|---|
| model | `full_actor_edge1_n5_seed20260803_pilot_v1` |
| seed | `20260803` |
| train | `edge1_full_horizon_v1/train.json.gz`，511 场景 |
| monitor | `validation_monitor_50.json.gz`，25 standard + 25 dense |
| budget | 3 x 5000 agent samples |
| Actor unlock | 6500 agent samples |
| epoch 1 | 冻结 5A 基线 |
| epochs 2-3 | Actor 更新 |

启动与停止：

```bash
scripts/start_training_full_actor_edge1_pilot.sh
scripts/stop_training_full_actor_edge1_pilot.sh
```

训练日志匹配：`logs/train_full_actor_edge1_n5_seed20260803_pilot_v1_*.log`。

## 判断标准

先比较 epoch 1 基线与 epochs 2-3：

- 正信号：collision 下降，并且 agent success / full success 不出现明显下降；
- 强正信号：在固定 50 场景上重复出现 collision 下降，同时 full success 上升；
- 负信号：collision、timeout 上升，或 full success 明显下降；
- 无学习：更新后指标和动作统计都与 5A 基线基本相同。

若得到正信号，再对 421 场景完整验证，并测试未训练的 edge-2/多冲突泛化。若仍无学习，则“冲突过多”不是 Dense Actor 失败的充分解释，应停止继续堆训练轮数，转向 Actor 表达、观测可辨识性或优化目标诊断。

## 论文位置

该实验对应从单车到五车工作中的机制诊断：先验证单个局部双车冲突是否能被完整策略学习，再判断能否组合泛化到多车冲突。它能补齐证据链，但单独不构成论文创新点。

## 2026-08-03 运行结果

运行正常达到 `max_epochs=3`，没有异常退出。固定 50 场景评估结果如下：

| epoch | Actor 状态 | agent success | collision | unresolved | full success | timeout |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 冻结 5A 基线 | **0.816** | **0.184** | **0.000** | **0.420** | **0.000** |
| 2 | 解冻更新 | 0.812 | 0.188 | 0.000 | 0.400 | 0.000 |
| 3 | 继续更新 | 0.800 | 0.196 | 0.004 | 0.420 | 0.020 |

训练程序选择 epoch 1 为 best。相对原始 5A，Actor 参数变化为：

- epoch 1: relative L2 `0`，确认冻结基线未变化；
- epoch 2: relative L2 `0.00025936`；
- epoch 3: relative L2 `0.00043613`。

因此本次结果不满足正信号：碰撞没有下降，agent success 和平均奖励持续下降，epoch 3 还出现了 unresolved 与 timeout。它也不是训练崩溃，而是 Actor 发生了小幅、稳定的更新，但这些更新没有转化为策略收益。

### 当前结论边界

先前把结果解释成“单冲突简化仍无效”过强。后续 Critic 审计表明，Actor 解冻前的优化目标已经失真，因此本次 pilot **不能用于判断单冲突任务是否可学，也不能否定冲突数量是 Dense Actor 难训的重要原因**。它只证明当前 v9 协议没有改善。

当前不应提高 Actor 学习率、续训、进入 421 场景完整验证或 edge-2 泛化测试。提高学习率只会放大下述错误 Critic 梯度。

## 2026-08-03 训练链路复查

### 1. Critic 在危险状态持续鼓励加速

epoch 1 best checkpoint 中 Actor 仍是冻结 5A。对 checkpoint replay 的离线审计得到：

- interaction replay 共 `2321` 条；
- `697` 条状态的最近机器人距离 `<=1.2 m`，其中 `613` 条正在接近；
- 上述近距状态的 `dQ1/d(linear_action)` 有 `100%` 为正；
- 对 raw linear action `[-1,-0.5,0,0.5,1]` 扫描时，Q1 和 Qmin 在 `697/697` 条状态上都选择 `1`；
- 近距状态平均即时 reward 为 `-1.076`，近距且接近状态为 `-1.452`；
- 近距且接近样本的环境线速度均值约 `0.83`，速度 `<=0.5` 的样本只有约 `14%`。

这说明 Critic 没有学到“危险时减速”的条件动作排序。5A anchor 和 acceleration cap 只能阻止 Actor 比 5A 更快，不能产生正确的减速梯度。

### 2. 对口同状态 Gazebo 校准未通过

校准器已增加两个必要约束：

- `--anchor-agents conflict_pair`：只测 manifest 唯一冲突边中的两台车；
- `--reward-profile dense_v9`：真实分支使用本次训练的 v9 reward，而不是旧 D2 reward。

在 epoch 1 best checkpoint 上测试 2 个固定 edge-1 scenario：

| 项目 | 结果 |
|---|---:|
| 总分支 | 40 |
| 可重复分支 | 12 |
| 可校准状态组 | 3 |
| 可比较动作对 | 17 |
| Qmin 排序一致 | `6/17 = 0.353` |
| Qmin / N-step target MAE | `91.64` |
| Qmin bias | `-32.34` |

代表性状态中，raw speed 从 `-1` 到 `1` 时 Qmin 从 `-12.54` 单调升到 `-1.66`；真实分支中低速不碰撞，而 `0.5/1.0` 均碰撞并得到约 `-77` 的 N-step target。Critic 仍把碰撞全速动作评为最好。

原始记录：`local_data/critic_calibration_epoch1_v9_conflict_pair.json`。

### 3. edge-1 数据仍未形成纯双车训练信号

数据集的拓扑检查是正确的：每个场景完整路径只有一条冲突边。但当前训练协议仍有三处混入：

- interaction replay 按“任意可见邻车 `<=2 m`”标记，不按 manifest 冲突 pair 标记；
- epoch 1 全部 transition 中约 `46%` 被标为 interaction，而真正冲突 pair 只占 5 台车中的 2 台；
- cooperative reward 使用 `10 m` 前向视野内的活跃邻车平均，不只耦合冲突 pair。

因此“场景只有一条边”不等于 Critic 收到的是干净的双车动作因果数据。

### 4. Actor 观测是次级风险，不是本轮首要结论

Actor 仍只看 20 个激光扇区、目标和上一动作，不能显式区分机器人与墙，也没有相对速度。它可能限制最终泛化，但 epoch-16 已证明同一 24 维 Actor 能学到部分反射式单冲突避让，因此现在不应据此宣判任务不可学。应先修复 Critic 动作排序，再判断是否需要改观测。

## 下一次最小实验

下一步不是直接训练 Actor，而是做一轮独立 Critic 准入：

1. 冻结 5A Actor，全程不更新 Actor；
2. 在全部 5 台车中每步轮换一台 ego 做受控线速度覆盖，其他车执行冻结 5A；
3. replay 只保存当步被受控的 ego transition，避免其他车当前随机动作对转移产生隐藏混杂；
4. 关闭 v9 anchor、cap、恢复奖励和其他 Actor 约束，使用紧凑基础 reward；
5. Critic 未通过同状态 Gazebo 排序校准前，不解冻 Actor；校准可读取冲突 pair 真值作为测量标签，但训练不读取 pair；
6. 校准通过后只做 `5000` samples 的低学习率 Actor pilot，再决定是否增加 normal-state anchor。

这条实验不训练 pair 专家，不做冲突分解，也不引入 Gate。它仍然让一个 Actor 面向全部 5 台车和完整回合；这里只是先把“Critic 能否正确评价单冲突动作”从“Actor 能否更新”中拆开。这样下一次失败能定位到 Critic，下一次成功才有资格归因到 Actor 学习。

### 已启动的 Critic-only v2

启动时间：`2026-08-03 22:43`。

```bash
scripts/start_training_full_actor_edge1_simple_critic_v2.sh
scripts/stop_training_full_actor_edge1_simple_critic_v2.sh
```

| 项目 | 值 |
|---|---|
| model | `full_actor_edge1_simple_critic_v2_s20260803` |
| train | corrected edge-1 train，511 场 |
| monitor | corrected edge-1 monitor，50 场 |
| Actor | 5A 初始化，全程冻结 |
| 受控探索 | 全 5 车轮换 single ego；不读取冲突 pair |
| replay | 每步只保存受控 ego transition |
| reward | individual simple；无 cooperative / v9 附加项 |
| Critic | fresh 52D ego-motion local Critic |
| budget | 12000 replay samples，约 9000 次 warmup 后更新 |

日志：

```text
logs/train_full_actor_edge1_simple_critic_v2_s20260803_20260803_224303.log
```

训练结束后使用同一个 `individual_simple` reward profile 做同状态校准。pair 真值只用于选取测量对象，不进入训练、Actor 输入或部署策略。

### Critic-only v2 中途审计与止损

进程在 `5352` samples 后停止，最后一个完整保存并可恢复的 checkpoint 为 `4957` samples，未进入 Actor 更新。Actor optimizer step 始终为 `0`，参数相对 5A 的 L2 变化为 `0`。

| checkpoint | Critic updates | 1.2 m 逼近样本 | 负终止样本 | 逼近状态全速最优 | 逼近状态 `dQ/dv > 0` |
|---:|---:|---:|---:|---:|---:|
| 3071 | 92 | 145 | 19 | 4.1% | 13.8% |
| 3751 | 772 | 175 | 24 | 57.7% | 18.3% |
| 4957 | 1978 | 214 | 29 | 88.8% | 69.2% |

动作覆盖本身正常：4957 samples 的 raw linear 均值 `0.009`、标准差 `0.575`，低/中/高动作比例约为 `9.5% / 80.6% / 9.9%`。transition 的 ego state、action、reward、done 和 next state 索引一致，没有发现 replay 串线。

主要风险来自监督密度而非动作覆盖：

- 4957 条 replay 中只有 214 条属于 1.2 m 内正在逼近，仅占 4.3%；
- 负终止只有 29 条，而正终止有 171 条；
- 逼近样本分到五个速度档后每档只有 38 至 51 条，碰撞终止每档只有 0 至 3 条；
- 本轮只随机线速度，角速度与冻结 5A 输出的平均绝对差为 `4e-8`，因此不能据此解冻完整 Actor 的角速度分支。

CPU 同状态校准得到总体 pairwise 排序 `47/59 = 79.7%`，但这个数字不能作为冲突 Critic 通过：36 条可重复记录和 8 个有效状态组全部来自 `anchor_step=0`，`anchor_step=4` 的 50 条记录全部不可重复。有效分支主要测到“起点加速是否更快到达”，没有测到回合中途的冲突决策。

校准复现失败的工程原因也已定位：reset 后即使起点位置误差只有 `1e-5 m`，激光 Actor state 偶尔可相差 `2.19`；同一 action prefix 回放 4 步后，最大位置/航向误差达到 `0.24 m / 0.56 rad`。固定步长下的 controller settle 已改为按仿真时间等待并在失败时终止，但 smoke test 表明 Gazebo controller/传感器状态仍未被 reset 完整恢复。

因此当前结论是：v2 没有证明 Critic 失败，也没有达到 Actor 解冻准入。单看 Q 对全速的偏好会误杀，因为部分场景高速确实更优；单看 79.7% 排序又会误判，因为它只覆盖起点。下一轮训练前必须先满足：

1. 校准报告必须有 `post_initial_calibrated_groups > 0`，否则排序分数无效；
2. replay 必须在 Critic 更新前达到预设的逼近状态和负终止样本下限；
3. 若要解冻完整 Actor，必须补足角速度动作支持；否则首轮只能限制为线速度方向的更新。

归档结果：

- `local_data/critic_replay_audit_simple_v2_3071.json`
- `local_data/critic_replay_audit_simple_v2_3751.json`
- `local_data/critic_replay_audit_simple_v2_4957.json`
- `local_data/critic_calibration_simple_v2_4957_conflict_pair.json`
