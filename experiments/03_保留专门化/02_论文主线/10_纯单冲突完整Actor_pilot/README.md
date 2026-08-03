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

本实验否定的是：**Dense Actor 失败仅仅因为多冲突太复杂；换成纯单冲突并沿用 v9 保守配置后，它会自然开始改善。**

它还不能证明“完整 Actor 在单冲突上一定学不会”，原因是训练预算只覆盖 `170/511` 个训练场景，Actor 解冻后只有约 8500 agent samples，而且 `actor_lr=1e-6` 与 `anchor=0.5` 使策略变化很小。

当前不应直接进入 421 场景完整验证或 edge-2 泛化测试，因为最小监控集尚未准入。若继续验证完整 Actor 路线，下一步应只做一次更有辨识力的优化强度对照，而不是原配置续训；否则应回到局部专家与组合泛化主线。
