# G29 Reward-aware 事件证据 Gate

状态：离线训练已完成，尚未运行闭环；等待触发率审计后再启动 development。

G28 的退化来自把一步 `Delta r` 回归值直接作为 phase logit 修正。G29 保留 `2 m` 交互标签
和冻结双 Actor 的同状态 reward 差，但把 reward 预测限定为事件级 tie-breaker：只有预测优势
绝对值达到预先固定的置信阈值，且 B2 的 interaction logit 位于切换边界附近时，reward 才能
影响路由；弱或远离边界的 reward 证据不改变原有相位。

## 固定决策

```text
score = sigmoid(logit_B2 + alpha * exp(-|logit_B2-logit_on|/tau)
                * 1[|A_hat| >= 0.25] * A_hat)
alpha = 0.20, tau = 0.75, threshold = 0.43/0.33
```

`A_hat` 仍由 G27 的冻结 counterfactual reward 数据训练；B2 只用于初始化 GRU/interaction
head 和作为冻结对照。该版本不允许读取 G25/G26/G27-P4/G28 闭环结果来调参数。

## 失败原因与改动依据

G28 在同一 manifest/seed 下相对 B2 的 full success 下降 4.69 个百分点，robot collision
上升 1.72 个百分点，而 interaction selection share 只变化 0.63 个百分点。逐场转移为
29 场改善、41 场退化，说明问题集中在少量临界路由状态；因此 G29 先限制 reward 对临界状态
的影响范围和最小证据强度，再进行离线审计和一次 development 闭环。

## 准入顺序

1. 用 G27-P1 training/validation 数据训练唯一 G29 checkpoint，冻结 reward 目标、阈值和
   `alpha/tau`；不得用 G28 闭环结果选模型。
2. 审计 reward 证据触发率、相位变化率以及与 `2 m` 标签的方向一致性。
3. 只有离线审计通过后运行一批新的 Dense256 development 闭环，并同场运行冻结 B2 对照。
4. 若 G29 未达到替换条件，停止该版本并记录；B2 仍只作为对照，不写成论文最终方法。

## 当前离线结果

唯一 checkpoint 位于 `local_data/training/seed20260911_v2/`，由 G27-P1 training/validation
数据训练，未读取 G28 闭环结果。validation Huber 为 `0.030431`，与 G28 相同；reward 证据
触发率（`|A_hat| >= 0.25`）为 `22.81%`。在 57 个 validation 锚点上，selective score 与
冻结 B2 phase 的阈值决定一致（0 个额外相位改变）。这说明当前阈值足够保守，不能直接把 G29
当作已经有效；下一步必须在新的闭环状态上检查它是否能修正 B4 的错误临界路由，而不是只凭
离线回归指标宣布成功。
