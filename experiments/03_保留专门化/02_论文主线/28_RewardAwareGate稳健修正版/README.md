# G28 Reward-aware Gate 稳健修正版

状态：离线准入通过；独立 Dense256 development 闭环已完成；未通过替换 B2 条件。

## 目的

G27-P4 显示，原始一步 `Delta r` 的终止奖励使 reward head 近似输出常数，并在继续更新
B2 的 GRU/interaction head 时破坏了原有 phase calibration。G28 是导师提出的
“交互标签 + 双 Actor 同状态 reward 差”路线的修正版，不覆盖 G27，也不把 G27-P4 改写成
成功结果。

## 固定改动

1. `Delta r` 仍由同状态、同评价 reward 的两个冻结 Actor 一步反事实分支得到；不重新训练
   Actor，不改变 G27 的数据或 reward 定义。
2. 只使用 G27-P1 training split 计算稳健尺度
   `s = P95(|Delta r|)`，目标为
   `A = tanh(clip(Delta r, -s, s) / s)`。validation 不能参与尺度计算。
3. 从冻结 B2 初始化，冻结 B2 的 GRU 与 interaction head；只训练新增 advantage head。
4. 部署分数为
   `score = sigmoid(l + 0.25 * exp(-|l-l_on|/1.0) * deadzone(clip(A,-1,1),0.1))`，其中
   `l` 是冻结 B2 interaction logit，`l_on=logit(0.43)`。reward head 只在 B2 的切换边界
   附近作为小幅 tie-breaker，远离边界或优势绝对值不超过 `0.1` 时保持 B2 的相位决定。
5. 阈值 `0.43/0.33`、minimum hold `3`、stride `2` 和序列长度 `8` 全部沿用 B2，
   不做闭环结果驱动的搜索。

## 离线准入

在启动任何闭环前，必须保存训练/validation target 统计、常数基线与 advantage head 的
MAE/RMSE/Pearson、B2 phase 指标和 conservative fusion 指标。离线报告至少满足：

- robust target 的 validation Huber 不劣于 training-median 常数基线；目标的旧 MAE 筛查仍须
  原样报告，但不作为唯一准入条件。由于 validation target 大量为零且包含少数截断终止值，
  MAE 会偏向常数预测，不能代表本版本实际优化的鲁棒损失；
- advantage head 的 validation Pearson 为有限值且高于常数基线（常数基线记为 0）；
- 冻结 B2 phase head 的 frame/event 指标完全不变；
- conservative fusion 的 FPR 不高于 B2 超过预先登记的 0.02 容差，且 recall 不下降超过
  0.02。

这些是进入闭环的工程准入，不是对 sealed test 的统计显著性主张。首轮 MAE-only 筛查失败已
记录在训练 summary；后续只按本节完整准入条件判断。G27-P4 与 B2 仍是论文可审计记录。

## 当前离线结果

`seed20260911_final` 选择第 38 个 epoch。bounded target 的 train-only 尺度为 `1.52196455`；
validation Huber 从 training-median 常数基线 `0.04906` 降到 `0.03043`，Pearson 为 `0.4141`。
旧 MAE 筛查为 `0.20089` 对 `0.16102`，明确记为未通过，不被隐藏。冻结 B2 phase 的 FPR 为
`0.26306`，融合后为 `0.26208`，recall 为 `0.87792` 对 `0.87683`。checkpoint 的逐参数
冻结审计由 `scripts/audit_g28_offline_checkpoint.py` 完成后，才可登记新的 development 闭环；
这些结果不等于 sealed test 的成功证据。

原始 B4 stdout 曾被后续 B2 运行覆盖，无法逐字恢复；保留的 `.npy` 结果已由
`scripts/reconstruct_g28_b4_log.py` 生成逐 episode 的
`logs/b4_result_reconstructed.log`。该文件明确标注为 reconstructed，不包含不可恢复的
Gazebo 启动信息和 wall-clock `samples/sec`。

## Development 闭环

离线审计通过后，只运行一批新的 Dense256 validation manifest、B4 seed `20260911`、256
episodes 的闭环，并在同一 manifest、同一 seed 下完成冻结 B2 对照。该运行不读取 G27-P4，
不进入确认性统计。

### 闭环结果与停止结论

结果摘要见 `local_data/validation/p3_summary.json`；两个原始结果分别为：

- B4：full success `43.75%`（112/256），agent success `80.08%`，robot-level collision
  `19.77%`，unresolved `0.16%`，timeout `0.78%`，raw steps `33.39`，interaction-actor
  selection share `70.94%`。
- 同 seed B2：full success `48.44%`（124/256），agent success `81.95%`，robot-level
  collision `18.05%`，unresolved `0%`，timeout `0%`，raw steps `31.44`，interaction-actor
  selection share `70.31%`。

因此 B4 相对 B2 的 full success 下降 `4.69` 个百分点，robot-level collision 上升 `1.72`
个百分点，raw steps 增加 `1.96` 步；interaction-actor selection share 仅变化 `+0.63`
个百分点。B4 未通过预先登记的“不得低于冻结 B2”替换条件。G28 至此停止，不再追调 reward、
dead-zone、融合权重、阈值或 hold/stride。B2 仅作为冻结对照和初始化来源，不能据此恢复为论文
最终方法；若继续推进导师路线，必须登记新的 reward-aware 候选和独立准入协议。

## 数据边界

G28 只读 G27-P1 的 `audit/train.npz`、`audit/validation.npz` 及冻结 B2/A1 数据构建离线
窗口。不得读取 G27-P4、G25 sealed 或 G26 结果来选择尺度、融合参数、epoch、阈值或模型。
本目录的唯一 development 闭环已经完成；后续不得据此继续调参或把结果并入 G25 确认性统计。
