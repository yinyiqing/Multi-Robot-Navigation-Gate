# G30 / B6：单头 Router 与辅助一步 Reward 监督

状态：归一化重训完成，Dense validation 256 场闭环运行中。

## 目的

落实导师提出的同状态双 Actor 一步 reward 差监督，同时避免将 noisy 的即时 reward
差直接注入部署期路由决策。两个 Actor、G0/G1 和 B2 初始化均保持冻结。

## 方法冻结

- 主标签：最近机器人距离 `d <= 2 m` 的 interaction label `y`。
- 辅助标签：同一状态下两个冻结 Actor 各执行一步所得的
  `Delta r = r_I - r_N` 的符号。
- 仅当 `|Delta r| > 0.005`（由 P0 噪声界预先登记）时使用辅助样本。
- Router 为单一 interaction logit 输出；训练目标为
  `BCE(y, logit) + 0.25 * BCE(sign(Delta r), logit)`，辅助项只作用于可靠样本。
- 部署阶段不增加 reward head，也不做全范围 logit 融合；当 Gate 概率落在
  `[0.40, 0.60]` 不确定区间且 `|Delta r| > 0.005` 时，用 `sign(Delta r)` 作
  tie-breaker。原有 8-frame GRU、stride=2、hysteresis 和 minimum hold 保持不变。

## 准入顺序

1. 离线审计训练/验证样本的 reward 辅助样本数、`y` 与 `sign(Delta r)` 的一致率，
   并检查 phase FPR/recall 与输出校准。
2. 只有离线审计通过后，才在全新 development manifest 上运行一次 256 场闭环，
   并与同 manifest/seed 的冻结 B2 对照。
3. 不读取 G25 sealed 结果调参，不覆盖 G27/G28/G29，不训练 Actor。

## 当前实现

模型和损失函数位于 `TD3/reward_aware_gate.py`：
`AuxiliaryRewardTemporalGate`、`auxiliary_reward_gate_loss`。

本机当前 Python 环境缺少 PyTorch，因此单元测试暂无法执行；代码已加入
`tests/test_reward_aware_gate.py`，待进入项目训练环境后先运行该测试再进行离线训练。

## 运行记录

- 首次未归一化 checkpoint 的 6 个 episode 已停止并归档到
  `local_data/validation/invalid_attempt_20260911/`，不得读取。
- 修正后的 checkpoint 使用与部署控制器一致的 B3 `feature_mean/std`，并通过
  `tests/test_reward_aware_gate.py` 的 11 个测试。
- 当前队列脚本为 `scripts/run_g30_b6_validation.sh`，结果写入
  `local_data/validation/results/g30_dense256_b6_s20260911.npy`，与 B2 的配对评测
  需在同一 manifest/seed 下另行运行。
