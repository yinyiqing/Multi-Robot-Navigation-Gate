# 中止的训练前诊断

本目录只保存两次未形成有效结论的短运行，防止它们与正式实验混淆。

## safe-distance pilot

- 原模型名：`interaction_oracle_safe_distance_pilot_s20260724`
- 5D Actor/Critic warm-start。
- 在约 `6500/40000` agent samples 时停止。
- Actor 解冻阈值为 `21000`，因此该运行从未更新 Actor，不能用于判断 safe-distance reward 是否有效。
- checkpoint 已重命名为 `TD3/checkpoints/interaction_oracle_safe_distance_pilot_s20260724_aborted_preunlock.pt`，只用于追溯，不作为 warm-start。

## 5A short test

- 原测试名：`strong_interaction_5a_20260724_214416`。
- 只完成 4/140 场后停止。
- 当时错误地尝试用普通模型判断场景可解性，样本不完整，不用于模型比较。

完整日志在 `logs/`，5A 短测的状态和统计在 `results/`。
