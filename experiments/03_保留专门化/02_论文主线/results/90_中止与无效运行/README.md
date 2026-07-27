# 中止与无效运行

本组只保留复现和错误追溯信息，不能用于模型选择、趋势判断或论文结果。

| 实验 | 无效原因 |
| --- | --- |
| `D4_aborted_preunlock_diagnostics_s20260724` | Actor 解冻前停止或样本过少 |
| `D4_aborted_independent_5a_seed_s20260726` | 首次 validation 前停止 |
| `D4_aborted_e7_rewarm_balanced_preunlock_s20260726` | 混合旧采样协议，Actor 解冻前停止 |
| `D4_aborted_balanced_sampling_reset_bug_s20260726` | validation 重置训练游标，实际覆盖严重不足 |

修正后的有效替代实验必须在其他职责目录中使用新的实验 ID 保存。
