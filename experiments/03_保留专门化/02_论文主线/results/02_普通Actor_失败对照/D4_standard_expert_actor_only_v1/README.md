# D4 Standard Expert Actor-only Warm-start v1

状态：`failed diagnostic`。本轮证明“仅加载 5D Actor、随机初始化 Critic、无 Actor anchor 的完整微调”不能稳定得到 standard expert。

## 配置

- 初始化：加载 5D Actor，Critic 随机初始化。
- reward：`0.8` 自身 + `0.2` 距离加权邻居奖励。
- Actor：full fine-tune，LR `1e-6`，anchor `0`。
- Critic：geometry-local，LR `8e-5`。
- Actor 解锁：累计 `20000 agent samples`。
- 数据：fixed-v1 `standard/train`；正式 validation 固定回放前 100 场。

## 正式 Validation

| Epoch | Agent success | Collision | Unresolved | Full success | Timeout episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.846 | 0.138 | 0.016 | 0.510 | 0.080 |
| 8 | 0.838 | 0.148 | 0.014 | 0.540 | 0.070 |
| 9 | 0.848 | 0.146 | 0.006 | 0.560 | 0.030 |
| 10 | 0.850 | 0.148 | 0.002 | 0.560 | 0.010 |
| 11 | 0.786 | 0.198 | 0.016 | 0.400 | 0.080 |
| 12 | 0.746 | 0.188 | 0.066 | 0.250 | 0.310 |

同一协议内 best 为 epoch 10，但没有证明高于原始 5D；epoch 11-12 出现明显策略漂移和超时崩坏。该模型不得作为最终 standard expert，也不得读取 standard test。

诊断发现：v1 使用的训练代码把 300 步 timeout transition 写成 `done=0`，并按环境步数而不是有效 agent samples 更新 Critic。该问题已在后续训练代码修复，v1 结果不应与修复后的实验混合比较。

## 文件

- `smoke_epoch_001.log`：10 场随机 validation 的管线检查，不参与正式 best。
- `train_epoch_002_to_006.log`：40 场 validation 的早期训练诊断。
- `train_epoch_007_to_012.log`：100 场固定 validation 的正式曲线。
- `validation_epoch_007_to_012.npy`：正式曲线结构化数组。
- `best_epoch_010_actor.pth`：失败诊断中的最佳 Actor，仅用于复核与退化对照。
- `summary.json`：机器可读配置、结果和文件哈希。
