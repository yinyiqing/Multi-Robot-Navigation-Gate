# D4 Standard Expert Timeout-fix v3

状态：`promising diagnostic, not accepted`。本轮只验证 timeout terminal 与 Critic 更新比例修复，不读取 standard test。

## 配置

- 初始化：完整加载原始 5D Actor/Critic，全新 replay。
- reward：`0.8` 自身 + `0.2` 距离加权邻居奖励。
- Actor：full fine-tune，LR `1e-6`，anchor `1.0`，累计 `5000 agent samples` 后解锁。
- 数据：fixed-v1 `standard/train`；validation 固定回放前 100 场。
- 修复：timeout transition 写为 terminal；Critic 更新按有效 agent samples 归一化。

## Validation

| Epoch | Agent success | Collision | Unresolved | Full success | Timeout episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.870 | 0.130 | 0.000 | 0.580 | 0.000 |
| 2 | 0.882 | 0.114 | 0.004 | 0.620 | 0.020 |
| 3 | 0.870 | 0.124 | 0.006 | 0.530 | 0.030 |

epoch 1 是 Actor 未解锁的同协议 5D 基线。epoch 2 相对基线提高 `+1.2` agent-success points 和 `+4.0` full-success points，但 100 场单次评估不足以确认统计显著性；epoch 3 回落。因此 epoch 2 只作为候选，不进入 standard test。

timeout transition 已在 replay 中核对为 `done=1`。相对 5D，epoch 2 Actor 参数平均绝对变化约 `1.64e-5`，最大约 `4.85e-4`。

## 文件

- `train_epoch_001_to_003.log`：完整训练与 validation 日志。
- `validation_epoch_001_to_003.npy`：三轮结构化曲线。
- `best_epoch_002_actor.pth`：同协议 best Actor，仅供后续 validation 复核。
- `summary.json`：机器可读结果和哈希。
