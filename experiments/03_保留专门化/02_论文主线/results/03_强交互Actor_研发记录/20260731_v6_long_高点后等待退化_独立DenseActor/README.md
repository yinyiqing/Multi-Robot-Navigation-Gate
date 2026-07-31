# 20260731 v6 long：高点后等待退化（独立 Dense Actor）

## 目的

验证 v6「恢复前进」奖励是否能解决独立 Dense Actor 后期学成等待/卡住策略的问题。

## 配置

- 模型名：`independent_dense_actor_from_5a_recovery_v6_cpu_long_s20260730`
- warm-start：5A actor only
- 控制方式：新 Actor 独立控制全部 5 个机器人
- oracle / actor 切换：关闭
- 训练集：`datasets/fixed_v1/dense/train.json.gz`
- validation：`views/dense_validation_monitor_ultrafast_v3/validation.json.gz`
- 计划 epoch：30
- 实际停止：epoch 19 后人工停止
- 设备：CPU

## 主要结果

| epoch | success | full | collision | timeout | avg steps | 备注 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.768 | 0.420 | 0.232 | 0.000 | 31.9 | 初始尚可 |
| 7 | 0.764 | 0.480 | 0.228 | 0.040 | 41.4 | 第一次明显提升 |
| 11 | 0.836 | 0.540 | 0.160 | 0.020 | 44.6 | best checkpoint |
| 13 | 0.812 | 0.520 | 0.156 | 0.140 | 83.0 | 已开始变慢 |
| 15 | 0.748 | 0.360 | 0.168 | 0.360 | 140.8 | 等待退化明显 |
| 17 | 0.760 | 0.260 | 0.132 | 0.480 | 173.8 | timeout 主导 |
| 19 | 0.668 | 0.200 | 0.112 | 0.680 | 226.6 | 已严重退化 |

## 结论

v6 不是碰撞越来越多，而是后期又学成「保守等待/走不完」。

证据：

- collision 从 epoch 11 的 `0.160` 降到 epoch 19 的 `0.112`；
- 但 timeout 从 `0.020` 飙到 `0.680`；
- full success 从 `0.540` 掉到 `0.200`；
- avg steps 从 `44.6` 增到 `226.6`。

所以 v6 的恢复前进惩罚有帮助，但不足以阻止 TD3 后期 actor 被 critic 梯度推向低碰撞、低完成率的等待策略。

## 后续修正

本次之后加入 validation 退化早停机制：

- 保存 best checkpoint 仍以 `full_success` 为主；
- 当 validation 相比 best 出现明显 `full_success` 下滑或 `timeout` 上升时累计 bad count；
- bad count 达到 patience 后自动停止训练；
- 避免继续把 latest 训练坏。

这不是改变 TD3 主体，而是实验保护机制，防止高点后继续无意义过训练。

## 文件

- `logs/`：完整训练日志
- `checkpoints/`：本地保留 best/latest checkpoint，不进 git
- `pytorch_models/`：best、epoch 11、epoch 19 的 actor/critic 快照
