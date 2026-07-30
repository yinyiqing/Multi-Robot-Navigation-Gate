# 20260730 v6 CPU probe：恢复前进奖励下的独立 Dense Actor

## 目的

验证 v6 reward 代码和 Gazebo 稳定性，不用于证明训练收敛。

本次强制使用 CPU：

```bash
CUDA_VISIBLE_DEVICES=''
```

原因是当时 GPU 显存几乎占满，但 CPU RAM 已恢复充足。

## 配置

- 模型名：`independent_dense_actor_from_5a_recovery_v6_cpu_probe_s20260730`
- 初始模型：5A actor warm-start
- 训练集：`datasets/fixed_v1/dense/train.json.gz`
- validation：`datasets/fixed_v1/views/dense_validation_monitor_ultrafast_v3/validation.json.gz`
- 设备：CPU
- epoch：1
- 每个 epoch：10000 agent samples
- validation：50 episodes

## 结果

| epoch | success | full success | collision | timeout |
|---|---:|---:|---:|---:|
| 1 | 0.772 | 0.400 | 0.228 | 0.000 |

直方图：

- success_hist 0..5：`[0, 1, 6, 12, 11, 20]`
- collision_hist 0..5：`[20, 11, 12, 6, 1, 0]`

## 解释

这次只能说明：

1. v6 代码能跑；
2. CPU 模式下 Gazebo 稳定；
3. v6 起点没有明显坏掉；
4. 第 1 个 epoch 不能证明训练效果。

原因：

- `actor_update_delay_steps=21000`
- 第 1 个 epoch 只有 `10000 agent samples`
- 因此 actor 基本还没真正开始更新。

## 结论

CPU 路线可用于长跑。后续正式验证需要至少 5-10 epoch，最好看 30 epoch 长跑的 best checkpoint。
