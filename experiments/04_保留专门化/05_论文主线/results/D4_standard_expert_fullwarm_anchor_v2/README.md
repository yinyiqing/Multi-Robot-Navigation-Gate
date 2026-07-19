# D4 Standard Expert Full Warm-start + Anchor v2

状态：`failed diagnostic`。完整加载 5D Actor/Critic 并使用 anchor `1.0` 后，Actor 解锁仍导致连续退化；epoch 4 validation 未完成即停止。

| Epoch | Agent success | Collision | Unresolved | Full success | Timeout episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.864 | 0.134 | 0.002 | 0.530 | 0.010 |
| 2 | 0.850 | 0.140 | 0.010 | 0.520 | 0.050 |
| 3 | 0.818 | 0.158 | 0.024 | 0.450 | 0.110 |

该结果说明完整 Critic warm-start 和 anchor `1.0` 不能修复旧训练代码的后期 timeout 漂移。后续排查确认旧代码将 timeout transition 写成 `done=0`，因此 v2 不应与修复后的 v3 混合比较。

归档包含完整日志、三轮 validation 数组和同协议 best（epoch 1）Actor。SHA-256：

- `best_epoch_001_actor.pth`: `32faaffe842e6dce99b868883fa640da2c9b84a4f66e66d99b47346b92e1f4ac`
- `train_epoch_001_to_003_partial_004.log`: `71a18d62e2a2a93d4fec29c5fd2abe7831257941de7ed24b678792a79cd3dc1e`
- `validation_epoch_001_to_003.npy`: `9ea84c50ded7c373293b7ac731ae0f11e483d1a087233776eb6de3dd82f59f37`
