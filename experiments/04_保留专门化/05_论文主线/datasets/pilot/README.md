# Pipeline Pilot

这批数据只验证生成、固定划分和 manifest 回放，不用于训练或论文结果。

固定 seed：`20260717`。

| Pool | train | validation | test | reserve | train 平均冲突边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 10 | 3 | 5 | 2 | 0.90 |
| dense | 20 | 5 | 10 | 3 | 2.45 |

任务距离范围：

- standard pilot：约 `0.817-2.985 m`
- dense pilot：约 `0.905-2.294 m`

2026-07-17 使用真实五车 Gazebo 对 reserve 几何做了 reset pilot：standard `2/2`、dense `3/3` 通过，均无初始碰撞、无初始终止、传感器数据正常，复位位置误差低于 `0.15 m`。验证未加载 Actor。

冲突指标后来补充了“机器人到达后停在目标处”的处理，因此 scenario ID 随指标摘要更新；场景坐标由相同 seed 确定，没有变化。正式数据必须重新执行完整 Gazebo 筛选，不能直接放大这组 pilot 的结论。
