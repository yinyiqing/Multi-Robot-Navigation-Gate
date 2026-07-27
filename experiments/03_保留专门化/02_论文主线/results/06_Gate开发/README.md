# Gate 开发

这里保存两个冻结 Actor 之后的当前工作，不再混入旧感知探针或 Actor 训练结果。

| 阶段 | 状态 | 目标 |
| --- | --- | --- |
| [`D5_G0_robot_detector_v1/`](D5_G0_robot_detector_v1/README.md) | implementation complete / data pending | 单帧激光区分机器人和静态障碍 |
| D5-G1 | pending | 对机器人检测结果做跨帧跟踪，估计闭合速度和 TTC |
| D5-G2 | pending | 冻结两个 Actor，只训练状态级 Gate |

顺序不能颠倒：G0 不通过时，不用错误的身份信号训练 Gate。
