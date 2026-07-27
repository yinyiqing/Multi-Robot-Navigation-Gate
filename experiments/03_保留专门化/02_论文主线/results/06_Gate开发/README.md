# Gate 开发

这里保存两个冻结 Actor 之后的当前工作，不再混入旧感知探针或 Actor 训练结果。

| 阶段 | 状态 | 目标 |
| --- | --- | --- |
| [`D5_G0_robot_detector_v1/`](D5_G0_robot_detector_v1/README.md) | pilot complete | 候选召回通过；单帧硬语义分类未通过，保留形状软分数 |
| [`D5_G1_robot_tracking_v1/`](D5_G1_robot_tracking_v1/README.md) | pilot complete | 自运动补偿和关联通过；保留连续运动特征，拒绝简单EMA硬分类 |
| D5-G2 | next | 用privileged交互标签监督只读本机感知的状态级Gate |

G2 不要求 G0 先输出完美的机器人硬标签。它读取 G0/G1 的形状和运动连续证据，学习复现两个冻结 Actor 已验证的 `2.0 m` oracle 执行契约；仿真真值只生成训练标签。若 standard/0-edge 上静态障碍误激活过高，则停止 Gate。
