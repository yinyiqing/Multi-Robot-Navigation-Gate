# Gate 开发

这里保存两个冻结 Actor 之后的当前工作，不再混入旧感知探针或 Actor 训练结果。

| 阶段 | 状态 | 目标 |
| --- | --- | --- |
| [`D5_G0_robot_detector_v1/`](D5_G0_robot_detector_v1/README.md) | pilot complete | 候选召回通过；单帧硬语义分类未通过，保留形状软分数 |
| [`D5_G1_robot_tracking_v1/`](D5_G1_robot_tracking_v1/README.md) | pilot complete | 自运动补偿和关联通过；保留连续运动特征，拒绝简单EMA硬分类 |
| [`D5_G2_interaction_gate_v1/`](D5_G2_interaction_gate_v1/README.md) | G2-B v1 rejected | G2-A保留为诊断；单次8步反事实标签因带噪闭环不可重复而拒绝 |

G2 不要求 G0 先输出完美的机器人硬标签。它读取 G0/G1 的形状和运动连续证据，最终学习当前状态下哪个冻结 Actor 更好；`2.0 m` oracle只保留为上界和辅助基线。下一候选是多次带噪rollout统计标签，不使用已拒绝的单次硬标签。
