# Gate 开发

这里保存第一版Gate及其感知前端。当前新Gate应复用其中有效组件，但不得把本目录旧的
“下一步”当作已批准执行协议。

| 阶段 | 状态 | 目标 |
| --- | --- | --- |
| [`D5_G0_robot_detector_v1/`](D5_G0_robot_detector_v1/README.md) | pilot complete | 候选召回通过；单帧硬语义分类未通过，保留形状软分数 |
| [`D5_G1_robot_tracking_v1/`](D5_G1_robot_tracking_v1/README.md) | pilot complete | 自运动补偿和关联通过；保留连续运动特征，拒绝简单EMA硬分类 |
| [`D5_G2_interaction_gate_v1/`](D5_G2_interaction_gate_v1/README.md) | G2-B v1 rejected | G2-A保留为诊断；单次8步反事实标签因带噪闭环不可重复而拒绝 |

G2不要求G0先输出完美的机器人硬标签。它读取G0/G1的形状和运动连续证据；`2.0 m`
oracle只保留为监督、诊断和上界。多次带噪rollout只是一项历史候选，随后锚点恢复pilot
也未通过，不再被视为唯一下一步。新的Gate协议必须在父目录论文主线中重新登记。
