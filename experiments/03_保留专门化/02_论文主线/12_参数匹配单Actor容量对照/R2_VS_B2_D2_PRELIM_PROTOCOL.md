# R2-10k大Actor与B2同场预比较

状态：`registered / CPU evaluation / preliminary only`。日期：`2026-08-08`。

## 目的

R3-40k在Actor解冻后退化，R2-10k仍是冻结的大Actor候选。本实验只回答：在已经完成的
G11-D2独立validation上，R2-10k与B2是否存在明显性能差异。它不替代最终统一方法表，
也不读取sealed test。

## 固定输入

- Manifest：`fixed_v1/views/g11_d2_admission_v1/validation.json.gz`
- Manifest SHA-256：`6250b941f127d550641a621d4253e17ea0770ff3c0cb94e6254e1f26b9f4978a`
- 场景：`200`场，`0-edge/edge-1`各100场；顺序和评测seed沿用D2
- Evaluation seed：`20260809`
- Candidate：R2-10k Actor，SHA-256见`R2_N5_ADMISSION_RESULTS.md`
- Comparator：已归档的G11-D2 B2结果，不重新训练、不重新调阈值
- Device：CPU，独立ROS/Gazebo端口`15523/15623`

B2的历史D2结果与本次R2结果按`scenario_id`逐场配对。分析使用
`scripts/compare_actor_validation.py`，报告full success、agent success、collision、
timeout、平均步数及0-edge/edge-1分层结果。

## 边界

该实验只含0-edge和edge-1，不能回答大Actor在multi-edge上是否优于B2。multi-edge比较
必须使用G11-E或另一个预先冻结的互斥manifest，并重新运行所有待比较方法。

