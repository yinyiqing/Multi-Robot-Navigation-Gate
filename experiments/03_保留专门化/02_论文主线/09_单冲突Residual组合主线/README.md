# 单冲突Residual组合主线

状态：`closed / rejected evidence`。本目录保存Residual、整网续训和蒸馏的失败证据，
不再表示当前主线。

## 原候选路线

```text
冻结5A + epoch-16指导的Residual -> 独立Actor B
冻结Actor A/B + 可部署Gate       -> 多冲突零样本组合
```

[05_单冲突Residual Actor B](05_单冲突Residual_ActorB/README.md)已完成最小离线准入。
直接用24维单帧状态学习`epoch-16 - 5A`动作差没有通过，因此R1/R2没有启动。当前已经
冻结5A和epoch-16为两个独立角色，直接开发Gate，不再寻找新的Residual Actor B。

## 历史实验

| 目录 | 状态 | 结论 |
| --- | --- | --- |
| [`01_双教师离线蒸馏pilot`](01_双教师离线蒸馏pilot/README.md) | rejected | 5A轨迹上的逐帧蒸馏没有覆盖学生闭环分布 |
| [`02_D2b_Critic同状态校准`](02_D2b_Critic同状态校准/README.md) | rejected | fresh Critic的Q动态范围塌缩，不能解冻Actor |
| [`03_epoch16完整状态分叉pilot`](03_epoch16完整状态分叉pilot/README.md) | rejected after larger validation | 50场提升没有在200场复现 |
| [`04_epoch17_epoch18固定200场复核`](04_epoch17_epoch18固定200场复核/README.md) | complete / rejected | full success下降、collision上升，否定epoch-16整网续训 |

历史目录中的运行命令已移除，防止把被拒绝路线误当成当前入口。结果、日志和checkpoint
说明保留用于论文失败对照。

## 与旧Residual的区别

旧Residual是`冻结5D + 零初始化Residual + TD3自行探索`，没有使用epoch-16的局部
避让知识。新Residual固定为：

- base必须是5A；
- base永久冻结；
- epoch-16只提供单冲突状态下的动作教师；
- Residual先学习教师动作差，再进入闭环TD3；
- Actor和Gate训练均禁止使用多冲突场景。

该组合已经完成R0并被拒绝。这里的设计差异只用于解释历史，不构成重启理由。
