# G12-R2B 5A流程参数匹配Actor结果

状态：`completed / 30k frozen as fair degraded baseline`。日期：`2026-08-14`。

## 结果

R2B从历史3D2 Actor函数保持扩宽到`24-1137-855-2`，使用五车procedural standard、
individual reward、fresh 24维Critic和历史5A的优化参数。训练完成`30,374` agent samples，
在冻结的120场N5 internal validation上得到：

| checkpoint | Actor状态 | Agent success | Full success | Collision | Timeout | 平均步数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10k / epoch 1 | 冻结 | `0.805` | `0.533` | `0.195` | `0.000` | `27.6` |
| 20k / epoch 2 | 刚越过解冻边界 | `0.800` | `0.508` | `0.200` | `0.000` | `33.3` |
| 30k / epoch 3 | 约10k更新 | `0.548` | `0.042` | `0.418` | `0.150` | `85.9` |

参数审计进一步确认：epoch 1继承块相对3D2的最大变化为`0`，新增隐藏单元到输出层的权重
L1也为`0`，所以它只是函数等价的扩宽3D2，不是训练出的参数匹配大Actor。epoch 2的继承
块最大变化为`2.97e-4`，新增输出权重L1为`0.113`；epoch 3分别为`4.80e-3/1.704`，说明
更新确实发生，但导航性能同时坍塌。

## 结论

R2B没有形成稳定的已训练大Actor。自动best位于Actor冻结阶段，不能拿它作为
已训练容量baseline；解冻后候选先小幅退化，随后严重坍塌。这复现了历史5A在Actor解冻后退化
的优化现象，并说明“严格复制5A最后一段训练”不适合训练加宽Actor。

R2B internal validation与F-A1的G11-F-C pilot不是同一manifest，因此不能把`0.533`与
`0.710`写成正式同场比较。项目接受退化作为对照结果，因此冻结真正更新过的epoch 3
作为R2B-30k公平容量baseline，并追加G11-F-C同场闭环评测。R2-10k继续只作
cross-protocol诊断，不进入公平容量baseline排名。

本实验只说明5A recipe的优化稳定性不足，不说明大Actor容量不足，也不支持双Actor+Gate
优于参数匹配单Actor。
