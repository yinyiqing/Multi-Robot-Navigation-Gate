# 5 智能体密集场景

本目录用于归档五车 dense case 测试结果。

## 当前状态

五车密集场景已开始。当前已完成共享 Policy Baseline，D2 几何邻域 Critic + Weighted08 正在作为下一组对照推进。

## 计划方法

| 编号 | 方法 | 状态 |
| --- | --- | --- |
| A | 共享 Policy Baseline | 已完成 |
| C | Weighted08 | 待定 |
| D2 | 几何邻域 Critic + Weighted08 | 待测 |

## 当前结果

| 编号 | 方法 | success_rate | collision_rate | full_success_rate | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| A | 共享 Policy Baseline | 0.977 | 0.019 | 0.910 | 已完成 |
| D2 | 几何邻域 Critic + Weighted08 | - | - | - | 待测 |

## 观察

五车 dense baseline 明显强于五车 standard baseline，说明当前 dense 设置更偏短程局部交互验证，并不一定比标准场景更难。
