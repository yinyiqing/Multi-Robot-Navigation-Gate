# 20260801 简化TD3动作覆盖实验C

## 目的

验证实验B的失败是否由Critic数据中的线速度覆盖不足造成。保持网络、奖励和TD3训练参数不变，仅改变前`20000`条replay的行为动作覆盖。

## 唯一改动

- 前`10000 agent samples`：5A提供转向，raw linear action均匀采样`[-1, 1]`，转向叠加标准差`0.10`的Gaussian noise；
- 后`10000 agent samples`：恢复完整5A action，线速度和转向都叠加标准差`0.10`的Gaussian noise；
- replay达到`6000`后开始更新Critic；
- Actor在`21000 agent samples`后才允许更新，因此本次4个epoch中始终冻结；
- 每`5000 agent samples`使用同一批固定50场dense validation验证一次。

随机线速度只用于训练采样，不用于validation。没有启用全动作随机、TTC、让行、停车规则、anchor、gradient gate或额外网络模块。

## 不变项

- warm-start：5A Actor；Critic重新初始化；
- 原始24维TD3；
- reward：`0.8自身 + 0.2活跃邻车`；
- progress `10.0`、forward `0.0`、turn penalty `0.05`、obstacle penalty `1.0`、timeout `-150.0`；
- batch `128`、Critic LR `2e-5`、Actor LR `2e-6`、gamma `0.999`、tau `0.005`。

## 判断标准

运行到`20000 agent samples`后停止并审计。只有危险状态（最小激光距离不超过`0.5m`）的速度Q排序不再偏好全速，才允许基于同一checkpoint解冻Actor。

若仍明显偏好全速，则停止该路线，不通过继续增加epoch或修改奖励碰运气。

## 脚本

- 启动：`scripts/start_training_dense_simple_td3_hparam_c.sh`
- 停止：`scripts/stop_training_dense_simple_td3_hparam_c.sh`
- 审计：`scripts/audit_simple_td3_critic.py`

## 结果

Actor全程冻结，参数与5A完全相同。固定validation结果如下：

| epoch | success rate | full success rate |
|---:|---:|---:|
| 1 | 0.724 | 0.360 |
| 2 | 0.736 | 0.400 |
| 3 | 0.756 | 0.400 |
| 4 | 0.712 | 0.320 |

Critic危险状态（最小激光距离不超过`0.5m`）速度排序：

| epoch | `Q(full)-Q(stop)` | 偏好全速的样本比例 |
|---:|---:|---:|
| 1 | +0.107 | 86.9% |
| 2 | -0.565 | 0.0% |
| 3 | +0.973 | 100.0% |
| 4 | +1.915 | 100.0% |

前`10000`条随机线速度数据中，危险状态五档动作数量为`114/123/113/114/131`，分布均衡；后`10117`条恢复5A采样后变为`231/12/4/10/358`，中间速度再次消失。

结论：动作覆盖确实能把Critic排序纠正，但在`10000`步停止覆盖后，5A的双峰动作分布再次把Critic带回错误方向。实验C不允许解冻Actor，由实验D将均衡覆盖延长到完整Critic warmup阶段。

正式日志位于`logs/formal/`，审计位于`audits/`。checkpoint、模型和TensorBoard文件保留在本地归档中，不提交GitHub。
