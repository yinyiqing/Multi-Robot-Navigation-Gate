# 双教师离线蒸馏pilot

## 目的

在启动Gazebo续训前，快速检查能否把不可部署的`5A + epoch-16` Oracle组合直接蒸馏为一个Actor B：正常帧回归5A动作，交互帧回归epoch-16动作。

## 数据与协议

- train/validation各100个互斥scenario，分别为`4004/4074`帧；
- 数据来自冻结5A轨迹，不是Oracle组合自己产生的轨迹；
- Oracle-positive rate为`0.585/0.568`；
- 训练按scenario和正负类别平衡；
- student从epoch-16精确初始化，checkpoint选择最小化`max(normal MSE, interaction MSE)`；
- 只做离线动作保真评估，不把结果解释为闭环导航成绩。

冻结模型SHA256：

- 5A：`fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5`
- epoch-16：`6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b`
- G0 detector：`0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56`

## 结果

| Pilot | 输入/初始化 | normal MSE | interaction MSE | teacher-choice accuracy |
| --- | --- | ---: | ---: | ---: |
| `pilot_v1` | 24D / epoch-16 | `0.353` | `0.133` | `0.575` |
| `pilot_generalist_init` | 24D / 5A | `0.254` | `0.254` | `0.540` |
| `pilot_front_label` | 24D / epoch-16 | `0.270` | `0.193` | `0.576` |
| `pilot_deployable_gate` | 76D G0/G1 / epoch-16 | `0.335` | `0.136` | `0.586` |

76D输入由原24D Actor state加4条机器人候选轨迹和8维全局统计组成。新增52列首层权重初始化为零，因此训练前输出与epoch-16逐位一致。该实现通过单元测试验证。

## 结论

四个pilot都没有学出可靠的逐帧教师选择。加入可部署机器人检测/跟踪特征后，teacher-choice accuracy只从约`0.575`提高到`0.586`，不足以进入闭环。继续增加同一数据上的epoch只会拟合5A行为分布，不能解决student运行后产生的新状态。

本pilot否定的是“在5A轨迹上直接逐帧行为克隆两个教师”的捷径，不是否定`epoch-16 -> 全程Actor B`路线。后者必须让Actor B在自己的轨迹上全程控制，通过导航回报学习推进和重启，同时约束其交互能力不被遗忘。

## 复现

训练入口：`scripts/train_policy_consolidation.py`。

本地产物保存在`local_data/`并由Git忽略；每个run包含`summary.json`、`history.json`和最佳Actor checkpoint。
