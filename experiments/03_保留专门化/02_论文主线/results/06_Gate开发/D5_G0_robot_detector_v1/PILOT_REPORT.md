# G0 单帧机器人检测 Pilot

日期：2026-07-27

结论：候选生成合格，单帧 CNN 未达到机器人/静态障碍硬分类的准入线。不能据此开始 Gate 训练，也不能靠增加 epoch 解决。G0 后续只作为候选和形状置信度来源，下一步进入带自运动补偿的 G1 目标跟踪。

## 固定数据

train 和 validation 各 100 场，均由四层各 25 场组成：

- standard / weak
- standard / interaction
- dense / weak
- dense / interaction

两组场景 ID 互不重叠，导航策略固定为 5A。sealed test 未采集、未评估。

| split | 场景 | 采样帧 | 候选 | 正候选 | 可见机器人 | proposal recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 100 | 1855 | 15425 | 4485 | 4649 | 0.9647 |
| validation | 100 | 1647 | 15433 | 4520 | 4684 | 0.9650 |

validation 四层 proposal recall 均稳定：

| 层 | proposal recall |
| --- | ---: |
| dense / interaction | 0.9585 |
| dense / weak | 0.9625 |
| standard / interaction | 0.9663 |
| standard / weak | 0.9762 |

## 单帧 CNN

基线在 epoch 19 取得最佳 validation F1：

| precision | recall（包含 proposal 漏检） | FPR | F1 | 中心偏移 MAE |
| ---: | ---: | ---: | ---: | ---: |
| 0.6626 | 0.8920 | 0.1949 | 0.7604 | 0.1163 m |

分层结果：

| 层 | precision | recall | FPR |
| --- | ---: | ---: | ---: |
| dense / interaction | 0.7716 | 0.9184 | 0.1940 |
| dense / weak | 0.6708 | 0.9159 | 0.2426 |
| standard / interaction | 0.6361 | 0.8479 | 0.1737 |
| standard / weak | 0.5590 | 0.8694 | 0.1807 |

准入要求是 precision `>=0.70`、recall `>=0.90`、FPR `<=0.10`，因此未通过。

## 已排除的原因

| 检查 | 结果 | 结论 |
| --- | --- | --- |
| 去掉中心回归 loss | F1 0.7600，FPR 0.2021 | 不是多任务 loss 干扰 |
| 纯分类延长到 80 epoch | train loss 0.52 降到 0.12，validation 无提升 | 不是训练不够，继续训练只会过拟合 |
| 中心裁剪到约 0.9 m | F1 0.7633，FPR 0.1990 | 邻近目标污染存在，但裁剪不能根治 |
| 负候选邻近已标注机器人审计 | 191 个在 0.6 m 内，其中 127 个误报 | 点簇拆分造成少量标签噪声，只解释约 6% 误报 |

失败样本显示，聚类器看到的是短墙段、墙角和箱体局部，而不是完整静态物体。VLP-16 的稀疏单帧局部形状与机器人外壳存在真实重叠；一个窗口内还可能同时出现中心静态候选和旁边机器人。单帧 CNN 可以提供软形状证据，但不能承担可靠的语义硬判定。

## 决策

1. 不扩大 G0 网络，不继续增加 epoch，不读取 sealed test。
2. 暂不采集 7200+900 场正式 G0 数据；先验证多帧跟踪是否能把 FPR 降下来。
3. G1 使用本机位姿补偿自运动，只从 LiDAR 候选计算轨迹持续性、相对速度、闭合速度、CPA 和 TTC。
4. 仿真中的机器人身份只用于离线评估，不进入部署输入。
