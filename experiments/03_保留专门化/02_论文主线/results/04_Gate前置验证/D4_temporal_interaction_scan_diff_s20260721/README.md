# D4 Temporal Interaction Scan-Difference Probe

状态：`rejected feature family`。自运动补偿的激光扇区最小值差分能召回紧迫交互，但误报过高，不接入 Gate 或 Actor。

## 方法

1. 使用两帧本机里程计将上一帧扇区最近点投影到当前机器人坐标系。
2. 用补偿后上一帧距离减去当前距离，估计闭合速度和到 `0.35 m` 激光碰撞边界的 TTC。
3. 预测需连续 2 帧 TTC `<= 2 s`。
4. 评估真值使用其他机器人位置，定义为中心距 `<= 1.5 m` 且到 `0.9 m` 安全边界的 TTC `<= 2 s`。其他机器人位置只用于离线评估，不输入部署特征。

阈值和 persistence 在运行审计前已提交，没有根据结果反向调整。

## 结果

| Input | Episodes | Frame precision | Frame recall | Frame false-positive rate | Episode activation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor 20-bin lidar | 60 | 0.283 | 0.808 | 0.636 | 1.000 |
| Independent 180-bin lidar | 30 | 0.311 | 0.909 | 0.738 | 1.000 |

20-bin 和 180-bin 使用不同规模的诊断子集，因此不将两者差值解读为显著的分辨率效应。但两者都在所有 episode 激活，且 frame-level 误报均远高于可用范围，足以拒绝该特征族。

## 根因

每个扇区只保留一个最小距离，没有物体身份和精确点角度。即使完成自运动补偿，相邻帧的最小点也可能来自不同墙面、箱体边缘或机器人部位。这种扇区最小值跳变会产生虚假高闭合速度；增加扇区数不能补回丢失的数据关联。

## 决策

- 不将当前 `TemporalInteractionEncoder` 输出接入 Actor/Gate。
- 不继续调整扇区数、TTC 阈值、deadband 或 persistence。
- 下一个最小可证伪步骤是保留降采样后的原始二维点及精确角度，在自运动补偿后进行移动簇关联。
- 如果原始点簇仍无法将 frame false-positive rate 降至 `0.10` 以下，停止手工 TTC 路线，改用带评估真值的学习式时序编码。

`20bin_summary.json` 和 `180bin_summary.json` 保留结构化结果，原始轨迹和日志已 gzip 归档。
