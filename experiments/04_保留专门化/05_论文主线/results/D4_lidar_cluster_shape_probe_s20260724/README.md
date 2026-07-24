# D4 3D Lidar Cluster Shape Probe

状态：`rejected feature candidate`。三维高度/尺寸特征优于单纯二维质心，但仍不能可靠区分机器人点簇与静态环境点簇，不接入Actor。

## 协议

- policy：冻结5D Actor，不训练导航网络。
- scenarios：固定 `sensor_probe.json.gz`，deep/close/margin各10场。
- deployable input：本机前视16线Velodyne XYZ点云、本机odometry和时间戳。
- label：仅离线使用Gazebo中其他机器人位置，将质心距离真实机器人不超过`0.60 m`的点簇标为robot。
- features：XY直径/长宽比、点数、Z最小/最大/均值/标准差/跨度。
- split：按risk band和standard/dense分组后交替划分，18个scenario校准、12个scenario独立评估。
- classifier：先审计单特征阈值，再训练类别平衡的8维逻辑回归；阈值只在校准集选择。
- 准入：独立评估precision `>=0.70`、recall `>=0.90`、FPR `<=0.10`。

## 结果

最佳单特征是`xy_diameter <= 0.5504 m`，独立评估：

| Precision | Recall | FPR |
| ---: | ---: | ---: |
| 0.485 | 0.988 | 0.660 |

8维逻辑回归在校准集选择`recall >= 0.90`下最低FPR工作点，独立评估：

| Precision | Recall | FPR |
| ---: | ---: | ---: |
| 0.651 | 0.912 | 0.307 |

三维特征可以保留大部分机器人点簇，但仍会接受大量尺寸和高度相近的箱体、墙角及点云碎片。把它串到二维运动探针后只能降低已不足`0.80`的运动召回，无法满足最终TTC观测准入线。

## 决策

停止继续组合手工高度、尺寸和速度阈值。保留“可部署相对运动/TTC”问题，但下一候选应直接以连续激光帧为输入，在仿真中用privileged CPA/TTC标签监督一个时序编码器；推理时仍只读取本机传感器。该方向会改变方法结构，需作为正式方法决策后再实现。

归档保留完整XYZ轨迹、测试日志和校准/独立评估摘要：

```text
05af92a340f4ebc6100c45e1cb1eef7c3609234db600a44265aa467a80a89bd1  test.log
44c49988e9f00549cfbeb18371c93c15347193cbf3c8006b996806d54aad453d  trajectory.jsonl.gz
```
