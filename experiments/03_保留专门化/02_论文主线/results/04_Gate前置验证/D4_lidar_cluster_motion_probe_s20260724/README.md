# D4 Deployable Lidar Cluster Motion Probe

状态：`rejected feature candidate`。原始前视点云支持相对运动/TTC方向，但二维点簇质心跟踪未达到接入Actor的准入标准。

## 问题

上一版扇区最小距离差分的 false-positive rate 为 `0.636/0.738`，无法稳定关联同一物体。本次改用实际可部署输入：

- 本机前视 Velodyne 点云；
- 本机 odometry；
- 时间戳。

Gazebo 中其他机器人轨迹只用于离线生成CPA/TTC真值，不进入特征或策略输入。

## 协议

- policy：冻结5D Actor，不训练任何网络。
- scenarios：固定 `sensor_probe.json.gz`，deep/close/margin 各10场，standard/dense各半。
- 点云：前方约180度、最大 `6 m`，二维 `5 cm` 体素降采样。
- tracking：紧凑点簇、ego-motion补偿、跨帧关联、最近5帧世界坐标质心线性回归速度。
- prediction：相对速度、closest point of approach和TTC。
- 准入：frame precision `>=0.70`、recall `>=0.80`、FPR `<=0.10`。

启动脚本最初误用了目标轮数变量，因此在人工停止前完成40场；分析器按scenario ID只使用每个固定场景第一次出现的30场，后10场循环重复不进入统计。

## 结果

默认物理参数下：

| Precision | Recall | FPR |
| ---: | ---: | ---: |
| 0.530 | 0.726 | 0.156 |

速度门槛扫描不存在合格工作点：门槛从 `0.35` 增至 `0.60 m/s` 时，precision 从 `0.630` 增至 `0.716`，但recall从 `0.674` 降至 `0.504`。完整扫描见 `operating_points.json`。

## 诊断

在530个真值危险agent-frame中：

- `97.92%` 的危险机器人附近存在原始点云；
- `97.55%` 成功形成点簇；
- `87.55%` 形成至少3帧的成熟轨迹。

因此原始传感器覆盖和二维聚类不是主要问题。误报帧中只有约 `19.1%` 的预测点簇靠近真实机器人，其余主要来自静态箱体、墙角或其碎片产生的质心抖动和错误关联。仅提高速度阈值会同时损害真目标召回。

结论是拒绝“二维点簇质心直接估速”，但保留“可部署相对运动/TTC”主线。下一候选必须增加目标身份或形状一致性，例如利用Velodyne三维高度轮廓进行机器人点簇筛选，再估计运动；在新的观测probe通过前不接Actor、不训练Gate。

归档保留完整日志、分析摘要、工作点扫描和gzip原始轨迹。

```text
8e0b530e2fdc005be5273b74ba69dcfb95271cfd85d5cbe28725b09b58acceb0  test.log
b182f39a860f955daaa2e48fe3b410b214616f6a2324b9ee545aac87f86370c2  trajectory.jsonl.gz
```
