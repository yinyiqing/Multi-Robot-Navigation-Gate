# D5-G1 Robot Tracking V1

状态：`跟踪器、v2 shard和20+20场pilot完成；保留连续运动特征，拒绝简单概率平滑硬分类`。

## 目的

G0 已证明单帧局部形状不足以可靠区分机器人与短墙段/箱体。G1 不改 TD3 Actor，也不把真值身份送入模型，而是在部署可得的信息上增加目标级时间证据。

## 输入

- G0 候选中心和机器人形状置信度；
- 本机里程计位姿与时间戳，用于消除本机运动造成的假速度；
- 当前帧局部 LiDAR 点簇几何。

## 输出

每条候选轨迹输出：

- 轨迹年龄和连续命中次数；
- 平滑后的 G0 形状置信度；
- 世界系速度和本机系相对速度；
- 闭合速度、最近接近距离、CPA 时间和 TTC；
- 是否具有足够的机器人交互证据。

Gate 后续读取这些可部署特征，不读取 Gazebo 模型位置。

## Pilot 协议

1. 扩展 shard，保存每个候选对应的本机位姿和采样时间。
2. 使用与 G0 相同且互不重叠的 pilot train/validation 场景重新采集。
3. G0 checkpoint 冻结，所有跟踪阈值只由 validation 选择。
4. 先比较单帧 G0 与 G0+G1 的 precision、recall、FPR，再决定是否开始正式数据采集。
5. sealed test 继续封存；两个 Actor 全程冻结。

结果与决策见 [PILOT_REPORT.md](PILOT_REPORT.md)。

原始运行日志已压缩归档在 `logs/`，包含一次单场 smoke、完整 train 和完整 validation 采集。

## 运行

先各跑一场检查 v2 shard，再扩大场景数：

```bash
DRL_ROBOT_PERCEPTION_TARGET_EPISODES=1 \
  bash scripts/experiment.sh start gate-robot-tracking-pilot-train
DRL_ROBOT_PERCEPTION_TARGET_EPISODES=1 \
  bash scripts/experiment.sh start gate-robot-tracking-pilot-validation
```

离线比较单帧和跟踪结果：

```bash
source env.python.sh
python scripts/evaluate_robot_tracking.py \
  --checkpoint experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt \
  --shard-dir experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G1_robot_tracking_v1/local_data/shards/pilot_validation
```
