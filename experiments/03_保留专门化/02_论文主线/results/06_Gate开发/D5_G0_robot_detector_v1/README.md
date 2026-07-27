# D5-G0 Robot Detector V1

状态：`代码、固定清单和单场景运行时冒烟完成；正式train/validation尚未采集，sealed test未读取`。

## 要解决的问题

Gate 不能把墙、箱子和其他机器人都当成同一种近距离障碍。G0 只回答一个问题：本机 VLP-16 当前看到的候选点簇是不是机器人。

这一步不训练 Gate，不修改 5A，也不修改 epoch-16 条件交互 Actor。

## 数据与输入

- 场景来自 [`robot_perception_v1`](../../../datasets/fixed_v1/views/robot_perception_v1/README.md)。
- 导航策略固定使用 5A，只负责产生实际运动轨迹。
- XYZ 点云投影为原生 `16 x 360` range view。
- 围绕二维点簇候选截取固定 `1.2 m` 物理宽度并重采样成 `16 x 64`。
- 模型输入为相对深度、高度和 valid mask 三个通道。
- 候选和 recall 的工作范围固定为本机 `<=4 m`；`4-6 m` 点云只作为窗口背景，不计为当前 Gate 的检测目标。
- Gazebo 中其他机器人位置只用于生成离线标签；模型输入和部署阶段不能读取它。
- 每个场景直接写一个压缩 `.npz` shard，不再保存巨大原始点云 JSONL。

每个 shard 同时保存 `visible_robot_count` 和 `missed_visible_robot_count`。因此最终 recall 的分母包括候选生成阶段漏掉的可见机器人，不会只在“已经找到的候选”上虚高。

## 准入线

阈值只由 validation 选择：

| 指标 | 要求 |
| --- | ---: |
| precision | `>= 0.70` |
| recall（含 proposal 漏检） | `>= 0.90` |
| 静态候选 FPR | `<= 0.10` |

G0 未同时满足三项时，停止 Gate 训练并继续改感知；不能用 test 选阈值。

## 运行

先采集 train，抽取一定数量后可并行采集 validation：

```bash
bash scripts/experiment.sh start gate-robot-perception-train
bash scripts/experiment.sh start gate-robot-perception-validation
bash scripts/experiment.sh status
```

需要先跑小批量时，在命令前设置场景数，例如：

```bash
DRL_ROBOT_PERCEPTION_TARGET_EPISODES=100 \
  bash scripts/experiment.sh start gate-robot-perception-train
```

训练：

```bash
source env.python.sh
python scripts/train_robot_detector.py \
  --train-dir experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/shards/train \
  --validation-dir experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/shards/validation \
  --output-dir experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model
```

`local_data/` 可重新生成且不进入 Git。test 评估入口已经实现，但只有模型、阈值和协议冻结后才允许使用。

## 运行时冒烟

2026-07-27 使用冻结 5A 在感知 train 的首个回放场景运行 1 次：

- 21 个采样帧；
- 263 个候选 patch；
- `51/51` 个 `<=4 m` 可见机器人形成正候选，proposal recall `1.0`；
- 单场景压缩 shard 为 `202398 bytes`；
- Gazebo、断点状态、真值标注和 `.npz` 写入链路均正常。

这是接口冒烟，不是分类器结果，也不能代替完整 validation 的 proposal recall。

## 下一步判断

1. 先用少量 train/validation 检查 proposal recall 和数据标签。
2. proposal recall 不足时先修候选生成，不能靠 CNN 补救。
3. proposal 足够但 FPR 高时，再比较局部 CNN 与 PointNet 点簇对照。
4. 单帧 G0 通过后，才进入 G1 跟踪和 TTC；不先上 GRU。

相关工作与选择依据见 [RELATED_WORK.md](RELATED_WORK.md)。
