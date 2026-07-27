# D5-G2 Interaction Gate V1

状态：`G2-A pilot完成但未通过准入线；G2-B/G2-C暂停；两个Actor冻结；sealed test封存`。

完整结果见 [PILOT_REPORT.md](PILOT_REPORT.md)。前方 180 度 Gate 的最佳 validation recall/FPR 为 `0.861/0.111`，standard/0-edge FPR 为 `0.070`；在两项 FPR 都不超过 `0.10` 时，最高 recall 只有 `0.845`。因此当前输入不能按预定标准复现 `2.0 m` Oracle。

## 目标

Gate 最终回答的是“当前状态应该调用哪个冻结 Actor”，不是单独回答“眼前物体是不是机器人”。开发分成四步，禁止跳过可观测性检查直接做端到端调参。

## G2-A：可部署 Oracle 模仿

目的：确认本机观测能否复现已经验证有效的执行契约。

- 教师标签A：360度最近活动机器人真值距离 `<=2.0 m`，严格复现原训练Oracle。
- 教师标签B：前方180度最近活动机器人真值距离 `<=2.0 m`，检查当前传感器下的可观测上限。
- 部署输入：24维 Actor 状态、G0形状分数、G1轨迹年龄/速度/闭合速度/CPA/TTC。
- 禁止输入：Gazebo机器人位置、scenario ID、standard/dense、interaction band。
- 两个 Actor 只做冻结前向推理，不更新参数。

`2.0 m` 真值只在仿真中生成训练标签。G2-A 不是最终方法，而是确认“已证明有效的 Oracle 切换能否由本机传感器近似”。两个标签必须同时报告，不能用前方标签的较高成绩冒充完整Oracle复现。

初始准入线：

| 指标 | 要求 |
| --- | ---: |
| oracle-positive frame recall | `>=0.90` |
| oracle-negative frame FPR | `<=0.10` |
| standard / 0-edge false activation | `<=0.10` |

阈值和滞回参数只由 validation 选择，sealed test 不参与。

## G2-B：冻结 Actor 反事实标签

G2-A 通过后，才从固定训练场景的同一仿真状态分别执行两个冻结 Actor 的短期分支。其他机器人、目标和初始物理状态保持一致。

标签优先级：

1. 避免碰撞；
2. 提高机器人间最小净空；
3. 保持目标进展；
4. 减少停滞和无意义转向。

只有两个分支差异超过预先固定的 margin 时才生成 Actor-choice 标签；不明确状态保持 5A。该阶段解决“附近有机器人”不等于“强 Actor 一定更好”的问题。

## G2-C：最终 Gate

- 小型二分类器输出强 Actor 概率；
- 默认使用 5A；
- 使用不同的 switch-on / switch-off 阈值和最短保持时间抑制抖动；
- 可以用 G2-A 权重初始化，再用 G2-B 标签训练；
- 不更新两个 Actor，不改变 TD3 Actor 结构。

## G2-D：端到端 Validation

在相同固定 validation 场景上比较：

1. 5A 全程运行；
2. `2.0 m` 真值 Oracle；
3. 最小 LiDAR 距离规则；
4. G2-A Oracle 模仿 Gate；
5. G2-C Actor-choice Gate。

必须报告 agent/full success、collision、timeout、平均步数、强 Actor 激活比例、切换次数和 standard/0-edge 误激活率。只有学习 Gate 明显超过可部署距离规则并接近 Oracle，才进入正式多 seed 和 sealed test。

## 当前执行顺序

1. 扩展 v3 shard，逐帧保存24维状态和 privileged教师标签。
2. 固定采集 G2-A train/validation pilot。
3. 训练小型 Oracle 模仿 Gate并检查准入线。
4. 通过后实现可复现的短期仿真分支，不提前做 G2-B。

## G2-A 采集入口

先各跑一场验证 v3 shard，再扩大 pilot：

```bash
DRL_ROBOT_PERCEPTION_TARGET_EPISODES=1 \
  bash scripts/experiment.sh start gate-interaction-pilot-train
DRL_ROBOT_PERCEPTION_TARGET_EPISODES=1 \
  bash scripts/experiment.sh start gate-interaction-pilot-validation
```

正式 pilot 使用默认的完整 100 场 manifest。训练命令：

```bash
source env.python.sh
python scripts/train_interaction_gate.py \
  --train-dir experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G2_interaction_gate_v1/local_data/shards/pilot_train \
  --validation-dir experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G2_interaction_gate_v1/local_data/shards/pilot_validation \
  --detector-checkpoint experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt \
  --output-dir experiments/03_保留专门化/02_论文主线/results/06_Gate开发/D5_G2_interaction_gate_v1/local_data/model/oracle_front_v1 \
  --label front --epochs 40
```
