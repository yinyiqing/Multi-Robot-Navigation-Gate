# D5-G2 Interaction Gate V1

状态：`G2-A诊断完成；G2-B v1单次反事实标签已拒绝；两个Actor冻结；sealed test封存`。

完整结果见 [PILOT_REPORT.md](PILOT_REPORT.md)。前方 180 度 Gate 的最佳 validation recall/FPR 为 `0.861/0.111`，并明显优于最小 LiDAR 距离规则。该结果只用于证明本机观测包含交互信息，不再作为 G2-B 的硬准入门槛。

## 目标

Gate 最终回答的是“当前状态应该调用哪个冻结 Actor”，不是单独回答“眼前物体是不是机器人”。G2-A是诊断和辅助基线，G2-B才生成最终Actor选择监督；禁止把`2.0 m`分类准确率当成最终Gate目标。

## G2-A：可部署 Oracle 模仿

目的：确认本机观测能否复现已经验证有效的执行契约。

- 教师标签A：360度最近活动机器人真值距离 `<=2.0 m`，严格复现原训练Oracle。
- 教师标签B：前方180度最近活动机器人真值距离 `<=2.0 m`，检查当前传感器下的可观测上限。
- 部署输入：24维 Actor 状态、G0形状分数、G1轨迹年龄/速度/闭合速度/CPA/TTC。
- 禁止输入：Gazebo机器人位置、scenario ID、standard/dense、interaction band。
- 两个 Actor 只做冻结前向推理，不更新参数。

`2.0 m` 真值只在仿真中生成训练标签。G2-A 不是最终方法，而是确认“已证明有效的 Oracle 切换能否由本机传感器近似”。两个标签必须同时报告，不能用前方标签的较高成绩冒充完整Oracle复现。

诊断参考线：

| 指标 | 要求 |
| --- | ---: |
| oracle-positive frame recall | `>=0.90` |
| oracle-negative frame FPR | `<=0.10` |
| standard / 0-edge false activation | `<=0.10` |

阈值和滞回参数只由 validation 选择，sealed test 不参与。

## G2-B v1：冻结 Actor 单次反事实标签（已拒绝）

从固定训练场景的同一仿真状态分别执行两个冻结 Actor 的8步分支。该思路本身直接回答“哪个Actor更好”，但v1把一次带噪rollout当作确定标签，经重复性审计后被拒绝。完整证据见[PILOT_REPORT.md](PILOT_REPORT.md)。

标签优先级：

1. 避免本车碰撞，再减少受影响机器人碰撞数；
2. 本车到达目标；
3. 在不明显牺牲目标进展时提高本车最小车间距；
4. 在不明显损害车间距时提高本车目标进展。

短期分支第一版固定为8个环境步。净空margin为`0.10 m`，目标进展margin为`0.05 m`，允许的净空优先分支进展退化最多为`0.05 m`。只有差异超过margin才生成Actor-choice标签；不明确状态标记为ambiguous，执行时默认5A。正式采集前要求同一Actor重复分支的碰撞/到达一致，净空误差不超过`0.05 m`，进展误差不超过`0.03 m`。最终8步pilot未满足该要求，因此未扩大数据，也未训练Gate。

下一候选G2-B v2不再要求单次轨迹逐点相同，而是对两个Actor分别做多次独立带噪rollout，比较碰撞率、到达率、净空和进展的期望与置信区间。只有小规模pilot显示标签稳定且非ambiguous比例足够，才扩大采集。

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

1. G2-A完成，保留为可观测性诊断与距离规则基线。
2. G2-B v1完成重复性审计并拒绝，禁止扩大或据此训练Gate。
3. 设计G2-B v2多次rollout统计标签，先固定重复次数、置信区间和准入线。
4. 仅在小规模pilot通过后，才扩大train/validation并训练G2-C。

## G2-B Pilot入口

以下入口只用于复现v1重复性失败，默认只跑1场；不是Gate训练数据入口：

```bash
bash scripts/experiment.sh start gate-counterfactual-pilot-train
bash scripts/experiment.sh status
```

禁止用`DRL_G2B_TARGET_EPISODES`扩大v1。validation入口仅保留代码完整性，sealed test没有入口。

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
