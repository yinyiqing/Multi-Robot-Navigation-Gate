# Pair Interaction Curriculum v1

用途：训练和验证保持原 TD3 结构的强交互 Actor。该数据集只覆盖基础双车冲突，不代表最终论文 test。

## 固定划分

| split | head-on | crossing | lane-swap | total |
| --- | ---: | ---: | ---: | ---: |
| train | 30 | 30 | 30 | 90 |
| validation | 10 | 10 | 10 | 30 |

- `train.json.gz`、`validation.json.gz`：已通过 Gazebo reset 检查的固定训练与验证集。
- `*_candidates.json.gz`：程序化生成的原始候选集。
- `*_rejected.json`：筛选报告。本版本两组均为 0 场拒绝。
- train 和 validation 使用不同 seed，scenario ID、generation seed 互斥。

## 场景约束

- 两车起点间距至少 `1.2 m`，目标间距至少 `0.8 m`。
- 两条同步名义路径恰好形成一条冲突边，最小同步间距小于 `0.45 m`。
- 起终点和直线路径通过静态地图净空检查。
- Gazebo reset 后检查实际位置、激光数据、初始碰撞和初始终止。
- 筛选不运行任何策略，禁止依据 5D 或本文方法的成功/失败删除场景。

## 复现

```bash
source /opt/ros/noetic/setup.bash
source env.python.sh
python3 scripts/build_pair_interaction_curriculum.py

python3 scripts/audit_fixed_scenarios.py --num-agents 2 \
  experiments/04_保留专门化/05_论文主线/datasets/pair_interaction_curriculum_v1/train.json.gz \
  experiments/04_保留专门化/05_论文主线/datasets/pair_interaction_curriculum_v1/validation.json.gz
```

训练入口：

```bash
bash scripts/start_training_pair_interaction_pilot.sh
```

该 pilot 使用完整 5D Actor/Critic warm-start、原 `24 -> 800 -> 600 -> 2` Actor、`0.8 self + 0.2 neighbor` reward。仅将基础前进奖励和低速停滞惩罚权重设为零，不改变 goal、collision 和 progress reward。
