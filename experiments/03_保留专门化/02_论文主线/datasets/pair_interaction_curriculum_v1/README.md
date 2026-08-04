# Pair Interaction Curriculum v1

状态：`historical diagnostic / route closed`。

用途：历史上用于训练和验证保持原TD3结构的双车强交互Actor。该数据集只覆盖基础双车
冲突，不用于当前双Actor或Gate训练，也不代表论文test。

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
  experiments/03_保留专门化/02_论文主线/datasets/pair_interaction_curriculum_v1/train.json.gz \
  experiments/03_保留专门化/02_论文主线/datasets/pair_interaction_curriculum_v1/validation.json.gz
```

历史pilot使用完整5D Actor/Critic warm-start、原`24 -> 800 -> 600 -> 2` Actor和
`0.8 self + 0.2 neighbor` reward。该入口已关闭，数据只保留用于复现与场景诊断。
