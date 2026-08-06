# 数据集索引

数据目录保持稳定路径，不随结果归档改名。训练、validation 和 test 的角色以各数据集自己的 README 与 manifest 为准。

| 目录 | 状态 | 用途 | 是否用于当前正式方法 |
| --- | --- | --- | --- |
| [`fixed_v1/`](fixed_v1/README.md) | frozen | standard/dense 固定随机场景及互斥划分 | 是，主场景池与基线 |
| [`fixed_v1/views/edge1_full_horizon_v1/`](fixed_v1/views/edge1_full_horizon_v1/README.md) | frozen derived view | 完整路径复算后的纯single-edge train/validation | 是，当前Gate的single-edge来源 |
| [`fixed_v1/views/g11_a1_gate_v1/`](fixed_v1/views/g11_a1_gate_v1/README.md) | frozen derived view | 导航train内部的full-path 0-edge与edge-1互斥划分 | 是，G11-A1采集与G12-P1/R1诊断清单 |
| [`fixed_v1/views/g11_d2_admission_v1/`](fixed_v1/views/g11_d2_admission_v1/README.md) | frozen validation view | 排除旧开发场景的200场0-edge/edge-1独立准入 | 是，G11-D2闭环准入 |
| [`fixed_v1/views/g11_e_edge2_generalization_v1/`](fixed_v1/views/g11_e_edge2_generalization_v1/README.md) | frozen validation partitions | exact-edge-2前50场pilot与后150场confirmation | 是，G11-E冻结后泛化诊断 |
| [`fixed_v1/views/g12_full_scene_selection_v1/`](fixed_v1/views/g12_full_scene_selection_v1/README.md) | frozen internal validation | 与train及G11-C/D2/E互斥的120场完整场景分层选择集 | 是，G12-R2至R4模型选择 |
| `candidates_20260717/` | provenance | fixed-v1 筛选前候选清单 | 否，不直接训练或测试 |
| [`pilot/`](pilot/README.md) | pilot | 生成、筛选和 manifest 回放冒烟验证 | 否 |
| [`pair_interaction_curriculum_v1/`](pair_interaction_curriculum_v1/README.md) | historical diagnostic | 两车 head-on/crossing/lane-swap 诊断 | 否，不再继续双车路线 |

当前机器人感知基线使用 [`fixed_v1/views/robot_perception_v1/`](fixed_v1/views/robot_perception_v1/README.md)。它只从导航 train 内部重新划分 `7200/900/900` 个感知 train/validation/sealed-test 场景，不读取导航 validation/test。

`fixed_v1` 下的 test 在方法和阈值冻结前不得用于调参。运行时 Gate 不得读取场景池名称、冲突边或其他离线标签。

## 固定场景内容

正式 manifest 中每个场景完整保存五台机器人的起点、目标、初始朝向、四个箱子、生成 seed、静态可行性和同步冲突指标。正式数据按以下顺序产生：

```bash
source env.python.sh

python scripts/generate_fixed_scenarios.py \
  --preset dense \
  --output-dir /path/to/dense_candidates \
  --train 7200 --validation 1200 --test 2400 \
  --seed 20260717
```

上例先预留约 20% 候选。然后分别对每个 split 做 Gazebo reset 检查，并按正式目标数截取：

```bash
python scripts/validate_fixed_scenarios.py \
  --input /path/to/dense_candidates/test.json \
  --accepted /path/to/dense/test.json.gz \
  --rejected /path/to/dense/rejected_test.json \
  --target-count 2000
```

验证命令需要已经 source ROS、catkin workspace 和 `env.python.sh`。它只检查传感器、初始碰撞和复位误差，不加载 Actor。

回放固定 split：

```bash
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH=/path/to/dense/train.json.gz
export DRL_MULTI_MANIFEST_SAMPLING=random  # 训练；测试使用 cycle
```

当前Gate训练不得无筛选地读取全部dense train。场景必须从导航train内部按0-edge和
corrected full-horizon single-edge构建互斥视图；multi-edge只用于冻结后的泛化评估。
具体Gate manifest、哈希和scenario互斥报告必须在新实验协议中登记后才能启动。

上述限制只适用于Gate，不适用于G12正式参数匹配单Actor。G12-R2从头复现完整课程，
G12-R3/R4读取完整standard/dense train并重采样strong-interaction子集；它不使用
`g11_a1_gate_v1`作为正式训练边界。G12内部完整场景validation已冻结为
`g12_full_scene_selection_v1`，并完成与全部navigation train及G11-C/D2/E的互斥审计。
