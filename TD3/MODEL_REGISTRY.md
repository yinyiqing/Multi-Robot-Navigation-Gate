# 模型注册表

更新时间：`2026-08-04`。代码继续使用历史artifact文件名以保持兼容；论文和新文档统一
使用本表短ID。当前方法见[PROJECT_STATUS](../PROJECT_STATUS.md)。

## 当前模型

| 模型ID | 角色 | artifact前缀 | 状态 |
| --- | --- | --- | --- |
| `generalist-5a` | 普通导航Actor N | `TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best` | frozen current |
| `interaction-epoch16` | 条件避障Actor I | `interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016` | frozen current |
| `learned-gate-g2a` | 第一版可部署Gate | G2-A checkpoint，见Gate结果README | frozen baseline; admission failed |
| `deployable-interaction-gate` | 当前待训练Gate | 新协议冻结后命名 | current pending |
| `gate-robot-perception` | G0/G1形状与相对运动前端 | G0/G1 pilot artifacts | frozen frontend baseline |

`interaction-teacher-epoch16`是历史文档中的同义ID，不表示它只用于Residual教师。当前
方法直接把该模型作为局部条件Actor调用。

## 对照与关闭模型

| 模型ID | artifact前缀 | 状态 | 用途限制 |
| --- | --- | --- | --- |
| `weak-interaction-5d` | `TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best` | baseline only | 历史普通导航对照 |
| `bridge-full-ft` | `TD3_multi_dense5_bridge_geo_critic_from_5d_best` | rejected | full fine-tune失败对照 |
| `bridge-head-only` | `TD3_multi_dense5_bridge_from_5d_head_only_best` | rejected | head-only失败对照 |
| `moderate-full-ft` | `TD3_multi_dense5_moderate_geo_critic_from_5d_best` | rejected | moderate覆盖训练失败对照 |
| `strong-interaction-s1` | `strong_interaction_curriculum_stage1_s20260723` | rejected | 危险加速，不作warm start |
| `actor-b-single-edge-residual` | `actor_b_from_epoch16_full_pilot_v1_s20260802`等历史候选 | rejected | R0/闭环路线均未准入 |
| `full-edge1-actor-pilot` | `full_actor_edge1_from_5a_s20260803` | rejected pilot | 解冻后退化；`best`主要对应冻结5A阶段 |
| `independent-dense-actor-*` | `independent_dense_actor_*` | rejected family | 不从局部峰值或`best`恢复训练 |

未列出的`TD3/checkpoints/*.pt`默认是历史训练恢复状态，不是当前模型。使用前必须在
[实验注册表](../experiments/EXPERIMENT_REGISTRY.md)确认其路线状态。

## 当前文件哈希

```text
fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5
  TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth

6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b
  TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth
```

## 文件含义

```text
TD3/pytorch_models/<prefix>_actor.pth   发布或评测Actor权重
TD3/pytorch_models/<prefix>_critic.pth  对应Critic权重
TD3/checkpoints/*.pt                    训练或测试恢复状态
TD3/results/*.npy                       本地统计快照
TD3/runs/*                              TensorBoard事件
```

只有通过正式协议并登记哈希的`pytorch_models`权重才能称为当前模型。`latest`、测试state、
TensorBoard目录和checkpoint文件名中的`best`都不能单独作为模型选择依据。

## 使用限制

- 5A和epoch-16当前冻结；不得在未修改论文协议时继续训练。
- epoch-16只能在运行时局部调用；其always-on结果只作失败边界。
- `2.0 m`真值距离只能做训练标签、诊断和oracle上界。
- 最终Gate推理只能使用本机传感器和本机导航状态。
- 若保留严格single-to-multi主张，需解决epoch-16训练集中11场full-horizon edge-2污染。
- 所有rejected模型只作失败对照，不得因checkpoint名含`best`重新成为候选。

## 新命名规则

新artifact使用：

```text
<method>_<scenario>_n<agents>_seed<seed>_<selection>
```

训练来源、commit、超参数、模型哈希和数据split写入实验manifest，不再编码进超长文件名。
历史模型不批量重命名，避免破坏日志和脚本引用。
