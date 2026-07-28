# D4 Dense Validation Actor Comparison

状态：`running`。目标是在同一份固定 `dense/validation` 清单上区分并比较：

1. 5A 全程独立控制；
2. 5D 全程独立控制；
3. epoch-16 强交互 Actor 全程独立控制；
4. 5A 与 epoch-16 Actor 按 `2.0 m` 真值交互条件切换的 oracle 组合。

前三项用于比较独立 Actor 能力；第四项只保留为条件策略上限，不得当作独立 Actor 或已训练 Gate 的成绩。sealed test 未读取。

## 协议

- 场景：`datasets/fixed_v1/dense/validation.json.gz`，固定 1000 场，按清单顺序运行；
- 机器人数量：5；
- 5A：`TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best`；
- 5D：`TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best`；
- 强交互 Actor：`interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016`；
- 5A 与强交互 Actor 的本轮 seed：`20260728`；历史 5D 完整基线 seed：`20260719`。

## 当前结果

| 方法 | episodes | agent success | collision | unresolved | full success | timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5A 独立 | 1000 | `3532/5000 = 0.7064` | `1465/5000 = 0.2930` | `3/5000 = 0.0006` | `309/1000 = 0.3090` | `3/1000 = 0.0030` |
| 5D 独立 | 1000 | `3561/5000 = 0.7122` | `1436/5000 = 0.2872` | `3/5000 = 0.0006` | `314/1000 = 0.3140` | `3/1000 = 0.0030` |
| epoch-16 独立 | running | - | - | - | - | - |
| 5A + epoch-16 oracle | running | - | - | - | - | - |

5D 相对 5A 的 agent/full success 仅高 `0.0058/0.0050`，目前应视为近似持平，而不是有实质差距。待两条活跃验证完成后再补逐场配对、交互分层和最终结论。

## 已归档文件

- `5a.log.gz`：完整测试日志；
- `5a_runner.log.gz`：启动器日志；
- `5a_evaluation.npy`：逐 episode 结构化结果；
- `5a_state.pt`：完成状态；
- `5a_tensorboard.tfevents`：TensorBoard 指标。

两条活跃日志暂留根目录 `logs/`，完成后归入本目录。
