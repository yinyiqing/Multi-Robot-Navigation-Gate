# D4 Dense Validation Actor Comparison

状态：`complete`。目标是在同一份固定 `dense/validation` 清单上区分并比较：

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

## 结果

| 方法 | episodes | agent success | collision | unresolved | full success | timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5A 独立 | 1000 | `3532/5000 = 0.7064` | `1465/5000 = 0.2930` | `3/5000 = 0.0006` | `309/1000 = 0.3090` | `3/1000 = 0.0030` |
| 5D 独立 | 1000 | `3561/5000 = 0.7122` | `1436/5000 = 0.2872` | `3/5000 = 0.0006` | `314/1000 = 0.3140` | `3/1000 = 0.0030` |
| epoch-16 独立（提前停止） | 256 | `905/1280 = 0.7070` | `196/1280 = 0.1531` | `179/1280 = 0.1398` | `59/256 = 0.2305` | `132/256 = 0.5156` |
| 5A + epoch-16 oracle | 1000 | `4238/5000 = 0.8476` | `728/5000 = 0.1456` | `34/5000 = 0.0068` | `545/1000 = 0.5450` | `18/1000 = 0.0180` |

5D 相对 5A 的 agent/full success 仅高 `0.0058/0.0050`，应视为近似持平。oracle 组合将 full success 从 `0.3090` 提高到 `0.5450`，但 epoch-16 Actor 独立控制时将大量碰撞转成了停车超时，因此它只是条件局部避险策略，不是独立 Dense Actor。独立运行在256场后停止，因为 `51.56%` 的场景已超时，继续到1000场不会改变定性结论。

## 与140场交互验证的区别

`strong_interaction_curriculum_v1/validation` 的140场每场恰好只有 `1` 条名义冲突边，因此 `5A + epoch-16 oracle` 可达到 `full_success=0.7000`。完整 dense validation 的场景结构是：

| 数据集 | 场景数 | 平均冲突边 | 至少1条 | 至少2条 | 至少3条 |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong validation | 140 | `1.000` | `100.0%` | `0%` | `0%` |
| dense validation | 1000 | `2.457` | `95.8%` | `74.7%` | `45.3%` |

因此 `0.7000` 和 `0.5450` 不是同难度的成绩。前者验证单次局部避让，后者要求同一场中连续处理多组冲突，并且五5辆车全部到达。

## 已归档文件

- `5a.log.gz`：完整测试日志；
- `5a_runner.log.gz`：启动器日志；
- `5a_evaluation.npy`：逐 episode 结构化结果；
- `5a_state.pt`：完成状态；
- `5a_tensorboard.tfevents`：TensorBoard 指标。
- `5a_plus_epoch16_oracle.*`：完整1000场 oracle 组合日志、结构化结果和状态；
- `epoch16_standalone_partial256.*`：提前停止的256场独立Actor诊断。
