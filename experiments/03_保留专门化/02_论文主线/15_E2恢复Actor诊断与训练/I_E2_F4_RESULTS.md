# I-E2-F4 四阶段恢复 Actor 结果

状态：`completed / rejected`。日期：`2026-08-13`。

## 结论

F4 按固定 `40k` agent-sample 预算正常结束，但 Actor 解冻更新后整体退化，未通过准入：

| 阶段 | Actor状态 | full success | agent success | collision | unresolved | timeout | 平均步数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch 1 / 20k | 冻结，E2等价基线 | `0.630` | `0.885` | `0.094` | `0.021` | `0.090` | `54.64` |
| epoch 2 / 40k | 21k后更新 | `0.605` | `0.880` | `0.099` | `0.021` | `0.100` | `66.10` |

更新后的 full success 下降 `2.5` 个百分点，collision 增加 `0.5` 个百分点，timeout
增加 `1.0` 个百分点，平均步数增加 `11.47`。除 unresolved 持平外，所有关键指标均变差。
因此停止 F4，不追加训练，不用它采集 Gate 数据。

## 结果解释

本次已经修正 I-E2-M 的主要覆盖问题：训练清单包含 multi-edge，Actor 梯度覆盖完整
`2.0 m` interaction window，不再只筛选近距离闭合风险帧，并加入近车恢复奖励。训练末期
Actor gradient gate 持续通过，所以失败不是 Actor 没有更新。

结果说明，把更新范围扩展到整个交互窗口并增强恢复推进奖励，仍未形成比冻结 E2 更好的
条件专家。当前配置的 Actor 更新同时提高了碰撞、超时和运行时长，不能解释为单纯训练
时间不足。

## checkpoint边界

训练器按 full success 保留了 epoch 1 为 `best`。但 epoch 1 位于 Actor 解冻阈值之前，
行为上只是冻结的 E2 等价候选，不是训练出的避障 Actor：

- `best_actor`: `20eaf19fee881456ccf9ddad2b5fe642a515fb658d4d9ee2c30f90748072de3c`
- `epoch_001_actor`: `2300838b8c433076d591e1633f0571bffe3faf67bbe9d2e5c3cfc24fd72aac2e`
- `epoch_002_actor`: `0d557bf08f58d4bf5f4c1156d6d7bc277c9c2b26d632b4eb3469c25467f5a027`

文件哈希因保存时机和target网络状态不同而不完全相同，但不改变epoch 1期间Actor被冻结的
事实。不得把 `best` 后缀当作F4训练成功的证据。

## 流水线边界

流水线没有生成 `E2 + I-E2-F4` 的N5 120场末段结果。若按脚本继续，测试对象会是
`${model}_best`，也就是上述冻结阶段候选；这只会重测E2等价行为，不能评价F4更新后的
Actor。内部200场选择已经明确拒绝epoch 2，因此不再额外消耗N5评测时间。

同一流水线在训练前完成的matched控制仍是有效的新证据：

| 方法 | full success | collision | timeout | 平均步数 |
| --- | ---: | ---: | ---: | ---: |
| E2 matched | `0.6917` | `0.0817` | `0.0917` | `52.45` |
| E2 + old epoch-16 recovery | `0.7750` | `0.0550` | `0.0500` | `41.98` |

旧epoch-16 recovery相对E2为15场改善、5场退化，配对exact `p=0.04139`；multi-edge
full success为`0.600 vs 0.425`。这说明旧模型血缘不同虽需披露，但其与E2组合的实证
收益在matched评测中成立；它仍是唯一已经证明有效的条件恢复Actor。

## 证据位置

- 训练日志：`logs/archive/diagnostic/e2_ie2_f4_fourphase/`
- 训练结果：`TD3/results/interaction_recovery_from_e2_fourphase_s20260813.npy`
- matched摘要：`local_data/ie2_f4_fourphase_pilot/summary.json`
- 协议：[I_E2_F4_PROTOCOL.md](I_E2_F4_PROTOCOL.md)
