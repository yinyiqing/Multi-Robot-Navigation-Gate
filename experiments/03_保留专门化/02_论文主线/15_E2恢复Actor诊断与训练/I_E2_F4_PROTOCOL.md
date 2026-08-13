# I-E2-F4 四阶段恢复 Actor Pilot

状态：`registered / short pilot only`。日期：`2026-08-13`。

## 目的

从冻结的 E2 普通 Actor 训练一个与 E2 匹配的条件恢复 Actor。它不是要求独立完成全程导航，
而是在 E2 的冲突窗口中减少碰撞、停滞和 timeout，并在恢复后把控制权交还给 E2。

冻结的大 Actor `R2-10k` 是完整导航对照，不是局部 Actor 的动作对照。最终比较必须在同一
N5 manifest 上比较完整闭环：`E2 + I-E2-F4` 是否超过 `R2-10k`，同时检查冲突窗口行为。

## 吸取的失败教训

1. 首版 I-E2 训练清单基本是 single-edge，导致 multi-edge 恢复能力没有训练覆盖。
2. I-E2 与 I-E2-M 都启用了 `actor_safety_focused`，Actor 梯度被限制在 `distance <= 1.0 m`
   且 `closing_speed > 0.1` 的逼近状态，排除了避让后的停滞、分离、恢复推进和 release。
3. I-E2-M 虽然覆盖了 multi-edge，但只经历 `422/2400` 个训练场景，且未形成有效恢复动作。
4. safe-recovery reward 在关键近车窗口被距离条件抑制，不能单独教会 Actor 解除僵持。
5. 之前没有先做短窗行为准入，就直接执行完整 40k，浪费了训练预算。

## F4 配置

- E2 warm start；Actor 部署输入保持 24 维。
- 训练清单：`ie2_multi_conflict_v1`，保留 edge-1/2/3+。
- Critic 使用 87 维 ego-motion 邻域上下文；动态 reward 保留。
- 只在 interaction window 收集和更新，但窗口内不再做 closing-risk 二次筛选，覆盖四阶段：
  `approach -> avoidance -> stalled recovery -> release`。
- safe-recovery reward 开启：progress bonus `0.8`，idle penalty `1.0`。
- 不再抑制 `1.2 m` 内的 safe-recovery 信号；否则关键近车恢复窗口仍然没有训练信号。
- Actor 前 `21k` samples 冻结，之后只做一个 `20k` 更新阶段；不自动延长。

## 准入

训练内部 200 场只用于选择 `20k` checkpoint，不是论文成绩。冻结后在 N5 120 场同场比较：

- E2 always-on；
- E2 + old epoch-16 recovery；
- E2 + I-E2-F4 recovery；
- R2-10k full Actor。

I-E2-F4 只有同时满足以下条件才进入 Gate 数据采集：

1. `E2 + I-E2-F4` full success 高于 E2，且不低于 R2-10k；
2. collision 不高于 E2，timeout 不高于 E2；
3. multi-edge 不低于 E2，且恢复窗口的 near-stop rate 低于 E2；
4. 不出现普通 zero-edge 的系统性退化；
5. 训练后离线动作审计显示它形成了恢复推进，而不是只改变角速度。

若不满足，停止该分支，不训练 Gate，也不追加 40k。
