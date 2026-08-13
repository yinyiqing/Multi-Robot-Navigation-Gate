# 新避障Actor重训

状态：`superseded diagnostic / completed but rejected`.

这条线是把原来 `5A -> local critic` 的 Actor B 训练模板，平移到新普通 Actor
`current_generalist_n5_efficiency_e2_s20260810_best` 上。原始 `5A` 实验不改，新的
模型名前缀、PID 和日志目录都单独分开。

该入口后来实际完成了4个epoch，但它是全场local-Critic更新，不是条件避障Actor协议：

- epoch 1/2/3/4 full success为`0.717/0.667/0.692/0.692`；
- dynamic reward关闭；
- oracle interaction rollout关闭；
- interaction-only Actor更新关闭；
- best未超过E2单独的`0.750`。

因此该实验只证明“给E2直接接fresh local Critic并全场更新”没有增益，不作为新避障
Actor。新的I-E2协议已转入
[15_E2恢复Actor诊断与训练](../15_E2恢复Actor诊断与训练/README.md)。旧入口仅供复核：

```bash
bash scripts/start_training_current_generalist_from_e2_local_critic.sh
bash scripts/stop_training_current_generalist_from_e2_local_critic.sh
```

默认 warm start：

- ordinary Actor：`current_generalist_n5_efficiency_e2_s20260810_best`
- local critic：开启
- train/eval manifest：沿用原 local-critic 模板

结果位于独立的`local_data/`，运行日志已归档到
`logs/archive/diagnostic/current_generalist_from_e2_local_critic_20260811/`。
