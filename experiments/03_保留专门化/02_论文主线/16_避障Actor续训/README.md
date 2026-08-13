# 避障Actor epoch 17-20续训

状态：`registered / running pending`。日期：`2026-08-13`。

## 目的

当前避障Actor在预注册的16轮预算边界达到最高validation full success：epoch 16为
`0.7071`，此前epoch 11/13最高为`0.6429`。因此固定追加4轮、每轮20k agent samples，
检查收益是否继续，而不是因边界best直接宣称收敛。

以后正文统一称该组件为“避障Actor”；`epoch-16`只用于标识历史checkpoint。

## 恢复边界

- 源checkpoint：
  `TD3/checkpoints/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_latest.pt`
- 源checkpoint SHA-256：
  `251b9c2efd61e7caf39e8534089c889e5708b7fc8cb55ea9ebe345fc31ef1788`
- 恢复状态：epoch计数、Actor/Critic、target网络、两个优化器、320k replay、
  `balanced_cycle`游标和全部2560场覆盖记录。
- 新模型ID：`avoidance_actor_from_5a_balanced_continue_e20_s20260813`。
- 原避障Actor及其best、epoch 1-16快照保持不变。

## 固定协议

- train：`strong_interaction_curriculum_v1/full_train.json.gz`，SHA-256
  `d5b9b1fb968c8752e54e66f1ea3f25e7c2bf45eae3f012a686008704964da142`；
- validation：同视图140场，SHA-256
  `3b2646a842b777f8c60dca4c452cb78eb3a223ffe59139b8501797aa1d23d583`；
- seed继续使用`20260724`，采样继续使用`balanced_cycle`；
- 5A在2米窗口外执行，避障Actor在窗口内执行和更新；
- Actor保持24维，Critic保持87维ego-motion邻域上下文；
- reward、Critic ranking、安全聚焦Actor更新、学习率和探索参数全部保持原正式训练值；
- 当前代码新增的safe-recovery、yield-priority、anti-stagnation、local-navigation、
  wall-clearance和固定物理步进均显式关闭；
- 总上限为epoch 20，即只增加80k samples；不得自动继续到epoch 21。

## 选择与停止

原epoch 16继续作为冻结fallback。只有epoch 17-20在同一140场internal validation上的
full success严格超过`0.7071`，且collision、timeout和平均步数没有不可接受退化时，才
成为候选。训练结束后必须用独立matched validation比较：

```text
5A + 原避障Actor
5A + 新避障Actor候选
```

训练内部新高不能直接替换原避障Actor，也不是Gate成绩。epoch 20完成后无论结果如何都
停止，先做配对复测，不继续追逐边界峰值。

## 运行入口

```bash
bash scripts/start_avoidance_actor_e17_e20.sh
```

实时日志统一写入：

```text
logs/active/avoidance-actor-e17-e20/
```
