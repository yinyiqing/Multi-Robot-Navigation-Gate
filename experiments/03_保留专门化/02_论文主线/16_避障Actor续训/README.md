# 避障Actor epoch 17-20续训

状态：`completed / epoch 17 candidate pending matched admission`。日期：`2026-08-14`。

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

完成后的归档日志：

```text
logs/archive/training/avoidance_actor_e17_e20/
```

## 训练结果

| epoch | full success | agent success | collision | unresolved | timeout | 平均步数 | 避障Actor占比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16（原模型） | `0.7071` | `0.9171` | `0.0786` | `0.0043` | `0.0143` | `54.5` | `53.2%` |
| 17 | **`0.7429`** | **`0.9214`** | `0.0657` | `0.0129` | `0.0357` | `57.9` | `56.3%` |
| 18 | `0.6500` | `0.9071` | `0.0757` | `0.0171` | `0.0429` | `64.1` | `61.7%` |
| 19 | `0.6857` | `0.9143` | `0.0671` | `0.0186` | `0.0500` | `73.6` | `63.4%` |
| 20 | `0.7071` | `0.9186` | **`0.0557`** | `0.0257` | `0.0643` | `87.2` | `69.6%` |

epoch 17是唯一超过原epoch 16 full success的续训点，并降低collision，但timeout增加
`0.0214`、平均步数增加`3.4`，所以只保留为候选。epoch 18-20没有延续成功率收益，且
避障Actor占比、unresolved、timeout和平均步数总体持续上升，表明继续TD3更新逐渐把策略
推向过度保守。epoch 18-20拒绝，不进入matched复测；不继续epoch 21。

自动best为epoch 17，Actor SHA-256：
`149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5`。它仍只是internal
validation候选，下一步必须按本文预注册方式与原epoch 16做独立、同场matched admission；
在该复测完成前，当前避障Actor仍为原epoch 16。
