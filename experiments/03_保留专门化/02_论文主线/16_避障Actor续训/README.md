# 避障Actor epoch 17-20续训

状态：`completed / epoch 17 rejected by matched admission`。日期：`2026-08-14`。

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
validation候选，随后按本文预注册方式与原epoch 16完成独立、同场matched admission。
该候选未通过效率准入，当前避障Actor仍为原epoch 16。

## 独立matched admission

状态：`completed / epoch 17 rejected`。

- manifest：`g12_full_scene_selection_v1/validation.json.gz`，120场，SHA-256
  `52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635`；
- 该集合与navigation train、G11-C/D2/E互斥，且没有用于选择epoch 17；
- 0-edge、edge-1、multi-edge各40场，standard/dense各60场；
- policy：`5A + 原epoch 16`与`5A + 候选epoch 17`，均使用相同2米真值oracle；
- seed：`20260814`、`20260815`，每个seed内共享场景ID、顺序、物理步进与终止条件；
- 总预算：`2 policies x 2 seeds x 120 = 480` episodes；顺序执行，禁止并发Gazebo；
- 该oracle只用于两个避障Actor的准入，不是可部署Gate结果。

候选epoch 17只有同时满足以下条件才替换原epoch 16：

1. 两个seed合并后full success严格更高，且逐场改善数大于退化数；
2. collision不高于原epoch 16；
3. timeout增加不超过`0.02`；
4. 平均步数不超过原epoch 16的`1.10x`；
5. 0-edge、edge-1或multi-edge中不得出现超过`0.05`的full-success退化。

启动入口：

```bash
bash scripts/start_avoidance_actor_matched_admission.sh
```

运行日志写入`logs/active/avoidance-actor-matched-admission/`，成功完成后自动归档到
`logs/archive/validation/avoidance_actor_matched_admission/`。

首轮执行中，原epoch 16的seed `20260814`已完成120场并通过结果审计。随后旧worker只按
端口终止Gazebo，没有先终止仍会respawn Gazebo的`roslaunch`，导致候选epoch 17的三次
启动均在episode 1前发生fixed-step传感器超时。三次启动不产生结果数组，全部作废并归档到
`logs/archive/rejected/avoidance_actor_matched_admission_cleanup_bug_20260814/`。worker已
改为先按本次动态launchfile终止`roslaunch`，再清理专用端口；恢复运行必须跳过已审计的
原epoch 16结果，从候选epoch 17的第1场重新开始。

修复后四组共`480/480`场完成，manifest顺序、唯一性和终止统计均通过审计。两个seed合并
结果如下：

| 避障Actor | full success | agent success | collision | unresolved | timeout | 平均步数 | 避障占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原epoch 16 | `0.6333` | `0.8875` | `0.1075` | `0.0050` | `0.0125` | `32.32` | `58.23%` |
| 候选epoch 17 | `0.6875` | `0.8983` | `0.0967` | `0.0050` | `0.0125` | `37.46` | `59.69%` |

epoch 17逐场改善`40`场、退化`27`场、持平`173`场，McNemar exact `p=0.1421`。按拓扑
看，0-edge/edge-1/multi-edge的full-success增量分别为`+0.0125/+0.1375/+0.0125`；
收益主要来自edge-1，multi-edge几乎没有改善。两个seed的full-success增量分别为`+0.1000`
和`+0.0083`，说明收益存在重复间波动。

候选通过成功率、改善/退化数、collision、timeout和拓扑退化五项检查，但平均步数为原
epoch 16的`1.159x`，超过预注册的`1.10x`上限，因此整体准入失败。epoch 17不得替换原
epoch 16，不追加Actor训练；当前避障Actor继续冻结为原epoch 16。结构化结果位于
`local_data/matched_admission/summary.json`，有效日志已归档到
`logs/archive/validation/avoidance_actor_matched_admission/`。
