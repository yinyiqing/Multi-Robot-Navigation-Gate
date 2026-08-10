# 当前场景普通Actor重训

状态：`route revised / N1 and N2 original-width broad passed / N3 pending`。
更新时间：`2026-08-10`。

这条线不是旧 `5A` 的继续修补，而是为当前双 Actor + Gate 方法重新建立普通导航
Actor N。已有 R2 结果说明，干净课程和连续 Critic 比直接从五车 fresh Critic 启动更稳；
因此本路线从单车 broad 重新开始，逐步扩展到五车，并在三车阶段前后再引入 local critic。

## 研究问题

> 在保持原普通 Actor 参数量和 24 维部署观测的前提下，能否按 R2 证明有效的课程方式
> 训练出一个强于旧 `5A`、但仍保持普通推进效率的 Actor N？

## 训练原则

- 不直接从五车 fresh local critic 启动；先建立单车和少车普通导航底座；
- 训练不是只拿 `0/1` 冲突片段，而是逐步回到完整 `fixed_v1` 五车训练池；
- 冲突样本可以被强调，但不能替代完整导航分布；
- `epoch-16` 继续只负责局部冲突窗口，普通 Actor 仍要学会整段推进、静态障碍和一般交互；
- Actor 仍只读本车 24 维观测，不引入 gate 标签或 oracle 切换信号。

## 正式路线

| stage | agents | 数据 | critic | 初始化 | 预算 |
| --- | ---: | --- | --- | --- | ---: |
| N1 | 1 | `g12_r2_curriculum_v1/n1` broad | 原24维Critic | Actor/Critic随机 | `100k` |
| N2 | 2 | `g12_r2_curriculum_v1/n2` broad | 原24维Critic | N1完整warm start | `60k` |
| N3 | 3 | `g12_r2_curriculum_v1/n3` broad | local critic候选 | N2完整warm start；若切critic，先登记保护策略 | `60k` |
| N5 | 5 | `g12_r2_curriculum_v1/n5` broad，随后完整fixed-v1混合 | 与N3兼容 | N3完整warm start | `80k+` |

N1/N2先不使用local critic，目的是复现 R2 中最稳的基础导航课程，但保持原宽度
`24 -> 800 -> 600 -> 2`。local critic 不再像旧pilot那样在五车阶段突然 fresh 接入；
若使用，必须在 N3 登记并保证之后阶段结构兼容。

## N1 登记

- experiment：`current-generalist-r2style-N1`
- model：`current_generalist_n1_original_broad_s20260810`
- Actor：`24 -> 800 -> 600 -> 2`
- 初始化：随机 Actor + 随机原24维 Critic，不加载5A、不resume
- train：`fixed_v1/views/g12_r2_curriculum_v1/n1/train.json.gz`
  - SHA-256：`c71e4e87bbc528782cb76dc7df076c493900523bb748b3fb646f3d77fa5f0263`
- validation：`fixed_v1/views/g12_r2_curriculum_v1/n1/validation.json.gz`
  - SHA-256：`9ab4c5913f683d01e3ab186ea591d373abe1e835180f4a0bfeb469990269b125`
- seed：`20260811`
- budget：`5 x 20k = 100k` agent samples
- eval：每20k做120场validation
- 关键超参：`batch=256`、`min replay=5000`、`gamma=0.999`、
  `actor lr=1e-4`、`critic lr=1e-4`、`policy freq=2`、`tau=0.005`
- exploration：`0.35 -> 0.08` over 100k，前5000 samples随机线速度探索
- reward：individual navigation；dynamic/local/wall/safe-recovery/anti-stagnation/
  base-stagnation/yield-priority均关闭；robot-proximity权重保持R2-S0的`5.0`，但N1单车
  无机器人邻居，实际不产生该项奖励
- 固定步进：disabled，与R2-S0保持一致

通过条件：以120场validation为准，full success至少`0.85`，collision不高于`0.10`，
timeout不高于`0.10`，且动作无饱和、Q无明显爆炸。若N1连续两个eval恶化或出现动作/Q
异常，停止诊断，不进入N2。

## N1 结果

正式 N1 于`2026-08-10`完成`5 x 20k = 100k` agent samples。运行日志已归档到：

`logs/archive/training/current_generalist_r2style/n1/train_current_generalist_n1_original_broad_s20260810_20260810_003324.log`

| epoch | samples | full success | collision | timeout | avg steps | avg final distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20k | `0.975` | `0.025` | `0.000` | `9.1` | `0.240` |
| 2 | 40k | `1.000` | `0.000` | `0.000` | `9.9` | `0.243` |
| 3 | 60k | `1.000` | `0.000` | `0.000` | `8.4` | `0.227` |
| 4 | 80k | `1.000` | `0.000` | `0.000` | `8.1` | `0.218` |
| 5 | 100k | `1.000` | `0.000` | `0.000` | `8.6` | `0.221` |

Best checkpoint由训练脚本在epoch 3更新：

- Actor：`TD3/pytorch_models/current_generalist_n1_original_broad_s20260810_best_actor.pth`
  - SHA-256：`bb1fb7b20132c1bccc63f557d27aaaf43b3d3bf8910c2f4116821720b1bf43e2`
- Critic：`TD3/pytorch_models/current_generalist_n1_original_broad_s20260810_best_critic.pth`
  - SHA-256：`002c299122020151e4e51dc8d52b5dc5240d70edbdfb62c5a594243ecd06fd50`
- Full checkpoint：`TD3/checkpoints/current_generalist_n1_original_broad_s20260810_best.pt`
  - SHA-256：`6ac490aa35d5ba43d1b18b8d6126c0609e81a85f55b153654fcd0191a26bbf3e`

结论：原宽度普通Actor从随机初始化开始可以稳定学会单车broad导航，N1通过进入N2的
准入。这只回答单车基础导航是否成立，不证明两车、五车或冲突能力。

## N2 登记

- experiment：`current-generalist-r2style-N2`
- model：`current_generalist_n2_original_broad_s20260810`
- Actor：`24 -> 800 -> 600 -> 2`
- 初始化：从N1 best完整warm start Actor和Critic，不允许actor-only fallback
- train：`fixed_v1/views/g12_r2_curriculum_v1/n2/train.json.gz`
  - SHA-256：`5fbd2df5241076041ea714b59286604915ebf1b13848482f7c34fd10cdc9087b`
- validation：`fixed_v1/views/g12_r2_curriculum_v1/n2/validation.json.gz`
  - SHA-256：`955132263cac9496a56eb8bb6f5132ca5ae41e930c926a7a9a13e8797bb903c9`
- seed：`20260814`
- 首段budget：`2 x 10k = 20k` agent samples
- eval：每10k做120场validation
- critic：原24维Critic，local critic disabled
- 关键超参：对齐G12-R2-S2，除Actor宽度和N1 warm start外，保持
  `batch=256`、`min replay=5000`、`gamma=0.999`、`actor/critic lr=8e-5`、
  `exploration 0.10 -> 0.03`、无随机直行动作、fixed physics `0.001`
- reward：individual navigation；dynamic/local/wall/safe-recovery/anti-stagnation/
  yield-priority均关闭；robot-proximity权重`5.0`

首段通过条件沿用R2-S2：20k的agent success不低于`0.80`、full success不低于`0.65`，
collision不高于`0.15`、timeout不高于`0.10`；若20k相对10k full success下降至少
`0.10`，或timeout增加至少`0.10`，回滚10k。通过后再登记剩余40k，不直接长跑。

## N2 结果

正式 N2 首段于`2026-08-10`完成`2 x 10k = 20k` agent samples。运行日志已归档到：

`logs/archive/training/current_generalist_r2style/n2/train_current_generalist_n2_original_broad_s20260810_20260810_101251.log`

| checkpoint | samples | agent success | full success | collision | unresolved | timeout | avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch 1 | 10k | `0.950` | `0.925` | `0.046` | `0.004` | `0.008` | `13.3` |
| epoch 2 | 20k | `0.950` | `0.925` | `0.046` | `0.004` | `0.008` | `11.9` |

Best checkpoint由训练脚本在epoch 2更新：

- Actor：`TD3/pytorch_models/current_generalist_n2_original_broad_s20260810_best_actor.pth`
  - SHA-256：`55917aea9176fe28a081b43d039f402d0518c6836c194794f4bf3989dd0812bd`
- Critic：`TD3/pytorch_models/current_generalist_n2_original_broad_s20260810_best_critic.pth`
  - SHA-256：`e5fbaf2ac7b0fbafabef1f25df38fa75d32d9ffc5c9177d37bb1c4f40fc0d3e5`
- Full checkpoint：`TD3/checkpoints/current_generalist_n2_original_broad_s20260810_best.pt`
  - SHA-256：`29b768de697d06d82a86839aa1e2cac90ff20a4cb70fe5d2483d2308b686dfee`

结论：N2首段通过准入。原宽度Actor从N1 best完整warm start后，能稳定进入两车broad
导航；20k相对10k没有full success退化，平均步数下降。该结果与G12-R2-S2加宽Actor
同阶段结果接近，但n2 validation只有约`17/120`个派生冲突场景，因此仍只证明两车完整
broad导航稳定，不证明冲突能力或五车能力。

## N3 登记

- experiment：`current-generalist-r2style-N3`
- model：`current_generalist_n3_original_broad_s20260810`
- Actor：`24 -> 800 -> 600 -> 2`
- 初始化：从N2 best完整warm start Actor和Critic，不允许actor-only fallback
- train：`fixed_v1/views/g12_r2_curriculum_v1/n3/train.json.gz`
  - SHA-256：`b6ff22964a8b1795a783f8af9360c123fae44b4b44a86de63e76a57b4a0b4422`
- validation：`fixed_v1/views/g12_r2_curriculum_v1/n3/validation.json.gz`
  - SHA-256：`f4b7d46fc488eb588007aa7ba72791545e750e691399da82c65d5cdf9f5938cc`
- seed：`20260815`
- 首段budget：`2 x 10k = 20k` agent samples
- eval：每10k做120场validation
- critic：原24维Critic，local critic disabled
- 关键超参：对齐G12-R2-S3，除Actor宽度和N2 warm start外，保持
  `batch=256`、`min replay=5000`、`gamma=0.999`、`actor/critic lr=8e-5`、
  `exploration 0.10 -> 0.03`、无随机直行动作、fixed physics `0.001`
- reward：individual navigation；dynamic/local/wall/safe-recovery/anti-stagnation/
  yield-priority均关闭；robot-proximity权重`5.0`

暂不在N3引入local critic，原因是N1/N2已经显示原宽度课程稳定，N3首段的主要问题是
能否复现R2-S3的三车扩展；此时加入local critic会同时改变输入结构、critic分布和Actor
梯度，难以判断性能变化来源。local critic是否进入应在N5或N5后的正式混合阶段另行登记。

首段通过条件沿用R2-S3：20k的agent success不低于`0.75`、full success不低于`0.55`，
collision不高于`0.20`、timeout不高于`0.12`；若20k相对10k full success下降至少
`0.10`，或timeout增加至少`0.10`，回滚10k。

## 已关闭的旧pilot

`2026-08-09` 的上一轮短跑日志：

`logs/archive/aborted/current_generalist_fullscene_local_critic_20260809/train_current_generalist_from_3d2_source_local_critic_s20260808_20260809_143056.log`

该运行不能作为正式结果使用，原因有两点：

1. 日志显示 `Resumed multi-agent training from checkpoint`、`Resume mode: True`、
   `Starting agent samples: 11021`，说明它误接了旧 checkpoint；
2. 当时配置为 `Actor anchor weight: 0.0`、`Actor Q normalization alpha: 0.0`、
   `Actor update delay steps: 20000`，fresh critic 尚不稳定时就开始无约束更新 Actor。

这组数只说明“误 resume + fresh critic 无保护 Actor 更新”会快速退化，不能说明单边冲突
学不会、普通 Actor 路线失败或 local critic 本身无效。

`2026-08-09 21:57`启动的五车guarded fresh local critic pilot也已中止并归档，因为它
仍然是直接五车启动，不符合`2026-08-10`正式课程路线。日志位于：

`logs/archive/aborted/current_generalist_fullscene_local_critic_20260809/`

其停止时约`31.7k` agent samples，Actor尚未解冻，只能作为旧路线中止记录。

`2026-08-10 00:16`第一次N1启动在日志头部发现`Base stagnation penalty weight=0.03`，
与R2-S0配置不一致，已在episode 1后立即停止并归档；未产生checkpoint。日志位于：

`logs/archive/aborted/current_generalist_r2style_n1_20260810_preflight/`

`2026-08-10 00:18`第二次N1启动后，前17个episode为`0/17`，而R2-S0同期前20个episode
为`6/20`。复核发现除Actor宽度外还存在两个不必要差异：seed为`20260810`而非R2-S0的
`20260811`，且当前强制`Fixed physics step size=0.001`而R2-S0为disabled。因此该运行
中止并归档，只保留为启动诊断。日志位于：

`logs/archive/aborted/current_generalist_r2style_n1_20260810_seed_physics_mismatch/`

后续N1正式运行必须对齐R2-S0的seed和physics，使核心差异只剩Actor宽度
`800x600` vs `1137x855`。

`2026-08-10 00:31`第三次N1 preflight已确认`Seed=20260811`和`Fixed physics=disabled`，
但日志中`Robot proximity penalty weight=0.0`仍与R2-S0的`5.0`不一致。虽然单车场景无
机器人邻居，该项理论上不影响reward，但为了日志条件也完全对齐，已在episode 3后停止并
归档。日志位于：

`logs/archive/aborted/current_generalist_r2style_n1_20260810_proximity_preflight/`

## 旧五车guarded配置检查项

旧五车guarded路线若未来用于诊断，启动后必须检查日志头部同时满足：

- `Resume mode: False`
- `Starting agent samples: 0`
- `Critic is newly initialized because actor-only warm start was requested.`
- `Actor update delay steps: 40000`
- `Actor anchor weight: 0.05`
- `Actor Q normalization alpha: 1.0`
- `Batch size: 256`
- `Minimum replay size: 5000`
- `Discount: 0.999`
- `Fixed physics step size: 0.001`
- `Robot safe distance: 0.0`

## 判断标准

1. N1只回答基础导航是否成立，不与五车B2/R2直接比较；
2. N2/N3只作为进入五车的稳定性门槛；
3. N5候选必须在同一冻结manifest上与旧5A、B2、oracle和R2-10k比较；
4. 不读取 sealed test；
5. 重点看 `full success / collision / timeout / 平均步数`，不要只盯单点峰值。

## 启动脚本

```bash
bash scripts/start_training_current_generalist_n1.sh
bash scripts/stop_training_current_generalist_n1.sh
bash scripts/start_training_current_generalist_n2.sh
bash scripts/stop_training_current_generalist_n2.sh
bash scripts/start_training_current_generalist_n3.sh
bash scripts/stop_training_current_generalist_n3.sh
```

运行日志统一写入：

`logs/active/current-generalist-r2style/`
