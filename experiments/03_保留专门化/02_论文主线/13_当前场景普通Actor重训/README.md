# 当前场景普通Actor重训

状态：`route revised / N1-N5 original-width broad passed first segment / N5 admission failed timeout`。
更新时间：`2026-08-10`。

这条线不是旧 `5A` 的继续修补，而是为当前双 Actor + Gate 方法重新建立普通导航
Actor N。已有 R2 结果说明，干净课程和连续 Critic 比直接从五车 fresh Critic 启动更稳；
因此本路线从单车 broad 重新开始，逐步扩展到五车。当前N路线保持普通Actor角色；邻域
critic和强交互reward主要由条件避障Actor `epoch-16` 承载。

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
| N3 | 3 | `g12_r2_curriculum_v1/n3` broad | 原24维Critic | N2完整warm start | `20k` |
| N5 | 5 | `g12_r2_curriculum_v1/n5` broad，随后完整fixed-v1混合 | 与N3兼容 | N3完整warm start | `80k+` |

N1/N2先不使用local critic，目的是复现 R2 中最稳的基础导航课程，但保持原宽度
`24 -> 800 -> 600 -> 2`。local critic 不再像旧pilot那样在五车阶段突然 fresh 接入；
若未来用于普通Actor repair，必须在N5后另行登记并保护Actor；当前不在N3/N5 clean
课程中加入。

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
梯度，难以判断性能变化来源。另一个角色边界是：条件避障Actor `epoch-16` 已经使用
`87`维邻域critic、distance-weighted reward、robot proximity speed penalty和oracle
交互窗口训练；普通Actor N应优先保持普通推进能力，避免被训练成第二个避障专家。

首段通过条件沿用R2-S3：20k的agent success不低于`0.75`、full success不低于`0.55`，
collision不高于`0.20`、timeout不高于`0.12`；若20k相对10k full success下降至少
`0.10`，或timeout增加至少`0.10`，回滚10k。

## N3 结果

正式 N3 首段于`2026-08-10`完成`2 x 10k = 20k` agent samples。运行日志已归档到：

`logs/archive/training/current_generalist_r2style/n3/train_current_generalist_n3_original_broad_s20260810_20260810_131049.log`

| checkpoint | samples | agent success | full success | collision | unresolved | timeout | avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch 1 | 10k | `0.919` | `0.833` | `0.075` | `0.006` | `0.017` | `19.1` |
| epoch 2 | 20k | `0.953` | `0.892` | `0.036` | `0.011` | `0.025` | `22.9` |

Best checkpoint由训练脚本在epoch 2更新：

- Actor：`TD3/pytorch_models/current_generalist_n3_original_broad_s20260810_best_actor.pth`
  - SHA-256：`e35eb07cadff85dc29bf9f470ea7df91e9a5fc34e8bcb99c1bdea2c70b15fdcd`
- Critic：`TD3/pytorch_models/current_generalist_n3_original_broad_s20260810_best_critic.pth`
  - SHA-256：`c044abb91d514a552207168c3031f01dcfd2afb3f1d6200599b4e951af19c629`
- Full checkpoint：`TD3/checkpoints/current_generalist_n3_original_broad_s20260810_best.pt`
  - SHA-256：`f8f461100e3f94e20771a5cdadf26a003a3e3f617e47356ae3b1048973e73c1a`

结论：N3首段通过准入。20k相对10k的full success和agent success明显提高，collision
下降，但timeout和unresolved略升；总体仍满足准入。与G12-R2-S3加宽Actor同阶段相比，
当前原宽N3的20k full success略高（`0.892` vs `0.883`），但timeout/unresolved更高，
因此只能说三车阶段表现同级且约束仍需关注，不能单独宣称原宽优于加宽。

## N5 登记

- experiment：`current-generalist-r2style-N5`
- model：`current_generalist_n5_original_broad_s20260810`
- Actor：`24 -> 800 -> 600 -> 2`
- 初始化：从N3 best完整warm start Actor和Critic，不允许actor-only fallback
- train：`fixed_v1/views/g12_r2_curriculum_v1/n5/train.json.gz`
  - SHA-256：`82f990dab54331ef55d3818fbe39b31fe00480dd99696987a5b85c5e2581ac1e`
- validation：`fixed_v1/views/g12_r2_curriculum_v1/n5/validation.json.gz`
  - SHA-256：`e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7`
- seed：`20260816`
- 首段budget：`2 x 10k = 20k` agent samples
- eval：每10k做120场validation
- critic：原24维Critic，local critic disabled
- 关键超参：对齐G12-R2-S4，除Actor宽度和N3 warm start外，保持
  `batch=256`、`min replay=5000`、`gamma=0.999`、`actor/critic lr=8e-5`、
  `exploration 0.10 -> 0.03`、无随机直行动作、fixed physics `0.001`
- reward：individual navigation；dynamic/local/wall/safe-recovery/anti-stagnation/
  yield-priority均关闭；robot-proximity权重`5.0`

当前不启动 `N3-interaction-aware` 分支。理由是 `epoch-16` 已经是使用邻域critic、
distance-weighted reward和oracle交互窗口训练出的条件避障Actor；普通Actor N的第一任务
是保持普通推进、静态障碍和弱交互能力。若N5 clean出现明显碰撞主导失败，再登记
N5 repair，并在保护Actor的前提下讨论是否加入轻量local critic或交互reward。

首段通过条件沿用R2-S4：20k的agent success不低于`0.75`、full success不低于`0.50`，
collision不高于`0.22`、timeout不高于`0.10`；若20k相对10k full success下降至少
`0.10`，或timeout增加至少`0.10`，回滚10k。N5候选通过后，必须在同一冻结manifest上
与旧5A、B2、oracle和R2-10k做配对比较，不能直接把训练内validation汇总写成最终方法表。

## N5 结果

正式 N5 首段于`2026-08-10`完成`2 x 10k = 20k` agent samples。运行日志已归档到：

`logs/archive/training/current_generalist_r2style/n5/train_current_generalist_n5_original_broad_s20260810_20260810_152554.log`

日志头部确认：

- 从`current_generalist_n3_original_broad_s20260810_best`完整加载Actor和Critic；
- `Resume mode: False`，`Starting agent samples: 0`；
- `Actor hidden dimensions: 800x600`；
- `Local critic enabled: False`，`Distance-weighted reward: False`；
- `Device: cuda`，五车`r1-r5`。

| checkpoint | samples | agent success | full success | collision | unresolved | timeout | avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch 1 | 10k | `0.890` | `0.667` | `0.083` | `0.027` | `0.117` | `57.1` |
| epoch 2 | 20k | `0.920` | `0.717` | `0.062` | `0.018` | `0.092` | `50.6` |

Best checkpoint由训练脚本在epoch 2更新：

- Actor：`TD3/pytorch_models/current_generalist_n5_original_broad_s20260810_best_actor.pth`
  - SHA-256：`53964e12c2d6c5f0855530f22bdd721170b911640883c7616b14dc21aa12cfeb`
- Critic：`TD3/pytorch_models/current_generalist_n5_original_broad_s20260810_best_critic.pth`
  - SHA-256：`5c9d420ac4916d635774eaa9db32fcdbaaa7bf2bd55bf6779393783d571c9173`
- Full checkpoint：`TD3/checkpoints/current_generalist_n5_original_broad_s20260810_best.pt`
  - SHA-256：`2f2f3878334ecdd99b709719c466badd2d60eece22905804eadb06cb14dff883`

结论：N5首段通过预登记准入。20k相对10k的full success提高`0.050`、agent success提高
`0.030`、collision下降`0.021`、timeout下降`0.025`，因此选择epoch 2 best；没有回滚到
10k。需要注意的是，20k timeout仍为`0.092`、平均步数`50.6`，明显慢于G12-R2-S4大Actor
10k fallback的`0` timeout和`20.39`平均步数。因此N5 clean只能说已形成可评估的原宽五车
普通Actor候选，不能直接声称已经优于大Actor或可替换旧5A。

下一步必须做同场配对准入：在同一冻结manifest上评估旧5A、N5-20k、B2/最终Gate候选、
R2-10k和oracle，并按0-edge、edge-1、multi-edge分层。N5若要替换旧`generalist-5a`，
至少需要证明普通能力保持、碰撞下降不以系统性timeout为代价。

## N5 同场配对准入

正式 N5-20k admission 于`2026-08-10`完成。该实验复用G12-R2-N5 admission中已经审计
通过的旧5A和R2-10k逐场结果，只新增运行N5-20k；三者使用完全相同的120场manifest、
顺序、seed `20260817`、固定物理步长和单Actor执行模式。结果、manifest顺序、缺失与重复
ID审计均通过。日志已归档到：

`logs/archive/validation/current_generalist_n5_admission/`

逐场结果和summary位于：

`experiments/03_保留专门化/02_论文主线/13_当前场景普通Actor重训/local_data/n5_admission/`

整体结果：

| policy | agent success | full success | collision | unresolved | timeout | avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧5A | `0.810` | `0.558` | `0.190` | `0.000` | `0.000` | `20.08` |
| R2-10k大Actor | `0.900` | `0.700` | `0.100` | `0.000` | `0.000` | `20.39` |
| N5-20k原宽Actor | `0.910` | `0.700` | `0.075` | `0.015` | `0.067` | `46.15` |

拓扑分层：

| topology | policy | agent success | full success | collision | timeout | avg steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0-edge | 旧5A | `0.975` | `0.900` | `0.025` | `0.000` | `21.73` |
| 0-edge | R2-10k | `0.995` | `0.975` | `0.005` | `0.000` | `17.78` |
| 0-edge | N5-20k | `0.980` | `0.925` | `0.010` | `0.050` | `33.08` |
| edge-1 | 旧5A | `0.850` | `0.625` | `0.150` | `0.000` | `22.70` |
| edge-1 | R2-10k | `0.915` | `0.725` | `0.085` | `0.000` | `18.10` |
| edge-1 | N5-20k | `0.915` | `0.725` | `0.075` | `0.050` | `33.95` |
| multi-edge | 旧5A | `0.605` | `0.150` | `0.395` | `0.000` | `15.82` |
| multi-edge | R2-10k | `0.790` | `0.400` | `0.210` | `0.000` | `25.30` |
| multi-edge | N5-20k | `0.835` | `0.450` | `0.140` | `0.100` | `71.42` |

配对检验：

- N5-20k vs 旧5A：overall为`24`场改善、`7`场退化、`89`场持平，
  McNemar exact `p=0.00333`；multi-edge为`13/1`，`p=0.00183`。
- N5-20k vs R2-10k：overall为`13`场改善、`13`场退化、`94`场持平，
  McNemar exact `p=1.0`；full success无可分辨差异。

准入结论：N5-20k相对旧5A的成功率和碰撞改善是成立的，但严格准入失败。唯一失败项是
overall timeout：旧5A为`0/120`，N5-20k为`8/120=0.067`，超过`+0.02`上限。它也明显慢于
R2-10k，平均步数为`46.15` vs `20.39`。因此N5-20k不能直接替换旧`generalist-5a`作为
当前主方法中的普通Actor；它可以作为“原宽课程重训确实改善碰撞/复杂拓扑，但带来等待
和超时”的诊断证据。

## N5 timeout诊断

对N5-20k同场配对结果做逐场复核，8个timeout case的共同特征如下：

- 8/8均没有碰撞；
- 7/8为`4`个agent成功、`0`个碰撞、`1`个unresolved；
- 剩余1/8为`3`个agent成功、`0`个碰撞、`2`个unresolved；
- 5A在这8场中有`4/8` full success，R2-10k有`5/8` full success；
- N5在这8场中累计`31`个agent成功、`0`个碰撞、`9`个unresolved。

timeout case分布：

| layer | count |
| --- | ---: |
| dense | `5` |
| standard | `3` |
| 0-edge | `2` |
| edge-1 | `2` |
| multi-edge | `4` |

具体case：

| idx | scenario_id | layer | edge count | N5 outcome | 5A | R2-10k |
| ---: | --- | --- | ---: | --- | --- | --- |
| 17 | `dense-20260718008953-bfb89e8891bd` | dense multi | 3 | `3S/0C/2U` | fail, `3C` | fail, `2C` |
| 20 | `standard-20260717006788-679e7ea561c4` | standard zero | 0 | `4S/0C/1U` | full | full |
| 23 | `standard-20260717006980-d62c7937eb62` | standard zero | 0 | `4S/0C/1U` | full | full |
| 42 | `dense-20260718009026-2691c66d60ce` | dense multi | 3 | `4S/0C/1U` | full | full |
| 73 | `dense-20260718009239-ef66bec842b1` | dense edge1 | 1 | `4S/0C/1U` | fail, `2C` | full |
| 80 | `dense-20260718009333-38e4ac2b5b23` | dense multi | 3 | `4S/0C/1U` | fail, `3C` | fail, `2C` |
| 87 | `dense-20260718008564-5ea3f677f504` | dense multi | 4 | `4S/0C/1U` | fail, `1C` | fail, `2C` |
| 105 | `standard-20260717006984-13351d0e9eb0` | standard edge1 | 1 | `4S/0C/1U` | full | full |

结论：N5的失败不是“不会避障”，而是避障/等待之后缺少恢复推进。继续加入local critic或
interaction-aware reward大概率会强化安全保守性，不是当前最优修复方向。若继续修Actor，
应该登记为`N5-efficiency-repair`，目标是减少unresolved、timeout和平均步数；但主线更
合理的下一步仍是回到Gate，用旧5A或R2-10k承担快速普通推进，用epoch-16承担短时避障。

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

## N5 efficiency repair E1 登记

N5-20k同场配对准入失败的主因不是碰撞，而是timeout和平均步数过高。8个timeout case
全部无碰撞，主要表现为避让或等待后没有恢复推进。因此E1只做短程效率修复，不引入
local critic或dynamic reward，不改变普通Actor的输入结构。

- experiment：`current-generalist-r2style-N5-efficiency-E1`
- model：`current_generalist_n5_efficiency_e1_s20260810`
- 初始化：从`current_generalist_n5_original_broad_s20260810_best`完整warm start
  Actor和Critic
- Actor：`24 -> 800 -> 600 -> 2`
- train/validation：沿用N5冻结manifest，不更换场景集合
- seed：`20260818`
- budget：`2 x 5k = 10k` agent samples
- eval：每5k做120场validation
- critic：原24维Critic，`local critic disabled`
- reward改动：`timeout_reward=-120.0`，开启safe recovery；progress从`20.0`升至
  `25.0`，forward从`0.5`升至`0.8`，turn penalty从`0.2`降至`0.15`
- 稳定性保护：`actor lr=2e-5`，`critic lr=6e-5`，Actor延迟`3000` agent samples后更新
- fixed physics：`0.001`

E1准入不是追求单点最高full success，而是检验N5能否在不牺牲安全的前提下修掉慢和
timeout：

- timeout必须低于N5同场配对基线`0.067`；
- collision不应高于`0.10`；
- full success不应明显低于`0.700`；
- 平均步数应显著低于N5同场配对基线`46.15`。

若E1 full success提升但timeout或collision越界，不能作为普通Actor N替换候选；若timeout
下降但full success明显下降，也只能说明奖励过度催促，需要重新调小效率项。

## N5 efficiency repair E1 结果

正式E1于`2026-08-10`完成`2 x 5k = 10k` agent samples。运行日志已归档到：

`logs/archive/training/current_generalist_r2style/n5_efficiency_e1/train_current_generalist_n5_efficiency_e1_s20260810_20260810_200019.log`

日志头部确认：

- 从`current_generalist_n5_original_broad_s20260810_best`完整加载Actor和Critic；
- `Resume mode: False`，`Starting agent samples: 0`；
- `Actor hidden dimensions: 800x600`；
- `Local critic enabled: False`，`Critic state dim: 24`，`Distance-weighted reward: False`；
- `Timeout terminal reward: -120.0`，`Safe-recovery reward: True`；
- `Device: cuda`，fixed physics step size为`0.001`。

| checkpoint | samples | agent success | full success | collision | unresolved | timeout | avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch 1 | 5k | `0.893` | `0.675` | `0.102` | `0.005` | `0.025` | `29.9` |
| epoch 2 | 10k | `0.872` | `0.650` | `0.123` | `0.005` | `0.025` | `27.3` |

Best checkpoint由训练脚本在epoch 1更新：

- Actor：`TD3/pytorch_models/current_generalist_n5_efficiency_e1_s20260810_best_actor.pth`
  - SHA-256：`69545e0356813139a9c130ffcff4ff4f532975c527603e29d288031f9e1edfc1`
- Full checkpoint：`TD3/checkpoints/current_generalist_n5_efficiency_e1_s20260810_best.pt`
  - SHA-256：`9065ddd69cf8eefb7c9a522b3dbc98ab6ddfca1581736b31255f2c8381fe21b2`

结论：E1不通过准入，不能替换N5-20k或旧5A。它把timeout从N5同场基线`0.067`降到
`0.025`，平均步数从`46.15`降到`29.9/27.3`，说明“恢复推进/效率修复”的方向有效；
但collision从N5基线`0.075`升到`0.102/0.123`，且full success从`0.700`降到
`0.675/0.650`。因此当前奖励改动过度催促Actor，牺牲了安全裕度。若继续E2，应从E1
epoch 1或N5-20k重新出发，降低forward/progress增幅、恢复turn penalty或加入更明确的
碰撞保护，而不是继续epoch 2。

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
bash scripts/start_training_current_generalist_n5.sh
bash scripts/stop_training_current_generalist_n5.sh
bash scripts/start_training_current_generalist_n5_efficiency_e1.sh
bash scripts/stop_training_current_generalist_n5_efficiency_e1.sh
```

运行日志统一写入：

`logs/active/current-generalist-r2style/`
