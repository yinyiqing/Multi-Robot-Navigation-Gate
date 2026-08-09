# 当前场景普通Actor重训

状态：`route revised / N1 original-width broad started`。
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
  base-stagnation/robot-proximity/yield-priority均关闭
- 固定步进：disabled，与R2-S0保持一致

通过条件：以120场validation为准，full success至少`0.85`，collision不高于`0.10`，
timeout不高于`0.10`，且动作无饱和、Q无明显爆炸。若N1连续两个eval恶化或出现动作/Q
异常，停止诊断，不进入N2。

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
```

运行日志统一写入：

`logs/active/current-generalist-r2style/n1/`
