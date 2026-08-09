# 当前场景普通Actor重训

状态：`guarded pilot ready / previous resumed run invalid`。
更新时间：`2026-08-09`。

这条线不是旧 `5A` 的继续修补，而是一个新的普通导航 baseline：
用 `5A` 的上游 warmstart 源模型、完整 `fixed_v1` 场景分布和 fresh local critic，
重新训练一个仍然只看本车 24 维观测的普通 Actor，检查旧 `5A` 是否主要卡在训练分布
和 critic 设计，而不是普通导航能力本身。

## 研究问题

> 在当前 frozen fixed-v1 全场景分布下，一个保持普通导航角色、但带 fresh local critic
> 的新 Actor，能否稳定超过旧 `5A`，并且不把自己训成半个 `epoch-16`？

## 训练原则

- 训练不是只拿 `0/1` 冲突片段，而是用完整 `fixed_v1` 训练池；
- 冲突样本可以被强调，但不能替代完整导航分布；
- `epoch-16` 继续只负责局部冲突窗口，普通 Actor 仍要学会整段推进、静态障碍和一般交互；
- Actor 仍只读本车 24 维观测，不引入 gate 标签或 oracle 切换信号。

## 协议

- Actor 初始化：`TD3_velodyne_multi_v4_curriculum_stage2_to_3d2_geo_critic_from_3a_guarded_best`
  的 actor-only warm start；
- Critic：fresh local critic；
- Actor 视图：仍只读本车 24 维观测；
- 训练场景：`fixed_v1/views/g12_r3_mixed_v1/train.json.gz`；
- 验证场景：`fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz`；
- 机器人数：5；
- local critic：开启，使用 `ego_motion` 上下文；
- resume：默认关闭，若同名 fresh checkpoint 已存在则拒绝启动；
- Actor 更新保护：默认先训练 fresh critic `40k` agent samples，再解冻 Actor；
- Actor 保守约束：默认 `Q normalization alpha=1.0`、warm-start anchor weight `0.05`、
  Actor gradient clip `1.0`；
- 目标：先跑短 pilot，判断它能否在不引入 gate 的情况下直接超过旧 `5A`。

## 已发现并修正的问题

`2026-08-09` 的上一轮短跑日志：

`logs/active/current-generalist-fullscene-local-critic/train_current_generalist_from_3d2_source_local_critic_s20260808_20260809_143056.log`

该运行不能作为正式 pilot 使用，原因有两点：

1. 日志显示 `Resumed multi-agent training from checkpoint`、`Resume mode: True`、
   `Starting agent samples: 11021`，说明它误接了旧 checkpoint；
2. 当时配置为 `Actor anchor weight: 0.0`、`Actor Q normalization alpha: 0.0`、
   `Actor update delay steps: 20000`，fresh critic 尚不稳定时就开始无约束更新 Actor。

这组数只说明“误 resume + fresh critic 无保护 Actor 更新”会快速退化，不能说明单边冲突
学不会、普通 Actor 路线失败或 local critic 本身无效。启动脚本已改为 guarded 配置，
下一次启动后必须检查日志头部同时满足：

- `Resume mode: False`
- `Starting agent samples: 0`
- `Critic is newly initialized because actor-only warm start was requested.`
- `Actor update delay steps: 40000`
- `Actor anchor weight: 0.05`
- `Actor Q normalization alpha: 1.0`

## 初始判断标准

1. `g12_full_scene_selection_v1` 的 120 场 validation 若不能稳定超过旧 `5A`，则只作为旧 `5A`
   偏弱的补充证据；
2. 若它能接近或超过参数匹配单 Actor，再考虑是否继续扩预算；
3. 不读取 sealed test；
4. 重点看 `full success / collision / timeout / 平均步数`，不要只盯单点峰值。

## 预期用法

- 先跑短 pilot，确认 fresh local critic 是否真的补上旧 `5A` 的结构缺口；
- 若普通 Actor 变得过于保守，再微调训练配比，而不是把训练收缩成 0/1 冲突专训；
- 这条线不替代当前 `5A + epoch-16 + Gate` 主方法，只是修正旧普通基线。

## 启动脚本

见 `scripts/start_training_current_generalist_from_5a_local_critic.sh`。
