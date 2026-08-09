# 当前场景普通Actor重训

状态：`draft / ready to run`。
更新时间：`2026-08-08`。

这条线不是旧 `5A` 的延续，而是一个新 baseline：
用当前冻结场景和一个 fresh local critic，重新训练一个普通导航 Actor，检查旧 `5A`
缺少临域 critic、且训练分布偏旧，是否正是性能落后的原因。

## 研究问题

> 在当前 frozen fixed-v1 场景分布下，一个仍然只输出单一控制策略、但带 local critic 的
> 新普通 Actor，能否明显抬高 5A 的 dense / multi-edge 表现？

## 协议

- Actor 初始化：冻结 `generalist-5a` 的 actor-only warm start；
- Critic：fresh local critic；
- Actor 视图：仍只读本车 24 维观测；
- 训练场景：`fixed_v1/views/g12_r3_mixed_v1/train.json.gz`；
- 验证场景：`fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz`；
- 机器人数：5；
- local critic：开启，使用 `ego_motion` 上下文；
- 目标：先做短 pilot，观察它能否在不引入 gate 的情况下直接超过旧 5A。

## 初始判断标准

1. 如果对 `g12_full_scene_selection_v1` 的 120 场内部 validation 没有稳定超过旧 5A，
   这条线就只作为“5A 不足”的补充证据；
2. 如果它能接近或超过参数匹配单 Actor，再考虑是否还需要继续放大或细调；
3. 不读取 sealed test。

## 预期用法

- 先跑短 pilot，确认 local critic 是否真的补上了旧 5A 的结构缺口；
- 之后再决定要不要扩预算；
- 这条线不替代当前 `5A + epoch-16 + Gate` 主方法，只是修正旧普通基线。

## 启动脚本

见 `scripts/start_training_current_generalist_from_5a_local_critic.sh`。
