# log_archive 说明

`logs/` 目录现在只保留当前直接要看的有效日志。

这里放已经归档的日志，包括：

- 无效续跑
- Gazebo 中途崩溃
- 被后续正式重跑覆盖的旧日志

## 当前该看什么

1. 当前最重要的参考基线
   - `test_std5_PAIR_5D_20260713_084935.log`
   - `test_stage3_asym_three_5_PAIR_5D_20260713_104759.log`

2. 当前密集训练参考
   - `train_multi_curriculum_stage3_asym_three_5_detached_20260713_003642.log`

## 已归档

- `archive/20260713_invalid_resumed_runs/`
  - 这里放的是这次误续跑的无效日志
  - 原因：测试从 `episode 120` 继续跑，不是从 1 开始
  - 这些日志不再作为正式结果引用

- `archive/20260713_gazebo_exit255_incomplete/`
  - 这里放的是重新从 1 开始后，中途遇到 `gazebo exit code 255` 的失败日志
  - 这些日志只保留排错信息，不作为正式结果引用

## 简单约定

- `logs/`：只放当前主线和关键基线
- `log_archive/`：放无效、被重跑覆盖、或阶段性淘汰的日志
