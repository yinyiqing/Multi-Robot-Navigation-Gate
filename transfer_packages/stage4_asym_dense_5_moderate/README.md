# stage4_asym_dense_5_moderate 迁移包

这个文件夹只放复现 moderate dense 环境需要的文件，不包含训练产物。

不要迁移：

- `logs/`
- `checkpoints/`
- `results/`
- `.pid`
- TensorBoard `runs/`

## dense 环境定义

`stage4_asym_dense_5_moderate` 是五车中等密度交互环境：

- 5 个机器人在同一个原始有墙地图中运行；
- 每个 case 有 2-3 个主交互机器人；
- 另外 2-3 个机器人提供空间压力或路径占用；
- 起点不贴脸，目标不扎堆；
- 每个 case 约 1-3 组直线路径交叉；
- 目的不是 hard stress test，而是训练 dense 专家 actor。

## 文件清单

必须迁移：

- `cases/stage4_asym_dense_5_moderate_cases.json`
- `scripts/start_training_detached_multi_curriculum.sh`
- `scripts/start_test_detached_multi_curriculum.sh`

训练 dense 专家时需要：

- `scripts/start_training_detached_dense5_moderate_geo_critic_from_5d.sh`
- `scripts/stop_training_detached_dense5_moderate_geo_critic_from_5d.sh`

注意：如果目标分支的 `start_training_detached_multi_curriculum.sh` 或
`start_test_detached_multi_curriculum.sh` 已经有自己的改动，不建议整文件覆盖。
只合并其中 `stage4_asym_dense_5_moderate` 相关分支和 stage 列表即可。

## 依赖 checkpoint

默认 warmstart 模型：

```text
TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best
```

目标线必须已有这个 checkpoint，否则需要一起迁移对应模型文件。

## 复现命令

先测固定 `5D` baseline：

```bash
scripts/start_test_detached_multi_curriculum.sh stage4_asym_dense_5_moderate
```

再训练 dense 专家：

```bash
scripts/start_training_detached_dense5_moderate_geo_critic_from_5d.sh
```

停止训练：

```bash
scripts/stop_training_detached_dense5_moderate_geo_critic_from_5d.sh
```

## 参考 baseline

固定 `5D` 在该环境上约为：

```text
success=0.513
collision=0.502
full_success=0.058
```

这个结果只作为 sanity check，不需要迁移日志文件。
