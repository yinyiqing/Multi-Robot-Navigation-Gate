# 保留专门化

本仓库现在只服务一条主线：

**先训练一个真正适合 dense 场景的第二个 actor；等第二个专家成立后，再考虑 gate。**

## 已确认结论

1. `5D` 是当前最稳的基础 actor。
   - `stage3_asym_three_5`：`success=0.902 / collision=0.097 / full=0.650`
   - `stage4_asym_dense_5_bridge`：`success=0.540 / collision=0.475 / full=0.250`
2. 当前 dense 训练入口是 `stage4_asym_dense_5_bridge`。
   - 它比 `stage3_asym_three_5` 更密集。
   - 它避免贴脸起点和目标重叠，固定 `5D` 能过一半左右，属于可学习的 dense 区间。
3. 过硬的 hard stage4 已清理。
   - 那组 case 目标点过近，固定 `5D` 只有约 `0.355 / 0.680 / 0.025`。
   - 它更像病态压力测试，不再保留为当前证据或入口。
4. 直接 full actor fine-tune 已经失败。
   - `5D -> bridge`：第 1 轮略升，之后掉到约 `0.215 / 0.790 / 0.000`。
   - `5A -> bridge`：前几轮贴近基线，actor 解冻后掉到约 `0.490 / 0.520 / 0.225`。
5. `5D` 退化、`5A` 维持原状，说明问题不只是 warmstart 选错。
   - `5D` 退化说明较强的已有策略会被新 dense critic / reward 梯度拉坏。
   - `5A` 不退化但也不进步，说明换一个更早期 actor 并不能自动学出 dense 专家。
   - 当前首要问题是 full actor 解冻不稳定，其次才是“5D 是否过拟合”。
6. 现在的问题不是 gate，也不是 attention。
   - `5A + 5D` 的 hard switch 没超过单独 `5D`。
   - oracle 多数仍选 `5D`，说明两个 actor 互补不够。
   - 没有可靠 dense 专家时，gate 只会学成“多数时候保护 5D”。

## 当前主线

```text
5D actor
  -> 保留已有导航主干
  -> 只开放少量可训练参数
  -> 在 stage4_asym_dense_5_bridge 上训练 dense 专家
```

下一步实验：

```bash
scripts/start_training_detached_dense5_bridge_head_from_5d.sh
```

这个脚本做的事情：

- 从 `5D` actor warmstart。
- 只加载 actor，新建 critic。
- 冻结 actor 前两层，只训练最后 action head。
- 延迟 actor 更新，先让 critic 在 bridge 分布上稳定。
- 日志先写到仓库根目录 `logs/`，跑完后再归档。

## 判断标准

必须同时满足：

- bridge 超过固定 `5D` baseline：`0.540 / 0.475 / 0.250`
- standard 不明显崩。
- 如果出现“第 1 轮还行，后面持续掉”，立即停。

## 暂停内容

这些先不做：

- full actor fine-tune
- `5A + 5D` learned gate
- attention / residual 修正器
- 已清理的 hard stage4 路线

## 保留证据

只保留能支撑当前路线的日志：

- `01_冲突验证/logs/test_std5_5D_20260707_205700.log`
- `01_冲突验证/logs/test_stage3_asym_three_5_5D_20260707_213454.log`
- `03_dense专家训练/logs/test/test_stage4_asym_dense_5_bridge_5D_BASELINE_STAGE4_BRIDGE_20260715_150100.log`
- `03_dense专家训练/logs/test/test_stage4_asym_dense_5_bridge_5A_BASELINE_STAGE4_BRIDGE_20260715_175834.log`

失败 run 的大日志和模型产物不保留，结论写在本文档里。
