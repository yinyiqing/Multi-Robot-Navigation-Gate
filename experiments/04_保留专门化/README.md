# 保留专门化

本目录只服务一条主线：

**训练一个真正适合 dense 场景的第二个 actor，然后再考虑 gate。**

## 当前结论

已经确认的事情：

1. `5D` 是目前最稳的普通到 dense bridge baseline。
   - `stage3_asym_three_5`：`0.902 / 0.097 / 0.650`
   - `stage4_asym_dense_5_bridge`：`0.540 / 0.475 / 0.250`
2. hard `stage4_asym_dense_5` 太难。
   - `5D` 固定测试：`0.355 / 0.680 / 0.025`
   - 只作为压力测试，不作为训练入口。
3. `stage4_asym_dense_5_bridge` 是当前 dense 专家训练入口。
   - 起点间距不过近，目标也不过度重叠。
   - 但仍能暴露真实密集冲突。
4. 直接 full actor fine-tune 会退化。
   - `5D -> bridge`：第 1 轮略升，随后掉到 `0.215 / 0.790 / 0.000`。
   - `5A -> bridge`：前 3 轮贴近基线，actor 解冻后掉到 `0.490 / 0.520 / 0.225`。
5. `5A + 5D` 互补不足。
   - hard switch 没超过单独 `5D`。
   - oracle 多数仍选 `5D`。

## 暂停的路

这些不是当前主线：

- 直接继续训练完整 actor。
- `5A + 5D` learned gate。
- attention residual / frozen residual 修正器。

attention residual 这轮已经证明“冻结基础 actor 不会崩”，但它不是新的 dense actor，且 residual 基本没有真正打开。因此它只作为失败旁证，不再继续。

## 当前主线

下一步不是 attention，也不是 gate，而是：

```text
5D actor
  -> 保留大部分已有导航能力
  -> 只开放小部分可训练参数
  -> 在 stage4_asym_dense_5_bridge 上训练 dense 专家 actor
```

优先尝试顺序：

1. 冻结前两层，只训练最后 action head。
2. 如果 head-only 不够，再加小 adapter。
3. 如果仍不够，再考虑逐层解冻，而不是直接 full fine-tune。

判断标准：

- bridge 必须超过 `5D` baseline：`0.540 / 0.475 / 0.250`
- standard 不能明显崩。
- 如果训练内一开始好、随后持续掉，立即停，不继续烧时间。

## 保留的基线日志

保留这些日志是为了复查结论：

- `01_冲突验证/logs/test_std5_5D_20260707_205700.log`
- `01_冲突验证/logs/test_stage3_asym_three_5_5D_20260707_213454.log`
- `03_门控注意力增强/logs/test/test_stage4_asym_dense_5_5D_BASELINE_STAGE4_20260715_140502.log`
- `03_门控注意力增强/logs/test/test_stage4_asym_dense_5_bridge_5D_BASELINE_STAGE4_BRIDGE_20260715_150100.log`
- `03_门控注意力增强/logs/test/test_stage4_asym_dense_5_bridge_5A_BASELINE_STAGE4_BRIDGE_20260715_175834.log`

失败 run 的大日志和无效模型产物不再保留，结论写在本文档里。
