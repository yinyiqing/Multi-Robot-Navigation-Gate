# R2C 修正版：R2-10k 完整 checkpoint 延续

状态：`registered / not started`。日期：`2026-08-15`。

## 目的

验证已经从单车到五车训练完成的加宽单 Actor，在完整五车混合训练分布上继续训练时，
是否能够保持 R2-10k 的能力。该实验不是重新训练大 Actor，也不是重新设计 Critic。

## 固定起点

- Actor：`capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001_actor.pth`
- Critic：`capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001_critic.pth`
- Actor 结构：`24 -> 1137 -> 855 -> 2`
- Actor 与 Critic 必须同时加载；禁止 actor-only warm start
- 起点为 R2-S4 10k fallback，已通过五车准入

## 唯一变化

相对 R2-S4，只把训练数据扩展为冻结的完整五车混合清单：

```text
g12_r3_mixed_v1/train.json.gz
```

该清单包含 standard、dense、strong 和 multi-edge 场景。validation 固定为：

```text
g12_full_scene_selection_v1/validation.json.gz
```

不得读取 Gate 数据、G11-D2、G11-E 或 sealed test。

## 明确禁止的改动

- 不加载 5A、epoch-16、Gate 或 R3 checkpoint；
- 不使用邻域 Critic；Critic 保持 R2 的 24 维输入；
- 不使用动态 reward、distance-weighted reward 或 cooperative reward；
- 不冻结 Actor后再解冻；Actor 从 continuation 的第一个 sample 起正常更新；
- 不使用新的 Q normalization、anchor、Actor gradient gate 或 privileged target；
- 不按 validation 峰值扫描 checkpoint。

## 预算和停止条件

- pilot：`20,000 agent samples`，每 `10,000` samples 评估一次固定 120 场；
- seed：`20260827`；
- Actor/Critic 学习率：`8e-5 / 8e-5`，其余 R2-S4 超参保持不变；
- 若相对 R2-10k 的 full success 下降超过 `0.05`，或 collision 增加超过 `0.03`，
  或 timeout 增加超过 `0.02`，停止并冻结 R2-10k；
- pilot 通过后，另行登记是否追加预算；本脚本不会自动启动第二阶段。

该实验只回答“R2 大 Actor在完整分布上继续训练是否保持能力”，不能把结果解释为
R3 的邻域 Critic 或动态 reward 结果。
