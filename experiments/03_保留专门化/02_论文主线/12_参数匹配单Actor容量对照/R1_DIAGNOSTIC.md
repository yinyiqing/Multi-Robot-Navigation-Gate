# G12-R1：原宽度训练稳定性诊断结果

状态：`completed / archived diagnostic / rejected as model`。日期：`2026-08-06`。

## 1. 运行有效性

- 模型：`capacity_original_width_r1_n5_seed20260810`；
- Actor：原宽度`24 -> 800 -> 600 -> 2`，`501,802`个可训练参数；
- 初始化：冻结5A Actor直接加载，fresh原始24维Critic；
- 数据、seed、reward、学习率、20k Critic warm-up和validation均与P1相同；
- 设备：CUDA，RTX 4090；
- Epoch 1：`20,058` agent samples；Epoch 2：`40,050` agent samples；
- 运行按预注册退化保护正常结束，不是程序错误；
- 原始日志：`logs/archive/diagnostic/g12_r1/`。

## 2. 同协议闭环结果

| 运行 | Actor | checkpoint | agent success | collision | unresolved | full success | timeout | 平均步数 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | `1137x855` | Actor解冻边界 | `0.917` | `0.083` | `0.000` | `0.717` | `0.000` | `29.8` |
| P1 | `1137x855` | Actor更新20k后 | `0.537` | `0.457` | `0.007` | `0.050` | `0.033` | `53.0` |
| R1 | `800x600` | Actor解冻边界 | `0.915` | `0.085` | `0.000` | `0.717` | `0.000` | `28.0` |
| R1 | `800x600` | Actor更新20k后 | `0.735` | `0.248` | `0.017` | `0.283` | `0.083` | `58.7` |

R1的full success下降`0.433`、agent success下降`0.180`，同时触发两条预注册停止条件。
分层full success从interaction/weak的`0.517/0.917`降到`0.100/0.467`，退化不是只发生
在某一个场景层。

## 3. 动作漂移诊断

在R1 latest checkpoint保存的全部`40,050`个replay状态上，用同一批状态离线运行5A、
R1 Epoch 2和P1 Epoch 2 Actor。第一维按环境执行方式从`[-1,1]`映射为线速度`[0,1]`，
第二维保留角速度`[-1,1]`。

| Actor | 平均执行动作 `[linear, angular]` | 相对5A平均绝对漂移 | 任一维漂移`>0.1` |
| --- | --- | --- | ---: |
| 5A | `[0.641, -0.304]` | `[0, 0]` | `0` |
| R1 Epoch 2 | `[0.956, -0.334]` | `[0.314, 0.111]` | `46.1%` |
| P1 Epoch 2 | `[0.989, -0.924]` | `[0.347, 0.620]` | `71.8%` |

这些replay状态来自训练策略本身，不代表独立状态分布，因此只用于机制诊断，不能替代
闭环validation。它们显示两种宽度都明显偏离5A并趋向更激进的线速度；P1还出现更严重的
单侧角速度漂移。

## 4. 结论

R1回答了预注册问题：**P1坍塌不能归因于Actor参数量翻倍。** 原宽度在同一fresh-Critic、
同一数据和同一无行为约束TD3协议下也发生严重退化，因此共同根因是Actor解冻后的训练
稳定性和能力覆盖问题。

P1比R1退化更严重，动作漂移也更大。这只支持“函数扩宽可能放大不稳定性”的单seed机制
证据，不足以独立证明扩宽必然更难训练，更不能证明大Actor容量不足。

R2仍按修订路线从随机初始化复现完整课程。R3/R4必须保留行为锚定、Q尺度归一化、梯度
裁剪和完整场景内部validation，不能复用P1/R1的无约束更新协议。

## 5. Artifact哈希

| artifact | SHA-256 |
| --- | --- |
| `capacity_original_width_r1_n5_seed20260810_best.pt` | `2ae91fed13f57340551322399fcaa309966b555c123d80cc8ae82118bcb35d4b` |
| `capacity_original_width_r1_n5_seed20260810_latest.pt` | `072443424fdaef08204bd0877542dec589faf7d1afefa7614efb9d17d436fff4` |
| `capacity_original_width_r1_n5_seed20260810_epoch_001_actor.pth` | `40d05a3250c8e4e35753b5624e92a9dd95150d860537581d684115cdaf12a23b` |
| `capacity_original_width_r1_n5_seed20260810_epoch_002_actor.pth` | `cd2824495bef32efb255e81100e2f3b7aca943f8b8a71f6f0f22f8a6cb5b155c` |
