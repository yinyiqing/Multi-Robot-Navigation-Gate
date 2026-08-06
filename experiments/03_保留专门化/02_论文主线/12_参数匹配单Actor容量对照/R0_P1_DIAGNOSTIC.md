# G12-R0：P1训练坍塌诊断

状态：`completed / archived as diagnostic`。日期：`2026-08-06`。

## 运行身份

- 模型：`capacity_matched_actor_wide_n5_seed20260810_pilot`；
- Actor：`24 -> 1137 -> 855 -> 2`，共`1,003,127`个可训练参数；
- 初始化：从冻结5A做函数保持扩宽，最大输出误差`4.62e-7`；
- seed：`20260810`；设备：`cuda`；
- 数据：`g11_a1_gate_v1`的640场train与互斥120场validation；
- 更新：fresh 24维Critic，前20k samples冻结Actor，之后无约束TD3更新Actor；
- 原始日志：`logs/archive/diagnostic/g12_p1/`。

## 结果

| checkpoint | agent samples | agent success | collision | unresolved | full success | timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Epoch 1，Actor解冻边界 | `20,008` | `0.917` | `0.083` | `0.000` | `0.717` | `0.000` |
| Epoch 2，Actor更新20k后 | `40,007` | `0.537` | `0.457` | `0.007` | `0.050` | `0.033` |

同一批replay状态上的平均动作从`[0.546, -0.040]`漂移到`[0.992, -0.913]`。Critic
预测Q上升，但真实闭环成功率和回报下降。Epoch 2同时触发full success下降`0.667`和
agent success下降`0.380`两条预注册停止条件，运行正常早停，不是程序错误。

## 结论边界

P1只证明当前“5A函数保持扩宽 + fresh Critic + 无行为保持项”的TD3协议会发生Actor
动作饱和和闭环训练坍塌。它不能证明大Actor容量不足，也不能作为公平单Actor性能对照。

下一步R1用原宽度5A复现前40k协议。若R1同样坍塌，主要问题在fresh-Critic无约束更新；
若只有P1坍塌，才把排查重点放在扩宽初始化和新增参数的优化尺度。

## Artifact哈希

| artifact | SHA-256 |
| --- | --- |
| `capacity_matched_actor_wide_n5_seed20260810_pilot_best.pt` | `f4711d0baec4cc0b7e852086c78fab277359401f8447f5ab1c3ea533d115e9e1` |
| `capacity_matched_actor_wide_n5_seed20260810_pilot_latest.pt` | `7df54151655f5c8a1d6183935fcfd1aec76680c3859ba7ee7fe0d1fb64aac6a4` |
| `capacity_matched_actor_wide_n5_seed20260810_pilot_epoch_001_actor.pth` | `42a020e40f89a36b91ff309df8172e3deb739105b060caf35f79184067308f36` |
| `capacity_matched_actor_wide_n5_seed20260810_pilot_epoch_002_actor.pth` | `b4090e015e45326465dd15f16f6c6914dc02ea596bc5251d323ba066cf3b143d` |
