# G12-R3完整五车混合分布40k pilot协议

状态：`frozen / first 40k pilot authorized`。日期：`2026-08-08`。

## 目的

在不读取G11-D2、G11-E或sealed test的前提下，检查R2加宽Actor在完整standard/dense五车
分布和强交互重采样下能否稳定更新，并保持普通导航能力。R3是参数匹配单Actor对照的
pilot，不进入当前双Actor+Gate方法。

## 冻结输入

```text
R2 Actor:
  capacity_wide_r2_s4_broad_n5_seed20260816_epoch_001_actor.pth
  ace910553931873a275d66e3a964fd2b4716d30b6c68c8dcb3e7af96e56783ee

train:
  fixed_v1/views/g12_r3_mixed_v1/train.json.gz
  c2ce37e51e8e98423d6ed6d295a7f5cf54d02e76c42f6459ce35003c899e0841

validation:
  fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz
  52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635
```

训练manifest按`standard/strong/dense/strong`确定性cycle，strong内部按
`deep/deep/close/close/margin`平衡。别名ID只表示重采样，原ID保存在view metadata。

## 冻结训练配置

| 配置 | 值 |
| --- | --- |
| experiment | `G12-R3-40k` |
| model | `capacity_wide_r3_mixed_n5_seed20260818` |
| seed | `20260818` |
| Actor | `24 -> 1137 -> 855 -> 2`，全程控制五车 |
| Critic | fresh geometry-only local Critic，`legacy` 5维邻车槽位 |
| reward | `0.8 self + 0.2 visible-neighbor mean` |
| budget | `40k agent samples` |
| Actor warm-up | `21k`解冻阈值；保护20k评测边界并留1k episode越界guard |
| evaluation | 固定120场，在`20k/40k`评测 |
| batch / gamma / tau | `256 / 0.999 / 0.005` |
| Actor/Critic LR | `1e-5 / 8e-5` |
| policy frequency | `2` |
| exploration | `0.08 -> 0.03` over 40k |
| Q normalization alpha | `1.0` |
| safe-state anchor | `lambda_keep=1.0`，最近活动机器人距离`>2.0m` |
| Actor gradient norm clip | `1.0` |
| fixed physics step | `0.001` |

最近机器人距离、邻车几何和邻车reward只用于训练；Actor输入仍是本车24维观测。禁止oracle
rollout、oracle target policy和interaction-only Actor更新。R2 Critic输入维度与R3不同，
因此只加载R2 Actor，fresh Critic先完成20k warm-up。训练在episode结束后批量更新，解冻
阈值固定为21k，避免跨过20k的episode在第一次评测前提前更新Actor。

## 判断与停止

20k是Actor尚未解冻的边界诊断，不作模型选择。40k相对冻结R2参考同时满足：

1. 0-edge full success下降不超过`0.03`；
2. 总体full success不低于R2，且edge-1或multi-edge至少一个改善；
3. standard与dense均不下降超过`0.05`；
4. timeout增加不超过`0.03`；
5. 无NaN、持续动作饱和、Critic爆炸或异常梯度。

只运行本次固定配置。40k失败则停止并分析，不根据validation修改学习率、anchor或采样比例
重跑；通过才允许保持完全相同协议扩展R4。

## 命令与日志

```bash
bash scripts/experiment.sh start actor-g12-r3-40k
bash scripts/experiment.sh status
bash scripts/experiment.sh stop actor-g12-r3-40k
```

实时日志统一位于`logs/active/capacity-wide-g12-r3/`，完成后归档。
