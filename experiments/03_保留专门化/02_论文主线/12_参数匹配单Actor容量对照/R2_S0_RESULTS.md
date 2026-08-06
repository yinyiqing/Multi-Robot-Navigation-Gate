# G12-R2-S0单车Broad结果

状态：`completed / admission passed / S1 targeted evaluation pending`。日期：`2026-08-06`。

## 1. 运行身份

```text
experiment: G12-R2-S0
model: capacity_wide_r2_s0_broad_n1_seed20260811
Actor: 24 -> 1137 -> 855 -> 2
Actor parameters: 1,003,127
initialization: random Actor + random 24-dimensional Critic
budget: 100,004 agent samples
validation: 120 fixed n1 episodes every 20k samples
```

训练结束时覆盖`2674/3000`个不同train scenario。运行按冻结的5个epoch预算正常结束，
没有提前停止、异常退出、NaN或Critic数值爆炸。

## 2. Validation曲线

| epoch | agent samples | full success | collision | unresolved/timeout | avg steps | avg final distance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `20,007` | `0.983` | `0.000` | `0.017` | `26.9` | `0.272` |
| 2 | `40,017` | `1.000` | `0.000` | `0.000` | `9.9` | `0.226` |
| 3 | `60,012` | `1.000` | `0.000` | `0.000` | `9.9` | `0.235` |
| 4 | `80,006` | `0.992` | `0.008` | `0.000` | `9.0` | `0.232` |
| 5 | `100,004` | `1.000` | `0.000` | `0.000` | `9.8` | `0.232` |

冻结选择规则保留epoch 2为best；epoch 3和5与其full success并列，但没有超过既有best。
Epoch 2 checkpoint是后续S1诊断和S2 warm start的唯一默认候选，latest不得因时间更晚自动
覆盖best。

## 3. 数值诊断

TensorBoard全程记录`6429`个优化点：

- Critic loss从`680.79`下降到`68.09`，观测范围`36.35-694.93`；
- average Q从`-0.31`增长到`96.97`；
- max Q全程上限`151.12`，与当前回报尺度相容；
- 日志未出现NaN、Inf或P1式固定动作坍塌。

以上只说明S0训练稳定，不把单个episode中的边界动作解释为整体动作饱和率。

## 4. Artifact

| artifact | SHA-256 |
| --- | --- |
| `capacity_wide_r2_s0_broad_n1_seed20260811_best_actor.pth` | `7cb61925a4188e638859f88d38288e0431e5f05be489fa6107a77c7efaed3822` |
| `capacity_wide_r2_s0_broad_n1_seed20260811_best_critic.pth` | `1d5e9bbcc7062886548cf2691ce446993bb5a04be11d5a517cbcb9fa610ad752` |
| `capacity_wide_r2_s0_broad_n1_seed20260811_best.pt` | `7870e064bfdf09a762a63b6a2397207d96614463cef295cbd7a69f7cefde7223` |
| `capacity_wide_r2_s0_broad_n1_seed20260811.npy` | `b6e2afacd0f9d66bc3c04653d31c469a8d24e43760a0d76fd53fefb473869cc7` |

## 5. 结论与下一步

S0以明显余量通过`full success >= 0.85`、`collision <= 0.10`和`timeout <= 0.10`准入。
这证明加宽Actor可以从随机初始化形成稳定单车导航底座，也支持P1/R1坍塌来自原协议中的
fresh-Critic无约束更新，而不是“参数量增加必然学不会”。

该结论不涉及两车、五车或机器人冲突。启动任何S1训练前，先冻结并运行stage1/e/f/g固定
case的只评测诊断；只有发现具体能力缺口才在对应case补课。若全部固定case通过，则S1训练
预算记为`0`并直接进入S2，避免对已经稳定的broad能力做无目标更新。
