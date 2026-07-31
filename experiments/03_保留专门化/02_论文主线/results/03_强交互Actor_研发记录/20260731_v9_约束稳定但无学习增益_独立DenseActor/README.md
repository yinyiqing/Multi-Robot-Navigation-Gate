# 20260731 v9：约束稳定但无学习增益（独立 Dense Actor）

## 目的

针对v8的Critic危险加速，恢复Q尺度归一化和全状态5A anchor，并在危险接近时只禁止新Actor比5A更快，不再强制统一低速。

## 协议

- 5A Actor warm-start，新Actor独立控制完整episode；
- 24维Actor输入，ego-motion local Critic；
- 无oracle、无Actor切换；
- fixed-v1 dense train按cycle采样；
- 固定50场dense monitor；
- epoch 1-2冻结Actor，epoch 3开始更新；
- epoch 7结束后人工停止。

## 结果

| epoch | 状态 | success | full success | collision | timeout |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 冻结5A | 0.740 | 0.380 | 0.260 | 0.000 |
| 2 | 冻结5A | 0.756 | 0.380 | 0.244 | 0.000 |
| 3 | Actor解冻 | 0.748 | 0.380 | 0.252 | 0.000 |
| 4 | Actor解冻 | 0.744 | 0.300 | 0.256 | 0.000 |
| 5 | Actor解冻 | 0.748 | 0.340 | 0.252 | 0.000 |
| 6 | Actor解冻 | 0.752 | 0.400 | 0.248 | 0.000 |
| 7 | Actor解冻 | 0.736 | 0.360 | 0.264 | 0.000 |

epoch 6相对冻结5A只多完成`1/50`场，agent success和collision没有改善；epoch 7再次下降。Actor相对5A的线速度变化通常只有约`0.001-0.004`，没有形成有效新策略。

## 结论

v9成功避免了两种已知退化：

- 没有重现v8的统一危险加速；
- 没有重现v5/v6的持续等待和timeout。

但它只是把Actor限制在5A附近，没有解决“谁让谁、如何绕行、何时恢复”，因此拒绝作为独立Dense Actor候选。

## 评测协议新发现

monitor每轮确实使用同一50场、同一顺序，但旧训练没有启用Gazebo fixed physics stepping。同样冻结5A时，epoch 1和2前10场full success分别为`0.400/0.200`，说明固定场景不等于确定性rollout。

后续先启用固定物理步进并做重复性验证，再评估“迎面冲突统一靠右”的诊断规则。规则有效后才考虑生成示范数据，不直接继续堆reward或启动长训练。

## 文件

- `logs/`：完整训练日志；
- `checkpoints/`：best/latest完整训练状态，仅本机保留；
- `pytorch_models/`：epoch 1-7及best/latest权重，仅本机保留；
- `results/`：逐epoch评估数组；
- `tensorboard/`：完整训练事件。
