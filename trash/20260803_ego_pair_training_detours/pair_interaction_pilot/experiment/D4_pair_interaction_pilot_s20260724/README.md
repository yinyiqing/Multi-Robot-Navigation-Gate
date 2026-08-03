# D4 Pair Interaction Pilot

状态：`rejected diagnostic`。该实验只用于检查原 TD3 在结构化双车冲突中的学习方向，不作为五车强交互 Actor 候选。

## 协议

- 双车固定 train/validation：`90/30` 场，均衡覆盖 head-on、crossing、lane-swap。
- 完整 5D Actor/Critic warm-start，原 `24 -> 800 -> 600 -> 2` Actor。
- reward 保留距离加权 `0.8/0.2`，关闭基础前进奖励和低速停滞惩罚。
- epoch 1 Actor冻结；epoch 2完整Actor更新。

## 结果

| Metric | Epoch 1: frozen 5D | Epoch 2 | Delta |
| --- | ---: | ---: | ---: |
| Agent success | 0.467 | 0.400 | -0.067 |
| Collision | 0.450 | 0.600 | +0.150 |
| Full success | 0.300 | 0.367 | +0.067 |
| Timeout | 0.167 | 0.000 | -0.167 |
| Head-on full | 0.600 | 1.000 | +0.400 |
| Crossing full | 0.300 | 0.100 | -0.200 |
| Lane-swap full | 0.000 | 0.000 | 0.000 |

训练后只改善 head-on，crossing 明显退化，lane-swap 未学会。overall 多成功2个episode，但多出9个agent collision，因此拒绝该模型。该结果还暴露了按episode均衡不等于按transition均衡：冻结基线三类平均步数为`171.6/100.6/11.4`，lane-swap在replay中的相对权重极低。

后续不继续双车训练，也不回到 PAIR→THREE；双车结果只保留为 replay 分布诊断。
