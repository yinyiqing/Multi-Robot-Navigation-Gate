# G12-R2-S2两车Broad首段结果

状态：`completed / passed / 10k best frozen`。日期：`2026-08-07`。

## 运行

本次从R2-S0 epoch 2 best完整加载Actor与Critic，在冻结的两车完整broad train上训练
`20,000 agent samples`，每10k在固定120场n2 validation上评测。训练共完成
`13,630 env steps / 839 episodes`；无NaN、Q爆炸、固定动作坍塌或模型加载降级。

## 结果

| checkpoint | agent success | full success | collision | timeout | avg steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10k | `0.9417` | `0.9333` | `0.0583` | `0` | `11.02` |
| 20k | `0.9333` | `0.9250` | `0.0667` | `0` | `10.71` |

两次评测都通过首段准入。20k相对10k的full success只下降`0.0083`，未触发`0.10`
回滚线；但10k仍是运行内最优，因此冻结10k：

```text
model: capacity_wide_r2_s2_broad_n2_seed20260814_best
timestep: 10004
env steps: 6796
episode: 436
Actor SHA-256: 220698f1e4a918deb88d0b47f8c4f28b2330194401b4b82c80afe92d8f63f465
Critic SHA-256: acbecd846cbae2456e2a493ec545eeaf2718a11aa2cc6fe58c2a9d3af1fbe7ca
checkpoint SHA-256: 8c3260bd97cf9d38c0d56699817bb89e7dddd76bc67cb8a4412ee3e97199f895
```

20k latest只保留作诊断，不覆盖10k best。

## 结论边界

- 加宽Actor已稳定从单车进入两车完整导航，可以继续三车课程；
- n2 train中派生冲突场景仅`232/3000=7.7%`，validation中仅`17/120=14.2%`，因此
  本结果主要证明两车broad导航稳定性，不证明双车冲突能力已经提升；
- 当前评测汇总没有逐scenario记录，不能可靠拆分17个冲突case的成绩；
- 没有S0在同一n2 validation上的零更新基线，不能把10k成绩解释成相对S0的训练增益；
- 全部失败均为碰撞，无timeout或unresolved。

训练完成后的XMLRPC connection refused发生在ROS关闭阶段，不影响结果有效性。

