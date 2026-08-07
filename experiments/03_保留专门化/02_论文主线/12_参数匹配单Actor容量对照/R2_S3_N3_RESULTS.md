# G12-R2-S3三车Broad首段结果

状态：`completed / passed / 20k best frozen`。日期：`2026-08-07`。

## 运行

本次从R2-S2两车10k best完整加载Actor与Critic，在冻结的三车完整broad train上训练
`20,083 agent samples`，共`11,733 env steps / 505 episodes`。每10k在固定120场n3
validation上评测；无NaN、Q爆炸、固定动作坍塌或模型加载降级。

## 结果

| checkpoint | agent success | full success | collision | unresolved | timeout | avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | `0.9417` | `0.8750` | `0.0528` | `0.0056` | `0.0167` | `22.05` |
| 20k | `0.9472` | `0.8833` | `0.0528` | `0` | `0` | `14.00` |

20k在full success、agent success、unresolved、timeout和平均步数上均不差于10k，因此
冻结20k best：

```text
model: capacity_wide_r2_s3_broad_n3_seed20260815_best
timestep: 20083
env steps: 11733
episode: 505
Actor SHA-256: 0ad69f89378b88812c1ce2306a07c75fbd4d80a9616b1db3a18e6d36c9037f04
Critic SHA-256: 55a20491f6f498960d77284e44409c99d7d710bb5a39fb18a212a3d047650d67
checkpoint SHA-256: 9e2ba47d7a13aa7076ee944e6a3aa27128d7de1ca18bb00d8315c6629e6d6c56
```

## 结论边界

- 加宽Actor已稳定完成三车完整导航首段，可以进入五车课程；
- n3 validation包含`43/120`个冲突场景，其中7个为多冲突，但当前评测只保存汇总，
  不能可靠报告各冲突层成绩；
- full success从n2的`0.9333`到n3的`0.8833`不可直接解释为退化，因为agent数量、manifest
  和任务难度均已变化；n3内部10k到20k的结果显示训练稳定且效率改善；
- 该阶段仍不是参数匹配单Actor的最终五车论文成绩。

训练结束后的ROS输出为正常清理，不影响运行有效性。

