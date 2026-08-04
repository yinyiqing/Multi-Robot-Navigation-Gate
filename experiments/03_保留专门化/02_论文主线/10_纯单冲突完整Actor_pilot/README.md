# corrected edge-1完整Actor pilot

状态：`completed / rejected`。本实验只作为“降低冲突复杂度后，是否可以直接训练一个
完整Actor替代双Actor和Gate”的对照，不再续训。

## 问题

从5A warm start，在完整路径只有一条冲突边的五车场景中继续训练一个共享完整Actor，
验证单冲突是否足以让完整策略同时学会普通导航和避让。

## 协议

- Actor：加载5A best，五台车共享同一个完整Actor；
- Critic：fresh原始24维Critic；
- 训练：`edge1_full_horizon_v1/train.json.gz`，511个corrected edge-1场景；
- monitor：固定50个edge-1 validation，25 standard + 25 dense；
- replay：全部active agent transition；
- reward：与5A一致的individual navigation reward；
- Actor：前`20,000` agent samples冻结，之后更新；
- budget：`6 x 5,000` agent samples；
- seed：`20260803`；
- 模型前缀：`full_actor_edge1_from_5a_s20260803`。

训练不读取冲突pair身份，不选择受控ego，不使用Gate或local Critic。Pair元数据只在训练前
用于离线筛选edge-1场景。

## 结果

| epoch | Actor状态 | agent success | collision | unresolved | full success | timeout | 平均步数 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | frozen 5A / Critic warm-up | `0.832` | `0.164` | `0.004` | `0.440` | `0.020` | `31.5` |
| 2 | frozen 5A / Critic warm-up | `0.804` | `0.196` | `0.000` | `0.420` | `0.000` | `26.0` |
| 3 | frozen 5A / Critic warm-up | `0.820` | `0.180` | `0.000` | `0.440` | `0.000` | `30.2` |
| 4 | 解冻边界，更新量极少 | `0.804` | `0.196` | `0.000` | `0.420` | `0.000` | `25.4` |
| 5 | Actor已更新 | `0.672` | `0.264` | `0.064` | `0.160` | `0.300` | `132.1` |
| 6 | Actor已更新 | `0.552` | `0.396` | `0.052` | `0.100` | `0.260` | `128.0` |

Actor解冻后同时出现full success下降、碰撞上升和timeout。按预先规定的停止标准，本路线
拒绝，不进入421场完整validation，也不增加seed或训练预算。

`full_actor_edge1_from_5a_s20260803_best.pt`保存时间位于Actor解冻前，主要对应冻结5A
基线，不得把文件名中的`best`误认为已经学会单冲突的完整Actor。

## 结论边界

这50场pilot不能证明所有完整Actor方法都不可能成功；它证明的是当前最基础的
`5A warm start + corrected edge-1 + 原始TD3`没有产生学习增益，继续增加同协议训练预算
不合理。当前论文路线仍采用5A普通导航和epoch-16条件避障的在线分工。

运行日志保存在本机：

```text
logs/archive/rejected/full_actor_edge1/train_full_actor_edge1_from_5a_s20260803_20260803_235338.log
```

历史支线已隔离，默认不读取`trash/`。
