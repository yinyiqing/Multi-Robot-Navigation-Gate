# 普通 Actor 失败对照

这里集中保存“在 fixed standard 上继续训练一个 standard expert”的完整失败链。结论已经冻结：不再训练第三个普通导航 Actor，直接使用 5A。

| 实验 | 作用 | 结论 |
| --- | --- | --- |
| `D4_standard_expert_actor_only_v1` | Actor-only warm-start | 短期提高后退化 |
| `D4_standard_expert_fullwarm_anchor_v2` | 完整 warm-start + anchor | 未形成稳定提升 |
| `D4_standard_expert_timeoutfix_v3` | 修复 timeout/更新口径 | 小规模正向信号未稳定复现 |
| `D4_standard_expert_timeoutfix_v3_validation500` | 500 场正式复核 | 未超过冻结 5D，拒绝候选 |

本组只作为“直接覆盖式微调会退化”的证据，不再续训或从中选择 checkpoint。
