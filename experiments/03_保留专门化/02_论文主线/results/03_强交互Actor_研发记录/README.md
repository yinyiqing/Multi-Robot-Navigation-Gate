# 强交互 Actor 研发记录

这里保存正式强交互 Actor 形成前的机制排查和受控试验。它们解释最终配置为什么采用 5A warm-start、新 Critic、安全聚焦更新、训练期 oracle 分工和均衡全池采样。

| 实验 | 状态 | 关键结论 |
| --- | --- | --- |
| `D4_interaction_edge1_residual_pilot_s20260720` | rejected | residual 饱和，Q 上升但真实性能下降 |
| `D4_interaction_edge1_conservative_residual_v2_s20260720` | rejected | 抑制了饱和，但没有可靠成功率增益 |
| `D4_strong_interaction_curriculum_stage1_s20260723` | failed | Actor 在危险状态加速，Critic 动作偏好错误 |
| `D4_pair_interaction_pilot_s20260724` | failed diagnostic | 双车收益不能稳定迁移到五车强交互 |
| `D4_interaction_oracle_specialist_pilot_s20260724` | rejected | 修复状态分流后仍出现统一转向和危险加速 |
| `D4_safe_distance_warmstart_pair_legacy_context_s20260725` | stopped | 发现 Critic 使用世界坐标捷径，协议失效 |
| `D4_interaction_ego_motion_from_5a_s20260725` | rejected | ego-frame context 修复正确，但单独不足以提升 |
| `D4_interaction_dense_safety_reward_from_5a_s20260725` | rejected | 加强安全 reward 仍被危险样本稀少和 Critic 偏差限制 |
| `D4_interaction_critic_gradient_guard_from_5a_s20260725` | rejected | guard 阻止了错误解冻，但没有产生可训练 Actor |
| `D4_interaction_safety_ranking_from_5a_s20260725` | rejected | 修正线速度偏好后仍存在全局角速度漂移 |
| `D4_interaction_focused_actor_from_5a_s20260725` | superseded | 首次得到全面正向信号，成为正式配置来源 |
| `D4_interaction_focused_actor_from_5d_s20260725` | rejected comparison | 5D 全套配置的收益明显弱于 5A |
| `D4_interaction_focused_actor_5a_init_5d_weak_s20260726` | rejected | 更换外围弱 Actor 改变轨迹分布后退化 |
| `D4_interaction_full_random_sampling_pilot_s20260726` | rejected protocol | 随机有放回采样无法保证分层覆盖，改用 `balanced_cycle` |

正式训练产物不在本组，见 [`../05_当前冻结方案/`](../05_当前冻结方案/README.md)。
