# 强交互 Actor 研发记录

这里保存独立强交互Actor形成前的机制排查和受控试验。当前只保留5A warm-start、新ego-motion Critic、完整Dense独立rollout和固定数据协议；安全聚焦更新与训练期oracle分工属于已否定的条件Actor历史，不再用于独立Actor。

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
| `20260729_v3_减速约束_独立DenseActor` | rejected | 统一减速抑制危险加速，但full success没有提升 |
| `20260729_v4_让行停车_独立DenseActor` | rejected | 单轮峰值略升，未解决多车通行顺序 |
| `20260730_v5_协调重启_独立DenseActor` | rejected | 碰撞下降后转化为等待，最终timeout达到0.680 |
| `20260731_v6_long_高点后等待退化_独立DenseActor` | rejected | epoch-11有真实Dense收益，但后期再次退化到timeout=0.680 |
| `20260731_v6_epoch11_固定200场配对复核` | rejected candidate | full提高`0.110`，但timeout达到`0.120`且平均步数为5A的`2.59`倍 |
| `20260731_v7_前进奖励无净提升_独立DenseActor` | rejected | epoch-8与冻结5A的full success相同，目标冲突未解除 |
| `20260731_v8_Critic危险加速退化_独立DenseActor` | rejected | 移除统一减速后复现危险加速，确认未归一化Q与safe-only anchor无法限制Critic动作外推 |
| `20260731_v9_约束稳定但无学习增益_独立DenseActor` | rejected | 抑制危险加速和等待退化，但Actor基本停留在5A，未产生学习增益 |
| `20260801_简化TD3参数实验A` | rejected | 随机Critic未预训练且学习率过高，Actor在一轮内退化为危险加速 |
| `20260801_简化TD3整体调参实验B` | ready | 简化并校准基础reward，恢复Critic预训练与低学习率；先完成4轮Critic审计再决定是否解冻Actor |

v6 epoch-11的200场复核已完成：收益真实，但没有通过独立Dense Actor的timeout和效率验收。v8确认不能直接放开Critic；v9的单边危险加速上限虽然阻止两类退化，却没有产生学习增益。实验A进一步证明高学习率下随机Critic会立即破坏5A。当前只保留实验B作为下一条有效路线：先验证简单reward下的Critic动作排序，再恢复Actor训练。
