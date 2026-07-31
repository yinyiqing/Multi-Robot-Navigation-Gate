# Gate 前置验证

这里保存训练可部署 Gate 之前的可解性、风险定义和本机感知验证。当前 D5-G0 直接从这一组的失败边界继续，但不会复用已经读取过的 test 调参。

| 实验 | 状态 | 关键结论 |
| --- | --- | --- |
| `D4_scene_feasibility_probe_s20260724` | complete diagnostic | 固定强交互场景 reset 合法；简单停车规则不是可解性上界 |
| `D4_interaction_risk_probe_5d_s20260721` | complete diagnostic | deep/close 难度与闭合速度、TTC 相关 |
| `D4_interaction_risk_yield_oracle_s20260721` | rejected heuristic | 全局让行改善 deep，却损害 close/margin |
| `D4_temporal_interaction_scan_diff_s20260721` | rejected representation | 扇区最小距离差分缺乏物体关联，误报过高 |
| `D4_lidar_cluster_motion_probe_s20260724` | rejected representation | 二维点簇运动无法可靠排除静态障碍 |
| `D4_lidar_cluster_shape_probe_s20260724` | rejected representation | 手工三维形状特征仍有大量静态误报 |
| `D4_temporal_risk_encoder_20bin_s20260724` | rejected representation | 20-bin 时序输入丢失身份和局部结构 |
| `D4_highres_temporal_risk_encoder_s20260724` | rejected pilot | 180-bin + GRU 有改善，但小样本 holdout 仍不可部署 |
| `20260731_统一靠右诊断_无净收益` | rejected heuristic | 10对固定case总体与5A持平；实际触发case净少完成1台车 |

当前结论：Gate继续暂停。先得到能独立完成完整dense episode且稳定超过5A/5D的Dense Actor，再讨论两个Actor是否互补以及是否值得训练Gate。
