# Gate 前置验证

这里保存可解性、风险定义和本机感知的历史诊断。结论仍可作为方法边界，但本目录
不再定义当前下一步；当前协议见[论文主线](../../README.md)。

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
| `20260801_固定路径错峰可解性检查` | complete diagnostic | 0.6 m中心间距下161/200场只靠起步错峰可解；主要缺口是Actor未学会通行顺序 |

这些诊断否定了统一停车、统一靠右和手工TTC阈值，但不否定局部技能组合。当前冻结
5A和epoch-16，直接把二者作为普通导航Actor与条件避障Actor，只开发可部署Gate。
