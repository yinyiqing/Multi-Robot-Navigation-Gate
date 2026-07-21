# D4 Interaction-Risk Probe: Frozen 5D

状态：`diagnostic complete`。这是从 fixed-v1 train 池抽取的 60 场诊断 probe，不是论文 validation，不用于报告显著性或选择最终模型。

## 协议

- 冻结原始 5D Actor，不训练。
- edge-1 场景按同步名义路径最小间距分为 deep/close/margin。
- 每档 standard/dense 各 10 场，共 60 场，固定 seed `20260721`。
- 记录 1786 帧 Actor 输入、动作、位置与终止状态。
- 进入 `1.2 m` 接近区时，用连续位置差估计闭合速度和 TTC，步长按 `0.2 s`。

## 结果

| Group | Episodes | Agent success | Collision | Full success | Actual pair min distance median |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 60 | 0.830 | 0.170 | 0.517 | 0.723 m |
| deep `[0.0, 0.4)` | 20 | 0.690 | 0.310 | 0.150 | 0.429 m |
| close `[0.4, 0.6)` | 20 | 0.860 | 0.140 | 0.550 | 0.727 m |
| margin `[0.6, 0.9)` | 20 | 0.940 | 0.060 | 0.850 | 0.892 m |

几何风险分层呈现稳定难度梯度，比单独的 `conflict_edge_count == 1` 更有解释力。

| Encounter metric at about 1.2 m | Full-success episodes | Failed episodes |
| --- | ---: | ---: |
| encounters observed | 29 | 27 |
| closing speed median | 0.428 m/s | 0.730 m/s |
| TTC median | 2.547 s | 1.591 s |
| mean linear command, median | 1.000 | 1.000 |
| mean linear command, mean | 0.827 | 0.943 |
| mean lidar minimum, median | 0.995 m | 1.035 m |

成败组在接近区的静态距离和最小激光值接近，但闭合速度和 TTC 差异明显。失败组同时保持更高的前进命令，支持“单帧激光能看到近，但无法直接表示靠近速度”这一诊断。

29 个失败 episode 全部有离线冲突对的至少一名成员触发碰撞，其中 18 场是冲突对两名成员都触发碰撞。其余 11 场无法仅根据当前日志排除墙或第三台机器人，因此不宣称全部为冲突对互撞。

## 决策

- 不继续当前单帧 residual TD3 的 epoch、seed、容量或 anchor 调参。
- 下一步先在相同 60 场上运行固定优先级的冲突对让行 oracle，只验证场景是否可通过显式协调解决。
- 若 oracle 能降低 deep 碰撞，再给 Actor/Critic 增加局部相对速度/TTC 和邻居动作条件；否则先查场景可解性和低层控制。

结构化逐场结果见 `summary.json`，原始轨迹和日志以 gzip 保留。
