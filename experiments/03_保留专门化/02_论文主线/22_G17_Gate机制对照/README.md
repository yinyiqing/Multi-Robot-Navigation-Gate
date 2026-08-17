# G17 Gate机制对照

本实验只评估冻结主线中的运行时机制，不训练或修改任何 Actor、Gate 和阈值。

## 目的

在与 G17 相同的 120 个五车完整场景上，补充两个固定策略对照：

- `epoch17_always_on`：全程使用 `avoidance-epoch17`，检验 F-A1 的收益是否只是避障 Actor 本身带来的；
- `rule_2m_privileged`：使用仿真真值距离小于 2.0 m 时切换到 `avoidance-epoch17`，这是不可部署的特权距离规则，只作机制诊断，不称为 Oracle，也不作为方法结果。

G17 中已有的 `5A` 和 `F-A1` 结果不重复运行，最终由同一 manifest、同一 seed、同一统计协议比较。

## 固定协议

- manifest：`datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz`
- 120 个固定场景，五车；每个策略两个 seed：`20260824`、`20260825`
- 每个策略共 240 episodes；不读取 sealed test
- CPU 运行，使用独立 ROS/Gazebo 端口 `17623/17723`
- 运行前等待 `/tmp/local_critic_multi_robot_training.lock`，不影响已有训练

## 结果解释

若 always-on 接近 F-A1，说明 Gate 的收益有限；若 always-on 成功率较低或步数明显增加，而 F-A1 保持优势，才支持“按交互状态在线路由”这一机制。2m 规则只用于衡量可部署 Gate 与特权状态规则之间的差距。

结果和实时日志分别保存在项目根目录的 `logs/active/g17-gate-mechanism/`；完整成功后自动归档到 `logs/archive/validation/g17_gate_mechanism/`。

## 2026-08-17 结果

四个运行均一次完成并通过 120 场 manifest、shape 和终止计数审计。两个 seed 合并后的
240 场结果为：

| 方法 | Full success | Agent success | Collision | Timeout | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5A（复用 G17） | 0.5000 | 0.8075 | 0.1925 | 0.0000 | 21.63 |
| epoch17 always-on | 0.2667 | 0.7608 | 0.0817 | 0.5708 | 194.95 |
| F-A1（复用 G17） | 0.5958 | 0.8517 | 0.1483 | 0.0000 | 31.61 |
| 2m privileged distance rule | 0.7042 | 0.9050 | 0.0917 | 0.0083 | 33.74 |

配对 McNemar exact 检验：

- F-A1 相对 5A：42 场改善、19 场退化，`p=0.00444`；
- F-A1 相对 epoch17 always-on：100 场改善、21 场退化，`p=1.61e-13`；
- 2m 特权距离规则相对 F-A1：46 场改善、20 场退化，`p=0.00186`。

该结果确认 epoch17 不能全程承担导航；F-A1 的收益来自条件路由而非简单常开避障 Actor。
同时，F-A1 与不可部署的特权距离规则之间仍有 10.8 个百分点的显著差距，因此当前 Gate
有效但尚未充分逼近特权交互状态。下一步应先对 F-A1 相对 2m 规则的逐帧误触发、漏触发
和切换时序做诊断，再决定是否重训 Gate，不直接增加训练预算。
