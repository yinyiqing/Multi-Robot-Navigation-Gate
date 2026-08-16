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
