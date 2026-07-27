# Local-Critic Multi-Robot Navigation

本项目研究 ROS/Gazebo 中无通信、局部观测的多机器人导航。两个导航 Actor 执行时只使用本车 24 维观测；训练阶段允许 Critic 使用局部邻居几何信息。后续 Gate 只能使用本机传感器，不得读取其他机器人真值。

## 当前研究问题

已有 Actor 能完成普通五车导航，但在同步交叉、汇流和对向通道等高交互状态中明显退化。继续微调整个 Actor 又会覆盖已有能力。当前已经得到两个角色不同的 Actor：

- `generalist-5a`：普通导航 Actor。
- `strong-interaction-5a-balanced`：只在机器人紧迫交互状态中调用的条件交互 Actor，不是全程独立导航策略。

当前论文主线是：

```text
冻结 5A 普通导航 Actor 和 epoch-16 条件交互 Actor
  -> 用本机三维激光区分机器人与静态障碍
  -> 对机器人目标估计距离、闭合速度和 TTC
  -> 建立可部署的启发式 Gate 基线
  -> 只训练 Gate，在两个冻结 Actor 之间进行状态级切换
```

当前处于 `D5-G0`：Actor 开发暂停，正在解决 Gate 的可观测性前提，即机器人与静态障碍的区分。唯一决策源是 [论文协议](experiments/04_保留专门化/05_论文主线/README.md)。

## 当前结论

- 5A/5D 在同一 248 个弱交互 validation 场景上的 full success 为 `0.8750/0.8710`，无显著差异；选择 5A 是为了匹配条件交互 Actor 的训练分布。
- 冻结 5A 与条件交互 Actor 的 oracle 组合在 140 个强交互 validation 场景上将 full success 从 `0.421` 提高到 `0.700`，重复验证已复现。
- 条件交互 Actor 全程独立运行不符合其训练契约；论文方法是状态级策略选择，不再声称两个 Actor 分别对应 standard/dense 场景。
- 当前 `2.0 m` oracle 使用其他机器人真实位置，只是不可部署上界，不是最终 Gate。
- 20 维激光不区分机器人、墙和箱子。二维运动点簇、三维手工形状和现有时序风险编码均未达到准入线，不能直接接入 Gate。
- 两个 Actor 从现在起冻结。下一步只开发机器人感知与 Gate，不继续微调 Actor，不读取 test。

## 从这里开始

```bash
# 查看受支持的当前实验及状态
bash scripts/experiment.sh list

# 查看正在运行的受管实验
bash scripts/experiment.sh status

# 当前阶段不启动 Actor 训练；具体 Gate 准入顺序见论文协议
```

当前评估使用互斥结局口径：`success + collision + unresolved = agents * episodes`，同一步同时到达并碰撞时按碰撞处理。2026-07-16 以前的旧口径结果仅用于历史诊断。

## 仓库导航

| 位置 | 内容 | 状态 |
| --- | --- | --- |
| [论文协议](experiments/04_保留专门化/05_论文主线/README.md) | 研究问题、dense 定义、决策门和实验矩阵 | 唯一当前协议 |
| [实验索引](experiments/README.md) | 各阶段作用、状态和阅读顺序 | 当前索引 |
| [模型注册表](TD3/MODEL_REGISTRY.md) | 短模型 ID、实际文件名和使用限制 | 当前索引 |
| [脚本索引](scripts/README.md) | 当前入口、历史脚本和命名规范 | 当前索引 |
| [执行手册](README_执行文档.md) | ROS、Gazebo、后台进程和环境配置 | 运维参考 |
| `TD3/` | 环境、模型、训练和评测代码 | 源代码 |
| `catkin_ws/` | ROS 包、机器人模型和 Gazebo 插件 | 源代码/构建区 |
| `logs/`, `TD3/results/`, `TD3/runs/`, `TD3/checkpoints/` | 当前机器运行产物 | 本地，不提交 |

历史实验目录保留原名和原始 artifact 名，避免破坏复现。新的文档、命令和论文统一使用注册表中的短 ID。
