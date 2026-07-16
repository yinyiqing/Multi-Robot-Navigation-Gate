# Local-Critic Multi-Robot Navigation

本项目研究 ROS/Gazebo 中无通信、局部观测的多机器人导航。执行阶段每台机器人只使用本车 24 维观测；训练阶段允许 Critic 使用局部邻居几何信息。

## 当前研究问题

已有 `5D` 策略能完成普通五车导航，但在同步交叉、汇流和对向通道等高交互场景中明显退化。继续微调整个 Actor 又会覆盖已有能力。因此当前论文主线是：

```text
冻结 5D generalist
  -> 训练有动作幅度约束的 dense residual specialist
  -> 审计两个策略是否互补
  -> 互补性成立后训练本地时序 gate
```

当前状态为 `D0`：方法与实验协议已冻结，正在推进 `D1` 场景生成器。`D1-D3` 完成前不启动 residual 或 gate 训练。唯一决策源是 [论文协议](experiments/04_保留专门化/05_论文主线/README.md)。

## 当前结论

- `generalist-5d` 是当前冻结的普通导航基线。
- 随机缩小起点区域得到的 random dense 同时缩短了目标距离，只能作为 spatial-density 诊断，不能作为正式 dense benchmark。
- 五个 fixed moderate case 能暴露同步路径冲突，但只作为 canonical held-out 测试，不作为唯一训练分布。
- full Actor fine-tune、head-only 以及历史 `5A + 5D` 双 Actor 切换都没有形成可靠的能力增益。
- 正式训练前必须先实现任务距离匹配的 low/medium/high interaction-density 数据集，并完成修复口径下的 generalist baseline。

## 从这里开始

```bash
# 查看受支持的当前实验及状态
bash scripts/experiment.sh list

# 查看正在运行的受管实验
bash scripts/experiment.sh status

# 启动/停止 5D standard 正式评估
bash scripts/experiment.sh start eval-5d-standard
bash scripts/experiment.sh stop eval-5d-standard
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
