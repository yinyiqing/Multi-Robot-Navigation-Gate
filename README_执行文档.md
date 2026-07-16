# 执行手册

本文档只描述当前机器上的稳定运行流程。历史实验命令见 [脚本索引](scripts/README.md)，研究决策见 [论文协议](experiments/04_保留专门化/05_论文主线/README.md)。

当前阶段：`D0/D1`。只允许场景生成器开发和 generalist baseline 评估；在 D1-D3 完成前不启动 residual 或 gate 训练。

## 1. 环境

项目根目录：

```bash
cd /home/jiutian/Local-Critic-Multi-Robot-Navigation-gate
```

训练和评测脚本会自动加载：

```bash
source /opt/ros/noetic/setup.bash
source env.python.sh
source catkin_ws/devel_isolated/setup.bash
```

首次部署或依赖变更时使用：

```bash
sudo bash scripts/install_ros_noetic_system.sh
bash scripts/build_readme_workspace.sh
```

正常实验前检查：

```bash
nvidia-smi
df -h .
bash scripts/experiment.sh status
git status --short --branch
```

## 2. 当前实验入口

列出当前协议允许的实验：

```bash
bash scripts/experiment.sh list
```

启动 5D standard 正式评估：

```bash
bash scripts/experiment.sh start eval-5d-standard
```

停止对应评估：

```bash
bash scripts/experiment.sh stop eval-5d-standard
```

random dense 仅用于 spatial-density 诊断，不进入论文主表：

```bash
bash scripts/experiment.sh start diag-5d-random-dense
bash scripts/experiment.sh stop diag-5d-random-dense
```

不要直接执行 `start_training_detached_dense5_moderate_residual_from_5d.sh`。它只是 residual checkpoint 链路的脚手架，当前训练分布和验证协议尚未完成。

## 3. 配置正式评估

底层脚本通过环境变量接收配置。示例：

```bash
DRL_MULTI_TEST_TARGET_EPISODES=1000 \
DRL_MULTI_TEST_SEED=1000 \
bash scripts/experiment.sh start eval-5d-standard
```

正式 run 至少应记录：

- 实验 ID
- 模型 ID 和实际 artifact 前缀
- scenario ID
- seed 和 episode 数
- Git commit
- outcome metric version
- 完整场景 manifest 或生成参数

如果脚本没有把其中任何一项写入日志，该 run 只能算 diagnostic。

## 4. 监控

查看受管进程：

```bash
bash scripts/experiment.sh status
```

查看最新运行日志：

```bash
ls -lt logs/ | head
tail -f logs/<run-log>.log
```

查看 GPU 和 ROS/Gazebo 进程：

```bash
nvidia-smi
ps -eo pid,etime,cmd | rg 'test_velodyne|train_velodyne|roslaunch|gzserver'
```

TensorBoard：

```bash
source env.python.sh
tensorboard --logdir TD3/runs --bind_all
```

## 5. 运行产物

| 路径 | 内容 | 是否提交 |
| --- | --- | --- |
| `logs/` | 当前 run 的文本日志 | 否 |
| `TD3/checkpoints/` | 训练/测试恢复状态 | 否 |
| `TD3/results/` | `.npy` 统计快照 | 否 |
| `TD3/runs/` | TensorBoard event | 否 |
| `TD3/pytorch_models/` | 本机模型权重 | 默认否 |
| `experiments/<stage>/logs/` | 经筛选的证据日志 | 是 |
| `experiments/<stage>/README.md` | 配置、结果和结论 | 是 |

run 结束后的归档顺序：

1. 确认 episode 数、seed、模型和场景配置。
2. 验证 `success + collision + unresolved = agents * episodes`。
3. 将必要日志移入唯一实验归档位置，不复制多份。
4. 在对应 README 记录结果、口径和结论。
5. 删除本地恢复 state 之前，先确认无需续跑。

## 6. 进程规则

- 每个后台入口使用独立 PID 文件和 ROS/Gazebo 端口。
- 使用配套 stop 命令，只停止该 PID 的进程组。
- 不使用全局 `pkill python`, `pkill roslaunch` 或 `pkill gzserver`，它们可能终止其他实验。
- PID 文件存在但进程不存在时是 stale 状态；确认对应进程确实结束后再删除 PID 文件。
- 测试 Python 已结束但其 roslaunch/Gazebo 的父进程变为 PID 1 时，属于孤立仿真进程，应按端口和父子关系确认后单独停止。

## 7. RViz

训练和正式评估默认 headless。只有定性检查时才连接 X11 并启动观察脚本：

```bash
bash scripts/observe_rviz_multi_standard.sh
```

RViz 观察不作为正式指标，也不要在正式吞吐量测试期间开启。

## 8. 模型身份

不要根据长文件名猜模型角色。统一使用 [模型注册表](TD3/MODEL_REGISTRY.md)：

- `generalist-5d`：当前冻结基线
- `bridge-full-ft`：失败的 full fine-tune
- `bridge-head-only`：失败的 head-only
- `residual-specialist`：planned，当前不应有正式 artifact

新模型文件名不再编码完整训练谱系；来源、数据和超参数写入 manifest。

## 9. 提交

运行中的根目录日志、checkpoint、result 和 TensorBoard 文件由 `.gitignore` 排除。提交前执行：

```bash
git diff --check
source env.python.sh
python -m unittest discover -s tests -v
```

提交应按“实现、实验协议、实验结果”拆分，避免把尚未验证的训练产物与代码混在同一个提交中。
