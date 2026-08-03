# 执行手册

当前研究协议见[论文主线](experiments/03_保留专门化/02_论文主线/README.md)。

## 当前状态

```text
主线：从单冲突学习到多冲突组合泛化
当前：Residual Actor B的R0/R1短pilot实现前检查
训练：暂未开放
```

Actor B固定为`冻结5A + epoch-16指导的单冲突Residual`。禁止启动完整dense Actor、
epoch-16整网续训或5D零初始化Residual。

## 环境检查

```bash
cd /home/jiutian/Local-Critic-Multi-Robot-Navigation-gate
nvidia-smi
df -h .
bash scripts/experiment.sh status
git status --short --branch
```

脚本会按需加载ROS、`env.python.sh`和catkin工作空间。当前允许列出协议和检查受管
进程，但没有可启动训练：

```bash
bash scripts/experiment.sh list
bash scripts/experiment.sh status
```

## 数据边界

- Actor B训练：0-edge和exact-edge-1；
- Gate训练：0-edge和exact-edge-1；
- 零样本validation：exact-edge-2及以上；
- sealed test：所有模型和阈值冻结后一次性读取。

任何多冲突数据泄漏到训练都会使当前论文主张失效。

## 运行记录

正式run至少记录：实验ID、模型及SHA256、manifest及SHA256、seed、Git commit、
episode数、指标口径和完整环境变量。缺少任一项只能作为diagnostic。

当前运行产物写入`logs/`、`TD3/checkpoints/`、`TD3/results/`和`TD3/runs/`；形成结论
后只把必要证据归档到对应实验目录。

## 进程规则

- 每个后台入口使用独立PID文件和ROS/Gazebo端口。
- 只使用配套stop命令终止自己启动的进程组。
- 禁止全局`pkill python`、`pkill roslaunch`或`pkill gzserver`。
- 正式评测默认headless；RViz只用于定性检查。

## 提交前

```bash
git diff --check
source env.python.sh
python -m unittest discover -s tests -v
```

提交消息使用`英文前缀：中文主体`格式。
