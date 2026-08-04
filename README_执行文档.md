# 执行手册

当前状态见[PROJECT_STATUS](PROJECT_STATUS.md)，研究协议见
[论文主线](experiments/03_保留专门化/02_论文主线/README.md)。

## 当前状态

```text
Actor N：generalist-5a，冻结
Actor I：interaction-epoch16，冻结
当前工作：可部署在线Gate
Actor训练：关闭
sealed test：未开放
```

`2.0 m`真值切换只用于oracle上界和监督，不是部署输入。任何Actor重训、数据边界变化或
sealed test读取都必须先修改论文主线协议。

## 环境检查

```bash
cd /home/jiutian/Local-Critic-Multi-Robot-Navigation-gate
nvidia-smi
df -h .
bash scripts/experiment.sh list
bash scripts/experiment.sh status
git status --short --branch
```

脚本会按需加载ROS、`env.python.sh`和catkin工作空间。统一入口当前只列出状态，不开放
训练命令；历史`start_*`脚本存在是为了复现，不代表批准运行。

## 数据边界

- Gate训练：导航train内部的0-edge和corrected single-edge；
- Gate参数选择：互斥的小validation；
- multi-edge：冻结后的泛化边界评估；
- sealed test：所有模型、特征和阈值冻结后一次性读取。

epoch-16原训练集经完整路径复审有11场实际为edge-2。若论文保留严格single-to-multi
零样本主张，必须在test前完成corrected条件Actor重训；否则必须收窄主张。

## 运行记录

正式run至少记录：

- 实验ID与目的；
- 模型artifact及SHA-256；
- manifest、split及SHA-256；
- seed和Git commit；
- 完整环境变量；
- episode数、最大步数和指标口径；
- 预先规定的准入和停止条件。

缺少任一项只能作为diagnostic。当前运行产物写入`logs/`、`TD3/checkpoints/`、
`TD3/results/`和`TD3/runs/`；形成结论后把必要事实写入对应实验README。

## 进程规则

- 每个后台入口使用独立PID文件和ROS/Gazebo端口；
- 启动前检查现有Gazebo、训练和GPU进程；
- 只使用配套stop命令终止自己启动的进程组；
- 禁止全局`pkill python`、`pkill roslaunch`或`pkill gzserver`；
- 正式评测默认headless，RViz只用于定性检查。

## 结果解释

- 不同manifest的full success不能直接比较；
- partial结果必须报告实际场景数；
- `best` checkpoint不表示当前全局最优；
- oracle必须明确标注privileged；
- `unresolved`是episode结束时既未成功也未碰撞的agent；
- `full success`要求五台车全部到达；
- `success + collision + unresolved = agents * episodes`。

## 提交前

```bash
git diff --check
source env.python.sh
python -m unittest discover -s tests -v
```

提交消息使用`英文前缀：中文主体`格式。
