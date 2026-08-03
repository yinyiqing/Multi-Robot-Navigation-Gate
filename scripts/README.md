# 脚本索引

研究决策以[论文主线](../experiments/03_保留专门化/02_论文主线/README.md)为准。

## 当前入口

```bash
bash scripts/experiment.sh list
bash scripts/experiment.sh status
```

当前批准的实验是`actor-b-single-edge-residual`，但R0/R1实现尚未完成，因此统一入口
暂不开放start/stop。没有经过短pilot检查前不得直接拼接历史训练脚本启动长跑。

## 当前模型名称

- `generalist-5a`：冻结Actor A和Actor B的base；
- `interaction-teacher-epoch16`：冻结的单冲突动作教师；
- `actor-b-single-edge-residual`：待实现的独立Actor B；
- `single-to-multi-conflict-gate`：Actor B通过后再训练的Gate。

## 历史脚本

根目录仍有早期baseline、数据采集、感知和Gate诊断工具。它们只用于复核归档证据，
不代表当前建议。以下旧训练入口已删除，防止误启动：

- epoch-16整网全程续训及其200场比较；
- 5D + 零初始化Residual；
- 旧moderate Residual测试。

通用的场景生成、manifest审计、结果分析、机器人感知和Gate特征代码继续保留，因为
新主线仍需要这些基础设施。

## 运行规则

- Actor B和Gate训练只能读取0-edge与单冲突manifest。
- exact-edge-2及以上只能用于零样本validation/test。
- 每个后台入口必须使用独立PID和ROS/Gazebo端口。
- stop脚本只能停止对应PID的进程组，禁止全局`pkill`。
- 运行日志必须记录模型哈希、manifest哈希、seed、commit和完整配置。
