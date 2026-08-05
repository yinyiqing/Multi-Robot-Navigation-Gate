# 脚本索引

当前状态见[PROJECT_STATUS](../PROJECT_STATUS.md)，研究决策以
[论文主线](../experiments/03_保留专门化/02_论文主线/README.md)为准。

## 统一入口

```bash
bash scripts/experiment.sh list
bash scripts/experiment.sh status
```

G11-B1采集和G11-B2主seed聚合训练均已完成。当前入口保留用于复核；下一步是登记后的
固定50场闭环pilot：

```bash
bash scripts/run_g11_b_aggregated_training.sh 20260804
```

5A、epoch-16和A1主seed Gate均冻结，任何Actor训练脚本都不是当前入口。协议见
[`G11_B_student_rollout_v1`](../experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1/README.md)。

## 当前模型ID

- `generalist-5a`：冻结普通导航Actor；
- `interaction-epoch16`：冻结条件避障Actor；
- `learned-gate-g2a`：历史未过准入Gate基线；
- `deployable-interaction-gate`：G11-B2聚合Gate已训练，等待固定50场闭环pilot。

## 可复用工具

- `generate_fixed_scenarios.py`、`audit_*`、`build_*view*`：数据生成和审计；
- `evaluate_robot_detector.py`、`evaluate_robot_tracking.py`：Gate感知前端评估；
- `compare_actor_validation.py`及分析脚本：固定结果比较；
- `start_validation_*`：必须核对模型、manifest和split后才能复现历史评测。

## 历史入口

根目录保留大量早期baseline、Actor训练、感知和Gate脚本以支持复现。文件存在不表示当前
建议。以下家族均已关闭：

- `start_training_independent_dense_actor_*`；
- `start_training_dense_simple_td3_hparam_*`；
- `start_training_full_actor_edge1_from_5a.sh`；
- epoch-16整网续训、Residual、pair和controlled-ego相关历史入口；
- 旧stage课程训练入口。

历史脚本运行前必须先在[实验注册表](../experiments/EXPERIMENT_REGISTRY.md)确认状态，并
显式说明是复现而非当前训练。

## 运行规则

- Gate训练只使用当前协议允许的train split；
- sealed test在模型和阈值冻结前禁止读取；
- 每个后台入口使用独立PID和ROS/Gazebo端口；
- stop脚本只能停止对应PID进程组，禁止全局`pkill`；
- 日志必须记录模型/manifest哈希、seed、commit、配置和停止条件。
