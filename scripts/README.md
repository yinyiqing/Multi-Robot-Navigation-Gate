# 脚本索引

当前状态见[PROJECT_STATUS](../PROJECT_STATUS.md)，研究决策以
[论文主线](../experiments/03_保留专门化/02_论文主线/README.md)为准。

## 统一入口

```bash
bash scripts/experiment.sh list
bash scripts/experiment.sh status
```

G11-B1/B2、G11-C、G11-D2和G12-P1/R1均已完成。当前已登记的长跑入口是避障Actor
epoch 17-20固定续训：

```bash
bash scripts/start_avoidance_actor_e17_e20.sh
```

以下旧命令只保留用于受控复现，不是当前执行顺序：

```bash
bash scripts/run_g11_b_aggregated_training.sh 20260804
bash scripts/run_g11_d_seed_replication.sh
/usr/bin/python3 scripts/build_g11_d2_admission_view.py
bash scripts/experiment.sh start gate-g11-d2-admission
bash scripts/experiment.sh queue actor-g12-capacity-pilot
```

5A和原epoch-16 artifact均冻结且不得覆盖。当前唯一主方法Actor训练授权，是从
epoch-16完整checkpoint独立续训epoch 17-20；候选通过matched复测前不替换原模型。
G12-P1/R1只作训练稳定性诊断，checkpoint不得作为其他路线warm start。Gate协议见
[`G11_B_student_rollout_v1`](../experiments/03_保留专门化/02_论文主线/11_可部署在线Gate研究/G11_B_student_rollout_v1/README.md)，
容量对照见[`G12`](../experiments/03_保留专门化/02_论文主线/12_参数匹配单Actor容量对照/README.md)。

## 当前模型ID

- `generalist-5a`：冻结普通导航Actor；
- `interaction-epoch16`：冻结避障Actor fallback；独立epoch 17-20续训分支正在登记执行；
- `learned-gate-g2a`：历史未过准入Gate基线；
- `deployable-interaction-gate`：G11-B2聚合Gate在G11-D2通过导航准入，但效率准入失败。

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
- epoch-16无边界续训、Residual、pair和controlled-ego相关历史入口；当前仅允许已登记的
  epoch 17-20固定续训入口；
- 旧stage课程训练入口。

历史脚本运行前必须先在[实验注册表](../experiments/EXPERIMENT_REGISTRY.md)确认状态，并
显式说明是复现而非当前训练。

## 运行规则

- Gate训练只使用当前协议允许的train split；
- sealed test在模型和阈值冻结前禁止读取；
- 每个后台入口使用独立PID和ROS/Gazebo端口；
- stop脚本只能停止对应PID进程组，禁止全局`pkill`；
- 日志必须记录模型/manifest哈希、seed、commit、配置和停止条件。
