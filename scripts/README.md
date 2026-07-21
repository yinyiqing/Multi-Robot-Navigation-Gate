# 脚本索引

根目录脚本包含多年实验入口。日常操作优先使用统一入口，只有复现历史实验时才直接调用长脚本名。

## 当前入口

```bash
bash scripts/experiment.sh list
bash scripts/experiment.sh status
bash scripts/experiment.sh start eval-5d-standard
bash scripts/experiment.sh stop eval-5d-standard
```

`experiment.sh` 只暴露当前协议允许执行的实验。planned 训练不会提前加入入口。

## 当前映射

| 实验 ID | 底层脚本 | 状态 |
| --- | --- | --- |
| `eval-5d-standard` | `start/stop_test_detached_multi_stage2_to_5d_geo_critic_from_5a_guarded_best.sh` | current |
| `diag-5d-random-dense` | `start/stop_test_detached_dense5_random_5d.sh` | diagnostic |

固定数据 baseline 使用 `start_test_fixed_v1_5d.sh standard|dense`，两组可通过独立端口并行运行；对应停止入口为 `stop_test_fixed_v1_5d.sh`。

五个 fixed moderate case 暂时仍通过通用 curriculum 测试脚本运行；在 D1-D3 数据协议完成前不把它包装成正式入口。

## 长脚本命名

历史脚本遵循：

```text
start|stop _ training|test _ detached _ <historical-run-name>.sh
```

这些名字记录当时的训练谱系，但不再用于论文术语。新工作使用：

- 实验：`eval-5d-standard`
- 模型：`generalist-5d`
- 场景池：`standard`, `dense`

对应实际权重见 [模型注册表](../TD3/MODEL_REGISTRY.md)。

## 脚本分级

- `experiment.sh`：当前稳定入口。
- `generate_*`, `publish_*`, `observe_*`：环境工具。
- `generate_fixed_scenarios.py`：离线生成 standard/dense 固定候选清单。
- `validate_fixed_scenarios.py`：用策略无关的 Gazebo reset 检查筛选清单。
- `audit_fixed_scenarios.py`：检查固定清单 schema、split 互斥性和 Gazebo 标记。
- `build_interaction_views.py`：从冻结 manifest 确定性生成交互强度训练/验证视图，不改变原始场景。
- `start_training_fixed_v1_standard_expert.sh`：从 5D warm-start 训练 standard expert，可通过环境变量设置短检查或正式训练轮数。
- `start_training_fixed_v1_standard_expert_v2.sh`：完整加载 5D Actor/Critic，保留 0.8/0.2 reward，并使用 Actor anchor 的 v2 实验入口。
- `start_training_fixed_v1_standard_expert_v3.sh`：只验证 timeout terminal 和 Critic 更新比例修复的 3-epoch v3 入口。
- `start_validation_compare_fixed_v1_standard_v3.sh`：在完整 500 场 standard validation 上顺序比较原始 5D 与 v3 epoch 2。
- `start/stop_training_fixed_v1_edge1_residual_pilot.sh`：冻结 5D 主体，在平衡 edge-1 视图上先预热 Critic、再训练 bounded residual 的受控 pilot。
- `start/stop_training_fixed_v1_edge1_conservative_residual_v2.sh`：复用 edge-1 epoch 1 Critic，以归一化 Q 和基础动作约束训练单轮 conservative residual。
- `build_interaction_risk_views.py`：按同步路径最小间距将 edge-1 场景派生为 deep/close/margin 三档几何风险视图。
- `analyze_interaction_risk_probe.py`：回连风险 probe 的 manifest、episode 结果和逐帧轨迹，统计实际冲突对间距、闭合速度和 TTC。
- `compare_interaction_probe_summaries.py`：按 scenario ID 和几何风险层配对比较两次 probe，并计算只在指定风险层采用候选策略的诊断上限。
- `start/stop_test_interaction_risk_probe_5d.sh`：在 60 场均衡风险 probe 上运行冻结 5D，并可选记录逐帧轨迹 JSONL。
- `start/stop_test_interaction_risk_yield_oracle.sh`：在同一 probe 上用特权冲突对标签运行固定优先级停车让行上限。
- `start/stop_test_temporal_interaction_probe_5d.sh`：重放同一 60 场 probe，记录单机激光、位姿和时间戳，用于验证自运动补偿的时序闭合速度/TTC。
- `stop_training_fixed_v1_standard_expert.sh`：停止 standard expert 训练进程组。
- 训练 checkpoint 会按 validation 协议隔离 best，并在每轮验证后保存独立的 `epoch_NNN` 模型快照。
- 多机器人训练中 timeout transition 记为 terminal，Critic 更新按有效 agent samples 归一化；旧训练结果不与修复后结果混合。
- 当前映射表中的 start/stop：受支持的底层入口。
- 其他 start/stop：历史复现入口，不代表当前建议。
- residual/gate 脚本：脚手架；论文协议允许前不得执行。

所有后台脚本必须使用独立 PID 文件和 ROS/Gazebo 端口。停止脚本只能终止对应 PID 的进程组，不能使用全局 `pkill`。
