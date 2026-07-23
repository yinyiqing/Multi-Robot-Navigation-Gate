# 脚本索引

根目录脚本包含多年实验入口。日常操作优先使用统一入口，只有复现历史实验时才直接调用长脚本名。

## 当前入口

```bash
bash scripts/experiment.sh list
bash scripts/experiment.sh status
bash scripts/experiment.sh start train-strong-interaction
bash scripts/experiment.sh stop train-strong-interaction
```

`experiment.sh` 只暴露当前协议允许执行的实验。planned 训练不会提前加入入口。

## 当前映射

| 实验 ID | 底层脚本 | 状态 |
| --- | --- | --- |
| `train-strong-interaction` | `start/stop_training_strong_interaction_expert_pilot.sh` | current |
| `eval-5d-standard` | `start/stop_test_detached_multi_stage2_to_5d_geo_critic_from_5a_guarded_best.sh` | historical baseline |

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
- `build_strong_interaction_views.py`：生成 deep 主导、close/margin 约束的固定强交互 pilot 视图。
- `start/stop_training_strong_interaction_expert_pilot.sh`：复制 5D 完整 warm-start，训练独立的 8 帧 GRU 强交互 Actor；原始 5D 弱交互 Actor 不变。
- 旧 standard expert 与 edge-1 residual 入口已退出当前工作流；结论和必要产物保留在实验归档中。
- `build_interaction_risk_views.py`：按同步路径最小间距将 edge-1 场景派生为 deep/close/margin 三档几何风险视图。
- `analyze_interaction_risk_probe.py`：回连风险 probe 的 manifest、episode 结果和逐帧轨迹，统计实际冲突对间距、闭合速度和 TTC。
- `compare_interaction_probe_summaries.py`：按 scenario ID 和几何风险层配对比较两次 probe，并计算只在指定风险层采用候选策略的诊断上限。
- `analyze_temporal_interaction_probe.py`：以其他机器人位置生成评估真值，审计仅使用本机激光和里程计的时序闭合速度/TTC 特征。
- 风险 probe、让行 oracle 和扇区差分 TTC 的运行入口已在结论归档后移除；分析脚本保留用于复核归档数据。
- 训练 checkpoint 会按 validation 协议隔离 best，并在每轮验证后保存独立的 `epoch_NNN` 模型快照。
- 多机器人训练中 timeout transition 记为 terminal，Critic 更新按有效 agent samples 归一化；旧训练结果不与修复后结果混合。
- 当前映射表中的 start/stop：受支持的底层入口。
- 其他 start/stop：历史复现入口，不代表当前建议。
- residual/gate 脚本：脚手架；论文协议允许前不得执行。

所有后台脚本必须使用独立 PID 文件和 ROS/Gazebo 端口。停止脚本只能终止对应 PID 的进程组，不能使用全局 `pkill`。
