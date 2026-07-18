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
- `start_training_fixed_v1_standard_expert.sh`：从 5D warm-start 训练 standard expert，可通过环境变量设置短检查或正式训练轮数。
- `start_training_fixed_v1_standard_expert_v2.sh`：完整加载 5D Actor/Critic，保留 0.8/0.2 reward，并使用 Actor anchor 的 v2 实验入口。
- `stop_training_fixed_v1_standard_expert.sh`：停止 standard expert 训练进程组。
- 训练 checkpoint 会按 validation 协议隔离 best，并在每轮验证后保存独立的 `epoch_NNN` 模型快照。
- 当前映射表中的 start/stop：受支持的底层入口。
- 其他 start/stop：历史复现入口，不代表当前建议。
- residual/gate 脚本：脚手架；论文协议允许前不得执行。

所有后台脚本必须使用独立 PID 文件和 ROS/Gazebo 端口。停止脚本只能终止对应 PID 的进程组，不能使用全局 `pkill`。
