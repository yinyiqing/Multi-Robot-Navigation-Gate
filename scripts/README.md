# 脚本索引

根目录脚本包含多年实验入口。日常操作优先使用统一入口，只有复现历史实验时才直接调用长脚本名。

## 当前入口

```bash
bash scripts/experiment.sh list
bash scripts/experiment.sh status
```

`experiment.sh`只暴露当前协议允许执行的实验。两个Actor已经冻结，当前没有可启动的Actor训练。G0数据与模型协议实现完成前，`gate-robot-perception`只登记为planned，不提前加入启动入口。

## 当前映射

| 实验 ID | 底层脚本 | 状态 |
| --- | --- | --- |
| `gate-robot-perception` | 待G0数据协议实现后登记 | planned；当前工作 |
| `eval-heuristic-interaction-gate` | 待G0/G1通过后实现 | planned |
| `train-interaction-gate` | 待G2通过后实现 | planned |

历史fixed-v1 baseline、强交互Actor训练和点云probe仍保留底层脚本用于复现，但不再属于统一入口支持的当前实验。

## 长脚本命名

历史脚本遵循：

```text
start|stop _ training|test _ detached _ <historical-run-name>.sh
```

这些名字记录当时的训练谱系，但不再用于论文术语。新工作使用：

- 实验：`gate-robot-perception`
- 模型：`generalist-5a`, `strong-interaction-5a-balanced`, `interaction-gate`
- 场景池：`standard`, `dense`

对应实际权重见 [模型注册表](../TD3/MODEL_REGISTRY.md)。

## 脚本分级

- `experiment.sh`：当前稳定入口。
- `generate_*`, `publish_*`, `observe_*`：环境工具。
- `generate_fixed_scenarios.py`：离线生成 standard/dense 固定候选清单。
- `validate_fixed_scenarios.py`：用策略无关的 Gazebo reset 检查筛选清单。
- `audit_fixed_scenarios.py`：检查固定清单 schema、split 互斥性和 Gazebo 标记。
- `build_strong_interaction_curriculum.py`：生成 close→mixed→deep 三阶段固定课程及统一validation。
- `start/stop_training_strong_interaction_curriculum_stage1.sh`：已失败的历史Stage 1复现入口，不得作为当前Actor训练启动。
- 旧 standard expert 与 edge-1 residual 入口已退出当前工作流；结论和必要产物保留在实验归档中。
- `build_interaction_risk_views.py`：按同步路径最小间距将 edge-1 场景派生为 deep/close/margin 三档几何风险视图。
- `analyze_interaction_risk_probe.py`：回连风险 probe 的 manifest、episode 结果和逐帧轨迹，统计实际冲突对间距、闭合速度和 TTC。
- `compare_interaction_probe_summaries.py`：按 scenario ID 和几何风险层配对比较两次 probe，并计算只在指定风险层采用候选策略的诊断上限。
- `start/stop_validation_strong_actor_pair.sh`：已完成的历史诊断入口；其中条件交互Actor全程运行不符合当前训练契约。
- `build_weak_interaction_validation.py`：从standard/dense validation确定性派生`conflict_edge_count=0`的弱交互对照view。
- `start/stop_validation_weak_interaction_5a.sh`：在固定弱交互view上补测5A，与已有5D结果公平比较；不读取test。
- `analyze_temporal_interaction_probe.py`：以其他机器人位置生成评估真值，审计仅使用本机激光和里程计的时序闭合速度/TTC 特征。
- `start/stop_lidar_cluster_sensor_probe_5d.sh`：`shape`在固定30场sensor probe上记录体素降采样XYZ；`highres-holdout`在互斥的30场holdout上只记录180-bin前视激光，均不改变Actor输入。
- `analyze_lidar_cluster_probe.py`：用本机点云、里程计和时间戳进行点簇关联与CPA/TTC估计；其他机器人轨迹只作为离线评分真值。
- `analyze_lidar_cluster_shape_probe.py`：在独立scenario划分上审计三维高度/尺寸特征能否区分真实机器人点簇与静态环境点簇。
- `train_temporal_risk_probe.py`：用仿真CPA/TTC privileged标签监督同输入的单帧MLP与8帧GRU风险编码器，并在scenario级独立test上比较。
- `train_highres_temporal_risk_probe.py`：用旧30场的180-bin等价点云投影做train/validation，只在互斥的新30场180-bin holdout上做最终test。
- 风险 probe、让行 oracle 和扇区差分 TTC 的运行入口已在结论归档后移除；分析脚本保留用于复核归档数据。
- 训练 checkpoint 会按 validation 协议隔离 best，并在每轮验证后保存独立的 `epoch_NNN` 模型快照。
- 多机器人训练中 timeout transition 记为 terminal，Critic 更新按有效 agent samples 归一化；旧训练结果不与修复后结果混合。
- 当前没有受支持的start/stop训练入口。
- 其他start/stop均为历史复现入口，不代表当前建议。
- G0-G2通过前不得实现或启动learned Gate训练。

所有后台脚本必须使用独立 PID 文件和 ROS/Gazebo 端口。停止脚本只能终止对应 PID 的进程组，不能使用全局 `pkill`。
