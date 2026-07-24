# Interaction Risk V1

这是 fixed-v1 原始 manifest 的只读派生视图，不生成、删除或修改原始场景。所有场景均满足 `conflict_edge_count == 1`，再按策略无关的同步路径最小间距分层。

## 风险定义

| Risk | Minimum synchronized path separation | Physical meaning |
| --- | ---: | --- |
| `deep` | `[0.0, 0.4) m` | 标称等速路径存在深度冲突 |
| `close` | `[0.4, 0.6) m` | 接近双机器人外接圆尺寸 |
| `margin` | `[0.6, 0.9) m` | 进入安全裕度，但不一定发生几何重叠 |

Pioneer 3DX 上层外形约为 `0.442 x 0.381 m`，外接圆半径约 `0.292 m`，双机器人外接圆直径和约 `0.584 m`。这些阈值来自机器人尺寸与原始 `0.9 m` 冲突阈值，不读取任何策略成绩。

## 数据量

| Risk | Train available | Probe | Validation |
| --- | ---: | ---: | ---: |
| `deep` | 951 | 20 | 156 |
| `close` | 437 | 20 | 68 |
| `margin` | 1172 | 20 | 199 |

每档 probe 从原 train 池选取 standard/dense 各 10 场。根目录 `probe.json.gz` 合并三档共 60 场，便于只启动一次仿真。probe 与 validation ID 无交叉。

`sensor_probe.json.gz` 从每档的 standard/dense 各取 5 场，共 30 场，用于不改变 Actor 的传感器特征快速审计。

`sensor_holdout.json.gz` 使用每档的 standard/dense 剩余各 5 场，共 30 场，与 `sensor_probe.json.gz` 完全互斥。它只用于高分辨率时序风险编码的最终 test，不参与训练、validation或阈值选择。

重新生成：

```bash
source env.python.sh
scripts/build_interaction_risk_views.py \
  --output experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/interaction_risk_v1 \
  --probe-per-pool 10 \
  --seed 20260721
```

Combined probe SHA-256:

```text
13d6abc630c8c52565981f79b03926d944a36117a35fcdc9902bcf91e041714f  probe.json.gz
```
