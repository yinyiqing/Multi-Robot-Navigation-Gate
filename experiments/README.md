# 实验索引

先读[当前项目状态](../PROJECT_STATUS.md)。跨阶段的当前结论只以
[论文主线](03_保留专门化/02_论文主线/README.md)为准；历史README中的“当前”和
“下一步”不构成执行授权。

## 当前最短路径

```text
PROJECT_STATUS.md
  -> 03_保留专门化/02_论文主线/README.md
  -> EXPERIMENT_REGISTRY.md
  -> 论文主线/results/README.md
  -> 论文主线/datasets/README.md
```

## 阶段

| 目录 | 状态 | 作用 |
| --- | --- | --- |
| [`01_第一次尝试/`](01_第一次尝试/) | historical | 基础TD3、多车共享策略和reward机制验证 |
| [`02_课程学习/`](02_课程学习/) | historical / frozen source | 形成5A、5D；不从旧stage继续训练 |
| [`03_保留专门化/01_历史诊断/`](03_保留专门化/01_历史诊断/README.md) | historical / rejected | 覆盖训练、旧专家和旧Residual诊断 |
| [`03_保留专门化/02_论文主线/`](03_保留专门化/02_论文主线/README.md) | current | 冻结双Actor与可部署Gate |
| [`03_保留专门化/90_未启用方案/`](03_保留专门化/90_未启用方案/README.md) | inactive | 未启用兜底，不与主线并行 |

## 当前组件

| ID | 状态 | 角色 |
| --- | --- | --- |
| `generalist-5a` | frozen current | 普通导航Actor |
| `interaction-epoch16` | frozen current | 条件避障Actor |
| `2.0 m interaction oracle` | diagnostic upper bound | 真值组合上界，不可部署 |
| `learned-gate-g2a` | frozen failed-admission baseline | 历史学习Gate基线 |
| `deployable-interaction-gate` | current pending | 当前唯一开发组件 |

Residual Actor、完整Dense Actor、epoch-16整网续训和corrected edge-1完整Actor均已关闭。
详细原因见[实验注册表](EXPERIMENT_REGISTRY.md)。

## 归档规则

- 已完成实验保留事实、日志和必要checkpoint说明，不因失败删除证据；
- 历史artifact不批量改名，避免破坏复现路径；
- 当前结论必须回写当前协议或实验注册表，不能只留在日志；
- train、validation和sealed test保持scenario ID互斥；
- 失败和invalid运行不得因文件名含`best`重新成为候选；
- `trash/`默认禁止读取、搜索和修改。
