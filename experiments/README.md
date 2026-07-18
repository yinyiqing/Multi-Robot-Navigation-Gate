# 实验索引

本目录按研究演进归档。历史目录中的“当前主线”和“下一步”只代表当时判断；跨阶段的当前结论只以 `04_保留专门化/05_论文主线/README.md` 为准。

## 状态词

| 状态 | 含义 |
| --- | --- |
| `current` | 当前论文协议或正在补齐的正式实验 |
| `diagnostic` | 用来定位问题，不进入论文主表 |
| `baseline` | 可以按新口径重跑的正式对照 |
| `failed` | 结论有价值，但该方法分支停止 |
| `historical` | 仅用于追溯研究演进 |
| `planned` | 尚未实现或尚未达到准入条件 |

## 阶段注册表

| 目录 | 状态 | 作用 |
| --- | --- | --- |
| `01_第一次尝试/` | `historical` | 单车、多车共享策略、reward 和局部 Critic 的早期机制验证 |
| `02_课程学习/` | `historical` / `baseline` | 形成 5A、5D，并记录 PAIR/THREE 覆盖训练退化 |
| `04_保留专门化/01_冲突验证/` | `diagnostic` | 普通能力、dense 能力和旧模型冲突证据 |
| `04_保留专门化/02_双Actor切换/` | `failed` | 5A + 5D hard switch/oracle 缺少互补性 |
| `04_保留专门化/03_dense专家训练/` | `diagnostic` / `failed` | full fine-tune、head-only、random/fixed dense 诊断 |
| `04_保留专门化/04_安全兜底/` | `planned` | 暂停；不属于当前 D1-D3 工作 |
| `04_保留专门化/05_论文主线/` | `current` | 唯一论文协议、数据定义和决策门 |

编号 `03` 没有在仓库顶层复用：它代表已撤下的中间研究方向。保留编号空缺可以避免旧日志、提交和文档引用发生歧义。

## 当前实验 ID

| 实验 ID | 模型 | 场景 | 状态 | 目的 |
| --- | --- | --- | --- | --- |
| `eval-5d-standard` | `generalist-5d` | `standard-5` | `complete` | 1000 episodes: agent `0.8816`, full `0.5690` |
| `eval-5d-fixed-v1` | `generalist-5d` | fixed standard/dense | `complete` | standard full `0.5750`; dense full `0.2795` |
| `diag-5d-random-dense` | `generalist-5d` | `random-dense-5` | `diagnostic` | 仅区分 spatial density 与 interaction density |
| `eval-5d-canonical-moderate` | `generalist-5d` | 五个 fixed moderate cases | `baseline` | held-out interaction failure baseline |
| `train-residual-specialist` | `generalist-5d + residual` | fixed dense/train | `next` | D3 已完成，可以进入 D4 |
| `train-temporal-gate` | generalist + specialist | fixed standard/dense mix | `planned` | 专家互补性达到 D5 后才允许训练 |

## 阅读顺序

1. [论文主线](04_保留专门化/05_论文主线/README.md)
2. [现有场景对照](04_保留专门化/05_论文主线/SCENARIO_COMPARISON.md)
3. [保留专门化证据](04_保留专门化/README.md)
4. [dense 定义诊断](04_保留专门化/03_dense专家训练/logs/test/dense_definition_20260716/README.md)
5. [课程学习简明总结](02_课程学习/课程学习简明总结.md)
6. [完整历史总览](实验总览.md)

## 归档规则

- 正式 run 必须记录短实验 ID、模型 ID、scenario ID、seed、commit 和完整配置。
- 当前运行日志先写入根目录 `logs/`；结束并形成结论后再归档到对应实验目录。
- `TD3/checkpoints/`、`TD3/results/`、`TD3/runs/` 是可恢复运行产物，不是发布模型。
- 模型身份以 [模型注册表](../TD3/MODEL_REGISTRY.md) 为准，不从长文件名猜测角色。
- 历史失败保留结论和必要证据，不复制同一日志到多个位置。
