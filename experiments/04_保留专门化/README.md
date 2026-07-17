# 保留与专门化

状态：`current research branch`。具体方法、数据和实验顺序只以 [论文主线协议](05_论文主线/README.md) 为准。

## 为什么进入这条路线

历史实验反复出现同一现象：已有 Actor 在新 dense 课程中继续更新时，普通导航能力会被覆盖，而 high-interaction 能力也未稳定提高。

```text
5A / 5D 普通导航能力
  -> PAIR / THREE 渐进 dense：未形成稳定增益
  -> full Actor fine-tune：持续退化
  -> head-only：不崩，但表达力不足
  -> 5A + 5D switch/oracle：专家互补性不足
```

因此当前假设是：冻结 `generalist-5d`，只学习幅度受限的 residual specialist；只有 paired evaluation 证明专家互补后，才训练 temporal gate。

## 子目录角色

| 目录 | 状态 | 内容 |
| --- | --- | --- |
| `01_冲突验证/` | historical diagnostic | 5A、5D、PAIR 的能力覆盖证据；结果使用旧口径 |
| `02_双Actor切换/` | failed diagnostic | 5A + 5D hard switch 和 case oracle 未超过 5D |
| `03_dense专家训练/` | diagnostic / failed | full fine-tune、head-only、random/fixed dense 和 residual 脚手架 |
| `04_安全兜底/` | planned | 当前不推进 |
| `05_论文主线/` | current | dense 定义、数据划分、实验矩阵和决策门 |

## 已确认的证据

- 5D 是当前 generalist 候选，但旧 standard 结果必须按互斥指标重跑。
- full Actor fine-tune 在 moderate fixed cases 上逐轮退化。
- head-only 限制了破坏，但没有超过冻结 5D。
- random dense 同时缩短了任务距离，不能证明策略擅长高交互。
- 五个 fixed moderate cases 能暴露同步冲突，但不是正式训练分布，只保留为 canonical held-out。
- 历史 5A + 5D 没有足够的 `specialist-only success`，不能直接支撑 gate。

## 当前允许的工作

```text
D1  实现 conflict graph、standard/dense 生成器和 manifest 回放（已完成）
D2  完成 Gazebo 有效性筛选并冻结两个场景池的 train/validation/test（已完成）
D3  在固定 test manifest 上重跑 generalist-5d baseline
```

`D1-D3` 完成前不启动 residual 或 gate 训练。相关脚本虽然已经有结构脚手架，但不属于当前可执行入口。

## 名称

新文档统一使用短 ID：

- 模型：`generalist-5d`, `residual-specialist`, `temporal-gate`
- 场景池：`standard`, `dense`
- 当前评估：`eval-5d-standard`

历史 artifact 原名不修改，映射见 [模型注册表](../../TD3/MODEL_REGISTRY.md)。
