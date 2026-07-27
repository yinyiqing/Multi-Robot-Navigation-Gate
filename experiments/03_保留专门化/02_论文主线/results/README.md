# 结果目录索引

本目录按“这些结果在当前论证中起什么作用”分类，而不是按运行日期平铺。历史实验的叶子目录名保持不变，便于追溯日志、checkpoint 和提交记录。

## 当前应先看什么

当前冻结方案只看 [`05_当前冻结方案/`](05_当前冻结方案/README.md)：

```text
generalist-5a（普通导航，冻结）
  + strong-interaction-5a-balanced epoch 16（条件交互，冻结）
  -> 下一步只训练机器人感知和 Gate
```

其余目录分别提供基线、失败证据、机制排查和感知前置实验，不能从中重新挑选 Actor 覆盖当前冻结结论。

## 目录职责

| 目录 | 状态 | 作用 |
| --- | --- | --- |
| [`01_基线评估/`](01_基线评估/README.md) | baseline / complete | 5A、5D 和 fixed-v1 的冻结基线 |
| [`02_普通Actor_失败对照/`](02_普通Actor_失败对照/README.md) | failed diagnostic | 直接训练 standard expert 的退化证据 |
| [`03_强交互Actor_研发记录/`](03_强交互Actor_研发记录/README.md) | historical diagnostic | 强交互 Actor 从失败机制到正式配置的研发过程 |
| [`04_Gate前置验证/`](04_Gate前置验证/README.md) | diagnostic / prerequisite | 场景可解性、风险信号、机器人感知和时序表示验证 |
| [`05_当前冻结方案/`](05_当前冻结方案/README.md) | current / frozen | 当前两个 Actor 的正式训练和重复验证证据 |
| [`90_中止与无效运行/`](90_中止与无效运行/README.md) | invalid / aborted | 未形成有效模型比较的运行，仅保留溯源 |

## 归档规则

- 新结果先判断用途，再进入对应父目录；禁止重新平铺到 `results/` 根目录。
- 每个叶子目录只对应一个协议明确的实验或配对评估。
- 已归档叶子目录不改名、不覆盖；修正协议必须建立新实验 ID。
- `90_中止与无效运行/` 中的数值不得进入模型选择或论文主表。
- D3/D4 前缀是历史阶段 ID，不表示当前研究阶段；当前阶段是 D5-G0。
