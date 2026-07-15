# 03 dense 专家训练

这里放当前主线：训练第二个 dense 专家 actor。

## 当前实验

入口脚本：

```bash
scripts/start_training_detached_dense5_bridge_head_from_5d.sh
```

训练设置：

- warmstart：`5D` actor
- 训练环境：`stage4_asym_dense_5_bridge`
- actor 模式：`head_only`
- critic：新初始化
- 目标：超过固定 `5D` 在 bridge 上的表现

## Baseline

| 模型 | 场景 | success | collision | full |
| --- | --- | ---: | ---: | ---: |
| `5D` | `stage4_asym_dense_5_bridge` | 0.540 | 0.475 | 0.250 |
| `5A` | `stage4_asym_dense_5_bridge` | 0.530 | 0.475 | 0.275 |

已清理的 hard stage4 不再作为训练或测试入口。它目标点过近，固定 `5D` 也只有约 `0.355 / 0.680 / 0.025`，更像病态压力测试。

## 当前判断

- `5D -> bridge` full fine-tune 会退化。
- `5A -> bridge` full fine-tune 基本维持原状，随后也没有突破。
- 这说明问题不只是 `5D` 起点太拟合，而是 full actor 解冻本身容易把已有导航能力拉坏。
- 当前优先验证 head-only；如果不够，再加 adapter / residual，而不是回到 full actor fine-tune。

## 规则

- 运行中的日志临时放仓库根目录 `logs/`。
- 跑完后，有效日志归档到本目录；无效日志直接删。
- 不再把 attention / gate 实验混进这里。
