# Strong Interaction Expert Pilot V1

这是 fixed-v1 standard/dense 的只读派生视图，用于第一轮强交互 Actor pilot。场景仍来自原始冻结 split，没有生成、删除或修改原始场景，也没有读取任何策略成绩。

## 定义

所有场景满足 `conflict_edge_count == 1`，再按同步名义路径最小间距分层：

| Band | Separation | Train | Validation | 用途 |
| --- | ---: | ---: | ---: | --- |
| `deep` | `[0.0, 0.4) m` | 512 | 60 | 主要优化目标 |
| `close` | `[0.4, 0.6) m` | 128 | 40 | 决策边界 |
| `margin` | `[0.6, 0.9) m` | 128 | 40 | 防止过度干预的回归集 |

每档在 standard/dense 两个来源池之间等量抽取，train 与 validation 继承原始 split，因此 scenario ID 无交叉。训练时离线 band 只用于采样和正则权重，不进入 Actor 输入。

## Pilot 准入条件

- deep full success 相对同协议冻结 5D 至少提高 `15 percentage points`。
- deep collision rate 下降。
- close full success 下降不超过 `5 percentage points`。
- margin full success 下降不超过 `3 percentage points`。

未同时满足全部条件时，不扩展到三 seed，也不训练 Gate。

## 复现

```bash
python scripts/build_strong_interaction_views.py \
  --output experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/strong_interaction_v1
```

SHA-256：

```text
cd44c43e94961324c673b67ce0ce93b2c2bf1b13fc4457d185bf3c379c83b7fd  train.json.gz
2c8d124c57327ce18285dd1d91c67888ba819bacdf9ff3240fb12a39da9b0dc4  validation.json.gz
```
