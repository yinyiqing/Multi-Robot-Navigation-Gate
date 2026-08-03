# 全路径纯单冲突数据视图 v1

## 用途

该视图用于判断：完整 Actor 在 Dense 多冲突训练中学不出来，是否主要由场景中的冲突组合复杂度造成。

源数据为 `views/edge1_pilot`，并使用全路径冲突审计重新过滤。一个场景只有同时满足以下条件才保留：

- `full_path_conflict_edges == 1`
- `full_path_max_degree == 1`
- `full_path_simultaneous == 1`

这比原始 8 秒标签更严格。

## 固定划分

| 文件 | 数量 | 用途 | SHA256 |
|---|---:|---|---|
| `train.json.gz` | 511 | 训练 | `658750efd321e337d5756eacd040f83aea08385d1e849b1df064f92b14dcb492` |
| `validation.json.gz` | 421 | 后续完整验证 | `c43d2680e827b81ea9a96e0df124dd23adae413e2e52be5ad29c131e8c83d36d` |
| `validation_monitor_50.json.gz` | 50 | 训练期固定监控，25 standard + 25 dense | `f8020f1275aa68f57530798bcd81c3b3c180958f94bdebb9bd6aa6796c6ffe58` |

全路径复审结果：三份清单均为 `missing_edges=0`、`non_pure=0`。

剔除 3 个场景：

- train: `standard-20260717001630-484872781a2c`
- validation: `standard-20260717006526-5598e9dc128e`
- validation: `standard-20260717006348-de73fb691522`

## 复现

```bash
python3 scripts/build_full_horizon_edge1_view.py
```

构建脚本使用确定性选择和 `gzip mtime=0`。过滤依据为：

`results/04_Gate前置验证/20260803_全场景数据质量诊断/full_horizon_edge1_audit.json.gz`

审计文件 SHA256：`589017c8084c61aca97d5a58e2c9e7f51ed9efbdfb3454371bd056adfce16b49`。
