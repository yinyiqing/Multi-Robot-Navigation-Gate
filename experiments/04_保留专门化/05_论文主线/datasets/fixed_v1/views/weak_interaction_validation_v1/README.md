# Weak-Interaction Validation V1

该 view 从 fixed-v1 的 standard/dense validation split 确定性派生，只保留策略无关指标 `conflict_edge_count = 0` 的场景。它用于公平比较5A与5D的弱交互能力，不读取模型结果，也不修改原始场景。

| Source pool | Episodes |
| --- | ---: |
| standard validation | 206 |
| dense validation | 42 |
| total | 248 |

场景顺序保持为 standard 原始顺序后接 dense 原始顺序，scenario ID 与源 manifest 完全一致。运行时 Actor 不读取 `view` 标签。

复现：

```bash
python3 scripts/build_weak_interaction_validation.py
```

SHA-256：

```text
eb5061d6b61c1c3d57174f09308bf8c3f35c4b9d1cf8cc5971f003dd69ff3bb2  standard/validation.json.gz
2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7  dense/validation.json.gz
142e5a2316bdc572038fd5316d007869cd365b8f0fa5637e08cf22133f2e521e  validation.json.gz
```
