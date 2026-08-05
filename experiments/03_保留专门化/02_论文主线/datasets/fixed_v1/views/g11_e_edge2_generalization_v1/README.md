# G11-E Exact-edge-2自然泛化视图 v1

状态：`frozen validation partitions / sealed test unread`。冻结日期：`2026-08-05`。

来源为旧exact-edge-2确认集：

```text
experiments/03_保留专门化/02_论文主线/08_exact_edge2零样本确认/validation.json
SHA-256 d2fba86651326d439177fce1a7016ed38bee89be3ca9112c0e742994e568a14c
```

来源集合在任何当前A1/B2结果产生前已经按scenario ID哈希冻结。本视图不重新排序，固定
前50场为pilot、后150场为confirmation：

| partition | 场景数 | SHA-256 |
| --- | ---: | --- |
| `pilot.json.gz` | `50` | `f73f260ee4394b11e21a791085cf4957ca50ae93f3466df3882f0d15da932c16` |
| `confirmation.json.gz` | `150` | `72b4b975825bd33900db21a0cd08e19f0f210d9875caa9e8aa86d95b6b268049` |

审计结果：

- 两个partition的`conflict_edge_count`全部为2，ID交集为0，合计覆盖来源200场；
- pilot的`max_degree=1/2`为`23/27`，`simultaneous=1/2`为`34/16`；
- confirmation的`max_degree=1/2`为`79/71`，`simultaneous=1/2`为`85/65`；
- 与G11-A1 train/internal-validation、G11-C和G11-D2场景重叠均为0；
- 没有读取导航test或sealed test。

构建命令：

```bash
/usr/bin/python3 scripts/build_g11_e_edge2_views.py
```

构建器使用`mtime=0`的确定性gzip；重复构建必须得到相同哈希。运行和准入边界见
[`G11-E协议`](../../../../11_可部署在线Gate研究/G11_E_multi_edge泛化pilot/README.md)。
