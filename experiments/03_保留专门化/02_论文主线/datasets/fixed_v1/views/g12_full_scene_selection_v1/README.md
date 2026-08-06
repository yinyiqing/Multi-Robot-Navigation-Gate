# G12完整场景内部validation视图 v1

状态：`frozen internal validation / sealed test unread`。冻结日期：`2026-08-06`。

本视图是G12参数匹配单Actor的内部checkpoint选择集。它只从`fixed_v1`原始validation
构建，不读取导航test或sealed test，也不按任何Actor或Gate的运行结果筛选。

| 文件 | 场景 | SHA-256 |
| --- | ---: | --- |
| `validation.json.gz` | `120` | `52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635` |

固定分层使用原始manifest中的策略无关静态路径指标`metrics.conflict_edge_count`：

| stratum | eligible | selected |
| --- | ---: | ---: |
| standard-zero | `141` | `35` |
| standard-edge1 | `162` | `20` |
| standard-multi | `82` | `5` |
| dense-zero | `7` | `5` |
| dense-edge1 | `161` | `20` |
| dense-multi | `547` | `35` |

因此standard/dense各60场，zero/edge1/multi各40场。`multi`表示
`conflict_edge_count >= 2`。该指标由固定静态规划路径生成，默认冲突horizon为`8 s`；
这里没有把它误写成corrected full-horizon edge分类。

显式排除：

- `fixed_v1/standard/train.json.gz`和`fixed_v1/dense/train.json.gz`中的全部9000个ID；
- G11-C的50场、G11-D2的200场；
- G11-E pilot/confirmation的50/150场。

C/D2/E彼此共有450个不同ID，但G11-C来自navigation train，所以全部排除集合共有
`9400`个不同ID。最终120场与上述六个manifest逐一交集均为0，场景ID内部也无重复。

层内选择固定seed `20260806`，按`scenario_id`的SHA-256排序；输出gzip使用`mtime=0`。
连续两次重建得到相同SHA-256。重建命令：

```bash
/usr/bin/python3 scripts/build_g12_full_scene_selection_view.py
```

