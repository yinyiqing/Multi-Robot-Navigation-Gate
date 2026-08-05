# G11-D2独立闭环准入视图 v1

本视图只从导航`fixed_v1` validation构建，不读取导航test或sealed test。它在任何D2
策略运行前冻结，用于0-edge和corrected full-horizon edge-1的独立闭环准入。

| 文件 | 场景 | SHA-256 |
| --- | ---: | --- |
| `validation.json.gz` | `200` | `6250b941f127d550641a621d4253e17ea0770ff3c0cb94e6254e1f26b9f4978a` |

固定分层：

| stratum | eligible | selected |
| --- | ---: | ---: |
| standard-zero | `206` | `65` |
| dense-zero | `35` | `35` |
| standard-edge1 | `210` | `50` |
| dense-edge1 | `166` | `50` |

zero和edge-1各100场。dense-zero只有35场，是因为本视图显式排除了旧G3使用的dense
monitor 200场；没有从旧开发场景或test补齐。总指标之外必须逐层报告。

构建时还显式排除G11-A1内部validation和G11-C pilot。三类排除manifest合计320个唯一
scenario ID，最终交集均为0。所有zero候选使用与G11-A1相同参数重新计算完整静态路径，
结果为standard `206/206`、dense `35/35`保持full-path zero；edge-1直接来自冻结的
`edge1_full_horizon_v1/validation`。

完整性审计确认200个scenario ID唯一，`navigation_split=validation`，冲突边范围为
`0-1`。确定性选择使用seed `20260805`按`scenario_id`的SHA-256层内排序，gzip
`mtime=0`。重建：

```bash
/usr/bin/python3 scripts/build_g11_d2_admission_view.py
```
