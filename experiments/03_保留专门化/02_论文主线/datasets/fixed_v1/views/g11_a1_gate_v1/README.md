# G11-A1 Gate 数据视图 v1

本视图只从导航`fixed_v1` train构建，不读取导航validation/test或感知sealed test。
它用于冻结Actor条件下的G11-A1离线时序Gate pilot。

| 文件 | 场景 | 组成 | SHA-256 |
| --- | ---: | --- | --- |
| `train.json.gz` | 640 | 四层各160 | `a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026` |
| `validation.json.gz` | 120 | 四层各30 | `e261a7afbac8f7341ab13609c2662a2824a0ff383789287ad7733290389cd99d` |

四层为`standard/dense × full-path 0-edge/corrected full-path edge-1`。构建时排除旧
G0/G2 pilot train和validation的全部200个scenario ID。完整路径复算额外发现并剔除
6个原标签为0-edge、实际为edge-1的standard场景。

独立全量复算结果：train为`0->0: 320, 1->1: 320`，validation为
`0->0: 60, 1->1: 60`，`scenarios_with_missing_edges=0`。两个split无scenario重叠。

确定性重建：

```bash
/usr/bin/python3 scripts/build_g11_a1_gate_views.py
```

seed为`20260804`，gzip使用`mtime=0`。

兼容字段`view.perception_pool`和`view.interaction_band`与`gate_pool/gate_topology`
同步保存，保证现有recorder不会把分层写成`unknown`。
