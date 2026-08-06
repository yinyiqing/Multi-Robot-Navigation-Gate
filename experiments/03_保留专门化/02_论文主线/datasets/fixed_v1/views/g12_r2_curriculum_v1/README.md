# G12-R2逐车数课程视图 v1

状态：`frozen curriculum views / sealed test unread`。冻结日期：`2026-08-06`。

本视图用于参数匹配加宽单Actor从随机初始化完成`n1 -> n2 -> n3 -> n5`普通导航课程。
train保留`fixed_v1/standard/train`全部3000场；validation保留冻结的
`g12_full_scene_selection_v1`全部120场。各车数只截取`r1...rN`，不按冲突拓扑或模型
表现删除场景。

| agents | split | 场景 | SHA-256 |
| ---: | --- | ---: | --- |
| 1 | train | `3000` | `c71e4e87bbc528782cb76dc7df076c493900523bb748b3fb646f3d77fa5f0263` |
| 1 | validation | `120` | `9ab4c5913f683d01e3ab186ea591d373abe1e835180f4a0bfeb469990269b125` |
| 2 | train | `3000` | `5fbd2df5241076041ea714b59286604915ebf1b13848482f7c34fd10cdc9087b` |
| 2 | validation | `120` | `955132263cac9496a56eb8bb6f5132ca5ae41e930c926a7a9a13e8797bb903c9` |
| 3 | train | `3000` | `b6ff22964a8b1795a783f8af9360c123fae44b4b44a86de63e76a57b4a0b4422` |
| 3 | validation | `120` | `f4b7d46fc488eb588007aa7ba72791545e750e691399da82c65d5cdf9f5938cc` |
| 5 | train | `3000` | `82f990dab54331ef55d3818fbe39b31fe00480dd99696987a5b85c5e2581ac1e` |
| 5 | validation | `120` | `e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7` |

派生冲突图只过滤原始五车静态路径图中两端都被保留的边，不重新筛选场景，也不用于R2
采样。原始五车边数保存在`view.source_five_agent_conflict_edge_count`中，防止把派生n1/n2
边数误当成来源场景难度。

完整性审计：

- 每个文件的agent集合严格等于`r1...rN`；
- 每个车数的train/validation ID交集为0；
- 各车数保持完全相同的来源ID和顺序；
- 不读取navigation test、G11-D2、G11-E或sealed test；
- 连续两次重建得到相同哈希。

重建：

```bash
/usr/bin/python3 scripts/build_g12_r2_curriculum_views.py
```

