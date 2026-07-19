# Edge-1 Interaction Pilot View

这是 fixed-v1 原始 manifest 的只读派生视图，不生成、删除或修改场景。筛选条件仅为策略无关的 `conflict_edge_count == 1`。

| Split | Standard | Dense | Total | Sampling |
| --- | ---: | ---: | ---: | --- |
| train | 256 | 256 | 512 | 固定 seed 打乱后 cycle |
| validation | 212 | 211 | 423 | 全量固定顺序 cycle |

训练子集从 standard/dense 各自的全部 edge-1 train 场景中用 seed `20260720` 确定性抽取。validation 保留两个 pool 的全部 edge-1 场景，不做抽样。运行时 Actor 和 gate 不读取该离线标签。

重新生成：

```bash
source env.python.sh
scripts/build_interaction_views.py \
  --output experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/edge1_pilot
```

SHA-256：

```text
0206d1d3090024ac4b453ae88eff140c5476854f52ddf4eab997930241e39f23  train.json.gz
ad15ad19c83314fa1833a0abbe80184ed642f142e4210db73535405089bce93b  validation.json.gz
```
