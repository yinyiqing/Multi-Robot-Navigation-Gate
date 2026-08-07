# G12-R2五车底座20k配对准入结果

状态：`completed / narrowly failed timeout / 10k fallback registered`。日期：`2026-08-07`。

两个策略均完成固定`120/120`场，manifest顺序、重复和缺失审计通过。

| policy | agent success | full success | collision | unresolved | timeout | avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5A | `0.8100` | `0.5583` | `0.1900` | `0` | `0` | `20.08` |
| R2-S4 20k | `0.8983` | `0.6917` | `0.0967` | `0.0050` | `0.0250` | `29.68` |

R2相对5A为`18`场改善、`2`场退化、`100`场持平，McNemar exact
`p=0.000402`。分层full success：

| topology | 5A | R2 20k | improved/degraded | p |
| --- | ---: | ---: | ---: | ---: |
| 0-edge | `0.900` | `0.975` | `4/1` | `0.375` |
| edge-1 | `0.625` | `0.675` | `3/1` | `0.625` |
| multi-edge | `0.150` | `0.425` | `11/0` | `0.000977` |

dense full success从`0.3833`提高到`0.5833`，standard从`0.7333`提高到`0.8000`。大Actor
明显降低碰撞并改善复杂场景，不存在“成绩由简单场景撑起”的问题。

五项准入中四项通过。唯一失败项是overall timeout：R2为`3/120=0.025`，相对5A增加
`0.025`，超过冻结上限`0.020`，即离散计数上多1场。因此必须严格判定20k未通过，不能
事后放宽阈值。

S4训练协议预先保存了10k checkpoint，且训练内评测的timeout为`1/120`。在不修改场景、
seed、阈值或5A结果的前提下，登记唯一一次10k fallback。若10k仍未通过，R2停止，不再
扫描其他checkpoint；若通过，则用10k作为R3 warm start和R2冻结参考。

