# G11-C 固定闭环Pilot视图

该视图从冻结的`g11_a1_gate_v1/validation.json.gz`按原顺序选取50场，不读取navigation
test或sealed test。

| stratum | 场景数 |
| --- | ---: |
| standard zero | 13 |
| standard edge-1 | 12 |
| dense zero | 12 |
| dense edge-1 | 13 |

总计standard/dense各25场、zero/edge-1各25场。生成命令：

```bash
/usr/bin/python3 scripts/build_g11_c_pilot_view.py
```

`validation.json.gz` SHA-256：

```text
1bf044cb5ff9d7d80c14d860d1108481af1d422cf403b26869f8b963012f0e91
```
