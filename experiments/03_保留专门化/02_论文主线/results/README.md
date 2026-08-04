# 结果目录索引

本目录按证据作用归档。当前路线只以[父目录论文协议](../README.md)为准，历史目录名中的
“当前”不表示它仍是当前方案。

## 当前证据链

```text
5A普通导航基线
  + epoch-16局部调用有效性
  + 2.0 m真值oracle组合上界
  + 历史Gate可部署基线与失败边界
  -> 新的可部署在线Gate
```

Residual Actor和独立完整Actor不在当前链路中。

## 目录职责

| 目录 | 状态 | 作用 |
| --- | --- | --- |
| [`01_基线评估/`](01_基线评估/README.md) | frozen baseline | 5A、5D和fixed-v1基线 |
| [`02_普通Actor_失败对照/`](02_普通Actor_失败对照/README.md) | rejected | standard expert微调退化证据 |
| [`03_强交互Actor_研发记录/`](03_强交互Actor_研发记录/README.md) | historical / rejected family | epoch-16形成过程和独立Dense Actor失败记录 |
| [`04_Gate前置验证/`](04_Gate前置验证/README.md) | diagnostic | 可解性、风险信号、感知和规则边界 |
| [`05_当前冻结方案/`](05_当前冻结方案/README.md) | frozen evidence | 5A、epoch-16及oracle组合核心证据；目录名为历史命名 |
| [`06_Gate开发/`](06_Gate开发/README.md) | frozen baseline | G0/G1/G2-A与第一版Gate |
| [`90_中止与无效运行/`](90_中止与无效运行/README.md) | invalid | 不得用于模型选择或论文结果 |

corrected edge-1完整Actor结果位于
[`../10_纯单冲突完整Actor_pilot`](../10_纯单冲突完整Actor_pilot/README.md)，状态为
rejected pilot。Residual相关结果位于[`../09_单冲突Residual组合主线`](../09_单冲突Residual组合主线/README.md)，
状态为rejected。当前Gate的G11-A0/A1离线结果位于
[`../11_可部署在线Gate研究`](../11_可部署在线Gate研究/README.md)；G11-A1已通过离线
准入，但在闭环结果产生前仍不得写入导航主表。

## 归档规则

- 已完成实验保留事实、日志来源和必要checkpoint说明；
- 历史README中的旧“下一步”不构成执行授权；
- partial、validation、test和oracle必须明确标注；
- `90_中止与无效运行`数值不得进入论文主表；
- 新Gate结果建立新实验ID，不覆盖G2-A历史结果。
