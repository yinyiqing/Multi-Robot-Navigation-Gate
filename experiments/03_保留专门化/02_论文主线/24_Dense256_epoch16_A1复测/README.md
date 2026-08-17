# Dense256 epoch16+A1复测

状态：`registered / running`。登记日期：`2026-08-17`。

G17完整场景复测中，epoch16+A1成为暂定learned-Gate候选。本实验只补Dense256当前
套件缺失的epoch16+A1，不重跑已有5A、epoch17+F-A1、epoch16+B2及两种2m特权规则。

固定协议：Dense validation manifest前256个场景、seed `20260810`、五车、固定物理步进；
A1阈值`0.28/0.18`、hold 3、stride 2。所有Actor、detector、Gate和阈值冻结，不读取
sealed test。

若epoch16+A1在Dense256上稳定超过epoch17+F-A1且保持合理timeout/效率，则主方法切回
`5A + epoch16 + A1`；否则保留epoch17+F-A1，G17差异只记为无显著差异的诊断。
