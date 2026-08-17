# Dense256 epoch16+A1复测

状态：`complete / epoch16+B2 selected for Dense main task`。登记日期：`2026-08-17`。

G17完整场景复测中，epoch16+A1成为暂定learned-Gate候选。本实验只补Dense256当前
套件缺失的epoch16+A1，不重跑已有5A、epoch17+F-A1、epoch16+B2及两种2m特权规则。

固定协议：Dense validation manifest前256个场景、seed `20260810`、五车、固定物理步进；
A1阈值`0.28/0.18`、hold 3、stride 2。所有Actor、detector、Gate和阈值冻结，不读取
sealed test。

预注册判断原本只比较epoch16+A1与epoch17+F-A1；补测完成后，结合相同Dense256中已冻结
的epoch16+B2结果进行最终选择。最终选择不得跨manifest拼接，也不得只按单次峰值决定。

## 2026-08-17 结果

epoch16+A1的256场一次完成并通过审计。统一结果为：

| 方法 | Full success | Collision | Timeout | 平均步数 |
| --- | ---: | ---: | ---: | ---: |
| 5A | 0.2695 | 0.3000 | 0.0039 | 17.46 |
| epoch17 + F-A1 | 0.4023 | 0.2227 | 0.0078 | 32.94 |
| epoch16 + A1 | 0.3867 | 0.2250 | 0.0039 | 30.29 |
| epoch16 + B2 | **0.4258** | **0.2102** | 0.0078 | 34.40 |
| epoch16 + 2m privileged rule | 0.4766 | 0.1734 | 0.0156 | 34.65 |
| epoch17 + 2m privileged rule | 0.5078 | 0.1594 | 0.0156 | 40.12 |

epoch16+A1相对5A显著改善（48/18，`p=0.000287`），但没有超过epoch17+F-A1
（38/42，`p=0.7376`），因此拒绝“因G17点估计更高而直接切换到A1”的假设。
epoch16+B2是Dense256冻结Gate套件中最高的可部署组合，相对5A为56场改善、16场退化，
`p=2.40e-6`；相对R2B-best为59/18，`p=3.06e-6`；相对epoch17+F-A1为43/37，
`p=0.5764`，不能声称两种Gate之间显著。
它恢复epoch16特权规则收益的`75.5%`，但在G17上的效率和timeout代价高于F-A1。

综合高冲突主任务与完整场景保持，最终冻结epoch16+B2；论文只能主张其显著
超过5A并接近特权诊断规则，不能声称B2显著优于F-A1或student rollout本身带来显著收益。
