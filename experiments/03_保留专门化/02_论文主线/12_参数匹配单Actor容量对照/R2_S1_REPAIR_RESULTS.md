# G12-R2-S1定向补课结果

状态：`completed / broad admission failed / candidate rejected`。日期：`2026-08-07`。

## 1. 运行身份

```text
experiment: G12-R2-S1-repair-pilot
model: capacity_wide_r2_s1_repair_n1_seed20260813
warm start: S0 epoch 2 best Actor and Critic
training: 20,169 agent samples, 140 episodes
train cases: 8 geometrically unique repair cases
validation: frozen broad n1, 120/120 scenarios in manifest order
```

训练后的latest checkpoint完整包含Actor、Critic、优化器和20,169条fresh replay。训练结束
后的进程内评测最初因`curriculum -> manifest`临时切换缺失而没有启动；该问题发生在
checkpoint保存之后。修复提交为`41c1968`，随后从同一checkpoint原样导出Actor并独立完成
冻结120场评测，没有追加任何训练更新。

## 2. Broad准入

| 指标 | 结果 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| full success | `69/120 = 0.5750` | `>=117/120` | fail |
| collision | `2/120 = 0.0167` | `<=3/120` | pass |
| unresolved | `49/120 = 0.4083` | - | - |
| timeout | `49/120 = 0.4083` | `<=3/120` | fail |
| average steps | `143.32` | - | - |

逐场审计确认120个scenario ID与冻结manifest完全同序、无缺失、无重复，且
`success + collision + unresolved = 120`。按来源拆分，standard为`30/60=0.500`，dense为
`39/60=0.650`；前60场和后60场分别为`0.600/0.550`。退化贯穿整个集合，不能解释为个别
场景或末段仿真波动。

## 3. 漂移诊断

在本次fresh replay的20,169个训练状态上比较S0与S1确定性动作：

- 线速度动作维平均绝对偏移：`0.2992`；
- 角速度动作维平均绝对偏移：`0.5220`；
- 两维95%绝对偏移分位数：`1.4329 / 1.7044`；
- Actor参数相对L2漂移：`23.29%`。

训练前5k只采集replay，之后在8个重复repair case上进行约15k次全网络、无anchor TD3更新。
训练末段连续出现300步未完成，正式broad结果随后确认严重基础导航遗忘。该候选没有P1式
单一固定动作坍塌，但其大范围动作函数漂移足以破坏S0的普通导航能力。

## 4. Artifact

| artifact | SHA-256 |
| --- | --- |
| latest training checkpoint | `6d6ca40a3973256ccfc1d2f51a5e829888cc0091d01fff66ddeda2c274bf1f81` |
| exported Actor | `e475be67d92d48277911731b4160afdbe21d25ceeba2f50160dbfe9734d18fe5` |
| exported Critic | `32c6aac27e93a6e0ff1e6834d20da67f97132ea22f3793599764ef46f97bb37c` |
| validation results | `50147e20edf4157c7ff9be90bfaa033d984adf2f069ada66d5269d6db1472646` |
| validation state | `c32992ad0076a6559912ac74b2ee9967883d2c64cbb40ef6faf90cf5a43d05f2` |
| validation summary | `58356b32ad071b4d9cba3a938593100a67feea8a63070f07e40fec73c3e610d1` |

结构化validation artifact位于`local_data/s1_repair_validation/`，运行日志归档到
`logs/archive/training/g12_r2/s1_repair/`。

## 5. 决定

该候选回滚，不进入126场targeted复测，不追加第二段20k，也不得作为S2 warm start。
当前唯一保留的加宽Actor底座仍是S0 epoch 2 best。

若继续修复S1，必须先登记新协议：repair case只能作为原n1 navigation-train分布中的
重采样部分，同时加入S0行为保持约束，并把首个broad回归检查提前到远小于20k的预算点。
不得使用broad validation本身训练，也不得从本次失败checkpoint续训。
