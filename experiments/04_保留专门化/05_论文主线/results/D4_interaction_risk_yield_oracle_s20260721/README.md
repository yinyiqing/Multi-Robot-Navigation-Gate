# D4 Fixed-Priority Yield Oracle

状态：`diagnostic complete; global policy rejected`。该 oracle 使用 manifest 中的特权冲突对标签，只用于验证显式让行的价值，不是可部署方法或 Gate 候选。

## 协议

- 与 frozen 5D probe 使用相同 60 个 scenario ID 和顺序。
- 冲突对中编号较大的机器人为固定让行方。
- 距离 `<= 1.2 m` 时停车，距离 `>= 1.4 m`、通行方结束或等待 20 步（约 4 s）后释放。
- 其余所有动作均由原始冻结 5D 产生，不训练、不调参。

## 总体结果

| Policy | Agent success | Collision | Unresolved | Full success |
| --- | ---: | ---: | ---: | ---: |
| frozen 5D | 0.830 | 0.170 | 0.000 | 0.517 |
| yield oracle on all edge-1 | 0.850 | 0.147 | 0.003 | 0.450 |

全局使用停车让行减少了单机碰撞，但 full success 下降并产生 1 场 timeout。它经常将“多台碰撞”变成“仍有一台失败”，没有解决整场协调。

## 风险分层

| Risk | 5D full | Oracle full | 5D collision | Oracle collision |
| --- | ---: | ---: | ---: | ---: |
| deep | 0.150 | 0.350 | 0.310 | 0.200 |
| close | 0.550 | 0.300 | 0.140 | 0.180 |
| margin | 0.850 | 0.700 | 0.060 | 0.060 |

让行对 deep 有正向价值，但对 close/margin 明显过度干预。这说明 Gate 不应以“视野里有车”或 `edge > 0` 为切换条件，而应识别紧迫的动态冲突。

## 逐场配对

| Risk | Both success | 5D only | Oracle only | Neither |
| --- | ---: | ---: | ---: | ---: |
| deep | 1 | 2 | 6 | 11 |
| close | 5 | 6 | 1 | 8 |
| margin | 13 | 4 | 1 | 2 |
| overall | 19 | 12 | 8 | 21 |

如果仅在 deep 使用本次 oracle 运行结果，close/margin 保留本次 5D 运行结果，组合诊断上限为：

- agent success `0.8667`；
- collision `0.1333`；
- full success `0.5833`。

该数值来自两次独立 Gazebo 运行的后验组合，只表示值得进一步验证的互补性，不是可直接写入论文主表的 Gate 结果。

## 决策

1. 拒绝“所有有交互场景都停车让行”的全局规则。
2. 保留两 Actor + Gate 主线，但两个 Actor 的语义收紧为“普通导航 / 紧迫交互”，不是简单的 standard/dense。
3. Gate 需要局部时序观测，优先表示经自运动补偿的闭合速度/TTC，不读取离线冲突标签。
4. 交互 Actor 的训练应首先集中在 deep，close 作为边界和回归集，margin 主要用来约束过度干预。

结构化结果见 `summary.json` 和 `paired_comparison.json`，原始轨迹与日志已 gzip 保留。
