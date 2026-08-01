# G3 Learned Gate 50场参数冻结

状态：`通过；冻结switch-on/off=0.44/0.34、minimum_hold_steps=3，进入200场准入`。

## 协议

- 场景：`dense_validation_monitor_v1`前50个固定scenario；
- seed：`20260802`；
- 普通Actor：冻结5A；
- 局部Actor：冻结epoch-16；
- detector：冻结G0 pilot-v1 epoch-19；
- Gate：冻结G2-A `oracle_front_v1` epoch-8；
- Gate checkpoint阈值：`0.44`；
- switch-on/off：`0.44/0.34`；
- 最短保持：3个环境步；
- 固定物理步：`0.001 s`；
- test未读取。

先完成1场smoke，确认原始点云、G0/G1、Gate、双Actor、结果记录和进程清理均
正常。随后不根据smoke结果修改参数，直接运行固定50场。

## 结果

learned Gate本轮结果：

| metric | learned Gate |
| --- | ---: |
| agent success | `214/250 = 0.856` |
| collision | `36/250 = 0.144` |
| full success | `27/50 = 0.540` |
| timeout | `0/50 = 0.000` |
| 平均环境步 | `26.3` |
| 强Actor平均激活比例 | `0.545` |
| 每场平均切换次数 | `11.62` |
| 平均Gate概率 | `0.463` |

按scenario ID与已有1000场5A和真值Oracle结果取相同50场比较：

| 方法 | agent success | collision | full success | timeout | 平均环境步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 历史5A | `0.720` | `0.280` | `0.300` | `0.000` | `23.9` |
| learned Gate | `0.856` | `0.144` | `0.540` | `0.000` | `26.3` |
| 历史`2.0 m`真值Oracle | `0.876` | `0.124` | `0.540` | `0.000` | `47.5` |

learned Gate相对5A为`14`场改善、`2`场退化、`34`场相同，McNemar exact
`p=0.00418`。在这50场上，Gate恢复的Oracle full-success增益比例为：

```text
(0.540 - 0.300) / (0.540 - 0.300) = 1.0
```

Gate与Oracle各有`7`场独有成功，说明二者总full success相同，但并非逐场等价。

## 冲突结构分层

| 层 | 场景数 | 5A | learned Gate | Oracle |
| --- | ---: | ---: | ---: | ---: |
| `edge=1` | 15 | `0.400` | `0.600` | `0.600` |
| `edge=2` | 16 | `0.188` | `0.500` | `0.438` |
| `edge=3` | 12 | `0.417` | `0.667` | `0.667` |
| `edge=4` | 4 | `0.000` | `0.250` | `0.500` |
| `max_degree=2` | 25 | `0.320` | `0.560` | `0.600` |
| `simultaneous=2` | 20 | `0.300` | `0.550` | `0.650` |

样本量较小的分层只作趋势判断。它至少确认收益不只来自edge-1，未见的edge-2/3
和最大度2场景均有正收益。

## 诊断与限制

- full-success失败场的平均强Actor激活比例为`0.581`，成功场为`0.515`；高激活不
  自动等于成功，200场必须继续检查过度调用。
- 平均每场`11.62`次模式切换偏多，但没有产生timeout，且平均步数只比5A增加
  `2.4`步。因此当前不以主观“切换次数太多”为理由修改滞回。
- 5A和Oracle来自相同scenario ID但较早的`20260728`运行，本轮Gate seed为
  `20260802`。Gazebo存在轨迹波动，因此这组配对用于开发准入，不作为最终论文
  统计。200场需要按固定协议重复，并在最终validation中补同进程或重复统计。

## 决策

当前参数超过预设50场准入要求：full success达到`0.540`、恢复超过60%的Oracle
增益、相对5A改善显著、无timeout，并在多冲突层保持正收益。

因此冻结：

```text
switch_on_threshold = 0.44
switch_off_threshold = 0.34
minimum_hold_steps = 3
```

不运行额外阈值网格，不启动G2-B v2反事实采集。下一步直接运行固定200场准入。
