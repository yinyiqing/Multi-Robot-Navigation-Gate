# G12-R2-S4五车Broad首段结果

状态：`completed / passed / 20k best frozen`。日期：`2026-08-07`。

## 运行与结果

从S3 20k best完整warm start，在五车完整standard train上完成`20,000 agent samples / 
8,178 env steps / 331 episodes`。固定120场内部validation包含40个0-edge、40个edge-1和
40个multi-edge场景。

| checkpoint | agent success | full success | collision | unresolved | timeout | avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | `0.8867` | `0.6667` | `0.1117` | `0.0017` | `0.0083` | `20.76` |
| 20k | `0.8917` | `0.6917` | `0.1000` | `0.0083` | `0.0417` | `35.45` |

20k的full success提高`0.0250`、agent success提高`0.0050`、collision下降`0.0117`；timeout
提高`0.0333`且平均步数增加`14.69`。未触发预注册的`0.10`回滚线，因此按full success
选择冻结20k，但必须明确它不是全面支配10k。

```text
model: capacity_wide_r2_s4_broad_n5_seed20260816_best
timestep: 20000
env steps: 8178
episode: 331
Actor SHA-256: 67290450484c1fedd493fb029804b914438c5fb46cdb189ba8c642c3d98b2715
Critic SHA-256: 8ff9483045e5945b7b2b84e124a998d9890d441f0ac5028975026d582190f542
checkpoint SHA-256: 597d916830e19212bb7ad27574538fe9c5aeefb01f8073326a0ea3bf5b17314b
```

## 结论边界

- R2已形成可继续评估的五车大Actor底座，没有发生训练坍塌；
- 当前训练评测没有逐scenario记录，不能拆分0-edge、edge-1、multi-edge或standard/dense；
- 进入R3前必须在同一内部validation上逐场配对评测5A与S4 best，验证普通能力保持；
- 本结果不是sealed test，也不是最终参数匹配单Actor论文成绩。

