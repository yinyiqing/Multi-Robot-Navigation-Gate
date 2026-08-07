# G12-R2五车底座配对准入协议

状态：`20k completed and narrowly failed / one 10k fallback authorized`。日期：`2026-08-07`。

## 目的

S4训练内评测只保存汇总，无法验证进入R3所要求的0-edge能力保持和来源分层。本实验在
冻结的内部validation上重新串行运行5A与S4 20k best，逐场保存结果并完成配对审计。

## 冻结输入

```text
experiment: G12-R2-N5-admission
policies: generalist-5a, capacity-wide-r2-s4-n5
episodes per policy: 120
evaluation seed: 20260817
physics step: 0.001
manifest: g12-r2-curriculum-v1-n5-validation
manifest SHA-256: e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7
5A Actor SHA-256: fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5
S4 Actor SHA-256: 67290450484c1fedd493fb029804b914438c5fb46cdb189ba8c642c3d98b2715
```

两个策略使用完全相同的scenario ID、顺序、seed、最大步数、物理配置和单Actor执行模式。
每个策略单独重启ROS/Gazebo。G11-D2、G11-E和sealed test均不读取。

## 审计与准入

结果必须各为`120 x 17`记录，scenario ID与manifest顺序完全一致且无重复。报告overall、
0-edge/edge-1/multi-edge、standard/dense以及六个交叉层，并给出逐场full success改善、
退化、持平和McNemar exact结果。

进入R3同时要求：

1. 0-edge full success相对5A下降不超过`0.03`；
2. overall agent success相对5A下降不超过`0.02`；
3. overall timeout相对5A增加不超过`0.02`；
4. standard和dense任一来源的full success下降不超过`0.05`；
5. 结果、manifest和模型哈希审计全部通过。

该准入只决定是否允许启动R3，不读取sealed test，也不替代最终多seed统一比较。

## 20k结果后的固定fallback

20k仅因`3/120=0.025` timeout超过`0.020`上限而未通过，其余四项通过。不得事后放宽阈值。
S4协议已预先保存10k checkpoint，且训练内评测timeout更低，因此只授权一次同协议10k
fallback：复用已审计的5A逐场结果，按相同manifest、顺序、seed和阈值评测10k。若仍失败，
R2停止；不得继续选择其他checkpoint或seed。20k完整结果见
[配对准入结果](R2_N5_ADMISSION_RESULTS.md)。
