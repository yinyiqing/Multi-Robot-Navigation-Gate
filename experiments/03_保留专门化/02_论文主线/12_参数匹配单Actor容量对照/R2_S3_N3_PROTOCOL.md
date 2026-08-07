# G12-R2-S3三车Broad首段协议

状态：`completed / passed / 20k best frozen`。日期：`2026-08-07`。

## 1. 目标

S2已经稳定通过两车首段，10k为冻结best。S3不继续追加简单两车预算，直接扩大到三车
完整broad分布，检验加宽Actor能否在增加机器人数量和冲突密度后保持导航稳定。

## 2. 冻结输入

```text
experiment: G12-R2-S3-n3-pilot
model: capacity_wide_r2_s3_broad_n3_seed20260815
seed: 20260815
agents: 3
Actor: 24 -> 1137 -> 855 -> 2
warm start: S2 10k best Actor + Critic
Actor SHA-256: 220698f1e4a918deb88d0b47f8c4f28b2330194401b4b82c80afe92d8f63f465
Critic SHA-256: acbecd846cbae2456e2a493ec545eeaf2718a11aa2cc6fe58c2a9d3af1fbe7ca
train SHA-256: b6ff22964a8b1795a783f8af9360c123fae44b4b44a86de63e76a57b4a0b4422
validation SHA-256: f4b7d46fc488eb588007aa7ba72791545e750e691399da82c65d5cdf9f5938cc
```

train与validation分别为`3000/120`场，每场保留`r1/r2/r3`。不按冲突边筛选：train的
派生边数为`0:2341 / 1:625 / 2:27 / 3:7`；validation为
`0:77 / 1:36 / 2:5 / 3:2`。G11-D2、G11-E和sealed test均不读取。

## 3. 首段配置

| 项目 | 值 |
| --- | --- |
| budget | `2 x 10k = 20k agent samples` |
| evaluation | 每10k评测固定n3 validation 120场 |
| replay | fresh，minimum `5000`，batch `256` |
| Actor/Critic | S2 10k best完整warm start，不允许actor-only fallback |
| gamma / tau | `0.999 / 0.005` |
| Actor LR / Critic LR | `8e-5 / 8e-5` |
| exploration | `0.10 -> 0.03` over 20k，无随机直行动作 |
| Actor anchor | `0` |
| reward | individual navigation基础reward |
| privileged input | 无local Critic、无oracle、无Gate |
| physics | fixed step `0.001` |

## 4. 准入

1. 10k和20k均完整保存与评测；
2. 20k agent success不低于`0.75`、full success不低于`0.55`；
3. collision不高于`0.20`、timeout不高于`0.12`；
4. 20k相对10k的full success下降至少`0.10`，或timeout增加至少`0.10`时回滚10k；
5. NaN、Q爆炸、持续固定动作或模型加载降级时立即停止。

本首段只决定是否继续五车课程，不产生参数匹配单Actor的最终论文成绩。由于评测器当前
只保存汇总，冲突分层结论必须等最终逐场统一评测，不能从本次汇总反推。

## 5. 执行

```bash
DRL_MULTI_DRY_RUN=1 bash scripts/experiment.sh start actor-g12-r2-s3-n3
bash scripts/experiment.sh start actor-g12-r2-s3-n3
bash scripts/experiment.sh status
bash scripts/experiment.sh stop actor-g12-r2-s3-n3
```

实时日志写入`logs/active/capacity-wide-g12-r2/s3-n3/`。

完成结果见[R2-S3三车结果](R2_S3_N3_RESULTS.md)。日志已归档到
`logs/archive/training/g12_r2/s3_n3/`，下一阶段见[S4五车协议](R2_S4_N5_PROTOCOL.md)。
