# G12-R2-S4五车Broad首段协议

状态：`frozen / first 20k pilot authorized`。日期：`2026-08-07`。

## 1. 目标

S3已经稳定通过三车首段，20k为冻结best。S4直接扩大到论文目标的五车完整场景，检验
加宽Actor能否形成R2普通五车导航底座。通过后才允许进入R3完整standard/dense混合分布
和强交互重采样；本阶段不调用Gate、5A或epoch-16。

## 2. 冻结输入

```text
experiment: G12-R2-S4-n5-pilot
model: capacity_wide_r2_s4_broad_n5_seed20260816
seed: 20260816
agents: 5
Actor: 24 -> 1137 -> 855 -> 2
warm start: S3 20k best Actor + Critic
Actor SHA-256: 0ad69f89378b88812c1ce2306a07c75fbd4d80a9616b1db3a18e6d36c9037f04
Critic SHA-256: 55a20491f6f498960d77284e44409c99d7d710bb5a39fb18a212a3d047650d67
train SHA-256: 82f990dab54331ef55d3818fbe39b31fe00480dd99696987a5b85c5e2581ac1e
validation SHA-256: e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7
```

train与validation分别为`3000/120`场，不按冲突边筛选。train边数分布为
`0:1159 / 1:1323 / 2:388 / 3:100 / 4:27 / 5:3`；validation固定为40个0-edge、
40个edge-1和40个multi-edge。G11-D2、G11-E和sealed test均不读取。

## 3. 首段配置

| 项目 | 值 |
| --- | --- |
| budget | `2 x 10k = 20k agent samples` |
| evaluation | 每10k评测固定n5 validation 120场 |
| replay | fresh，minimum `5000`，batch `256` |
| Actor/Critic | S3 20k best完整warm start，不允许actor-only fallback |
| gamma / tau | `0.999 / 0.005` |
| Actor LR / Critic LR | `8e-5 / 8e-5` |
| exploration | `0.10 -> 0.03` over 20k，无随机直行动作 |
| Actor anchor | `0` |
| reward | individual navigation基础reward |
| privileged input | 无local Critic、无oracle、无Gate |
| physics | fixed step `0.001` |

## 4. 准入

1. 10k和20k均完整保存与评测；
2. 20k agent success不低于`0.65`、full success不低于`0.30`；
3. collision不高于`0.30`、timeout不高于`0.15`；
4. 20k相对10k的full success下降至少`0.10`，或timeout增加至少`0.10`时回滚10k；
5. NaN、Q爆炸、持续固定动作或模型加载降级时立即停止。

通过只表示R2已形成可继续训练的五车底座，不表示达到论文对照要求。下一步必须按R3协议
加入完整standard/dense broad stream和强交互重采样，再用同一冻结validation分层判断。

## 5. 执行

```bash
DRL_MULTI_DRY_RUN=1 bash scripts/experiment.sh start actor-g12-r2-s4-n5
bash scripts/experiment.sh start actor-g12-r2-s4-n5
bash scripts/experiment.sh status
bash scripts/experiment.sh stop actor-g12-r2-s4-n5
```

实时日志写入`logs/active/capacity-wide-g12-r2/s4-n5/`。

