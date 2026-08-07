# G12-R2-S2两车Broad首段协议

状态：`completed / passed / 10k best frozen`。日期：`2026-08-07`。

## 1. 路线修订

S1 repair-only更新因窄分布遗忘被拒绝，但S0在五次broad n1 validation上保持
`0.983-1.000`。固定单车困难case不再作为进入多车训练的硬前置：它们用于记录S0能力
边界，不能让参数匹配单Actor长期停留在单车阶段。

S2因此直接从S0 epoch 2 best完整warm start，跳过失败S1 checkpoint。该修订不恢复
repair候选，也不把S0的单车成绩解释成多车能力；S2本身就是第一次正式多车训练。

## 2. 冻结输入

```text
experiment: G12-R2-S2-n2-pilot
model: capacity_wide_r2_s2_broad_n2_seed20260814
seed: 20260814
agents: 2
Actor: 24 -> 1137 -> 855 -> 2
warm start Actor: S0 epoch 2 best
warm start Critic: S0 epoch 2 best
Actor SHA-256: 7cb61925a4188e638859f88d38288e0431e5f05be489fa6107a77c7efaed3822
Critic SHA-256: 1d5e9bbcc7062886548cf2691ce446993bb5a04be11d5a517cbcb9fa610ad752
train SHA-256: 5fbd2df5241076041ea714b59286604915ebf1b13848482f7c34fd10cdc9087b
validation SHA-256: 955132263cac9496a56eb8bb6f5132ca5ae41e930c926a7a9a13e8797bb903c9
```

train与validation各保留原冻结standard来源的全部`3000/120`个scenario ID，但每个场景
只保留`r1/r2`。不按冲突边数筛选，不读取G11-D2、G11-E或sealed test。

## 3. 首段配置

| 项目 | 值 |
| --- | --- |
| budget | `2 x 10k = 20k agent samples` |
| evaluation | 每10k评测固定n2 validation 120场 |
| replay | fresh，minimum `5000`，batch `256` |
| Actor/Critic | S0 best完整warm start，不允许actor-only fallback |
| gamma / tau | `0.999 / 0.005` |
| Actor LR / Critic LR | `8e-5 / 8e-5` |
| exploration | `0.10 -> 0.03` over 20k，无随机直行动作 |
| Actor anchor | `0` |
| reward | individual navigation基础reward |
| privileged input | 无local Critic、无oracle、无Gate |
| physics | fixed step `0.001` |

两车共享同一个Actor并全程独立输出动作。这不是两个Actor切换实验，也不更新当前方法中的
5A或epoch-16。

## 4. 首段判断

首段只回答“能否从S0稳定进入两车完整训练”，不要求20k已经形成最终五车baseline。

1. 10k和20k均完整保存与评测；
2. 20k的agent success不低于`0.80`、full success不低于`0.65`；
3. collision不高于`0.15`、timeout不高于`0.10`；
4. 若20k相对10k的full success下降至少`0.10`，或timeout增加至少`0.10`，回滚10k；
5. 出现NaN、Q爆炸、持续固定动作或模型加载降级时立即停止。

通过后才登记剩余40k并决定S2 best；未通过先检查训练稳定性，不回到repair-only单车训练，
也不能直接写成大Actor容量不足。

## 5. 执行

```bash
DRL_MULTI_DRY_RUN=1 bash scripts/experiment.sh start actor-g12-r2-s2-n2
bash scripts/experiment.sh start actor-g12-r2-s2-n2
bash scripts/experiment.sh status
bash scripts/experiment.sh stop actor-g12-r2-s2-n2
```

运行时日志写入`logs/active/capacity-wide-g12-r2/s2-n2/`。

完成结果见[R2-S2两车结果](R2_S2_N2_RESULTS.md)。日志已归档到
`logs/archive/training/g12_r2/s2_n2/`，下一阶段见[S3三车协议](R2_S3_N3_PROTOCOL.md)。
