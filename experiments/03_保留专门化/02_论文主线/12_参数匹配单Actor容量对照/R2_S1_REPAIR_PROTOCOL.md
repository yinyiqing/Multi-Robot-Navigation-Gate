# G12-R2-S1定向补课协议

状态：`frozen / first 20k pilot authorized`。日期：`2026-08-06`。

## 1. 目的与边界

本实验只修复S0在固定单车困难case中暴露的贴墙与脱离能力，然后继续`n2 -> n3 -> n5`
课程。它属于参数匹配单Actor公平对照，不修改当前方法中的5A、epoch-16或Gate。

首段只授权`20,000 agent samples`，不得自动跑满S1的`80k`上限。训练不读取G11-D2、
G11-E、navigation test或sealed test。

## 2. 冻结输入

```text
experiment: G12-R2-S1-repair-pilot
model: capacity_wide_r2_s1_repair_n1_seed20260813
seed: 20260813
Actor: 24 -> 1137 -> 855 -> 2
warm start: S0 epoch 2 best Actor and Critic
Actor SHA-256: 7cb61925a4188e638859f88d38288e0431e5f05be489fa6107a77c7efaed3822
Critic SHA-256: 1d5e9bbcc7062886548cf2691ce446993bb5a04be11d5a517cbcb9fa610ad752
repair cases SHA-256: df5267aac5b671befe1df2c64d82503bda9acd4770b02e774e02717d32a9ace5
broad validation SHA-256: 9ab4c5913f683d01e3ab186ea591d373abe1e835180f4a0bfeb469990269b125
```

训练集为诊断得到的8个几何去重case。为避免同一几何因历史stage重复而被隐式放大，
每个几何只保留一次；近障恢复与反向脱离权重各为`3`，clear和safe两个三case组内权重
各为`1`，四组总权重均为`3`。

## 3. 训练配置

| 项目 | 值 |
| --- | --- |
| budget | `20k`，仅一个epoch |
| replay | fresh，minimum `5000`，batch `256` |
| Actor/Critic | 两者均从S0 best加载，不允许actor-only fallback |
| gamma / tau | `0.999 / 0.005` |
| Actor LR / Critic LR | `6e-5 / 6e-5` |
| exploration | `0.08 -> 0.025` over `20k`，无随机直行动作 |
| reward | S0 individual navigation基础reward + 冻结local-navigation与wall-clearance shaping |
| Actor anchor | `0` |
| evaluation | 更新20k后评测固定broad n1 `120`场 |

不恢复S0 replay或优化器状态，但加载其Actor和Critic权重；因此不是P1/R1中的fresh-Critic
解冻。repair case使用加权随机采样，broad validation使用冻结顺序cycle。

## 4. 首段停止与继续规则

20k结束后先读broad n1，不先看targeted结果：

1. full success至少`117/120 = 0.975`；
2. collision不超过`3/120 = 0.025`；
3. timeout不超过`3/120 = 0.025`；
4. 无NaN、Critic爆炸、动作持续单侧饱和或模型加载降级。

任一项失败，候选回滚且不得进入targeted复测。全部通过后，才用与S1诊断相同的126场
协议评测困难case。若targeted无明确改善，先审计实现与reward，不自动追加预算；只有
broad保持且targeted改善时，才另行登记下一段20k，S1累计上限仍为80k。

## 5. 执行

```bash
DRL_MULTI_DRY_RUN=1 bash scripts/experiment.sh start actor-g12-r2-s1-repair
bash scripts/experiment.sh start actor-g12-r2-s1-repair
bash scripts/experiment.sh status
bash scripts/experiment.sh stop actor-g12-r2-s1-repair
```

日志写入`logs/active/capacity-wide-g12-r2/s1-repair/`，完成后归档到
`logs/archive/training/g12_r2/s1_repair/`。
