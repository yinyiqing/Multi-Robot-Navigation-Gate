# G12-R2加宽单Actor普通导航课程协议

状态：`S0 passed / S1 diagnostic completed and repair required / S2-S4 budgets preregistered`。
日期：`2026-08-06`。

## 1. 固定边界

- Actor：`24 -> 1137 -> 855 -> 2`，共`1,003,127`参数；
- S0的Actor和原始24维Critic均随机初始化，不加载5A或任何历史Actor；
- 全部阶段只用本车24维部署观测，单一共享Actor全程控制当前全部机器人；
- train来自`g12_r2_curriculum_v1/*/train`，validation来自对应车数的同视图；
- 不读取Gate训练数据、G11-D2、G11-E、navigation test或sealed test；
- seed：`20260811`。

历史课程审计见[R2_CURRICULUM_AUDIT](R2_CURRICULUM_AUDIT.md)。

## 2. 课程预算

| stage | agents | 数据 | 预算 | 初始化 |
| --- | ---: | --- | ---: | --- |
| S0 broad | 1 | standard 3000场完整episode | `100k` | Actor/Critic随机 |
| S1 repair | 1 | stage1/e/f/g固定case | 最多`80k` | S0完整warm start |
| S2 broad | 2 | standard 3000场完整episode | `60k` | S1 Actor/Critic完整warm start |
| S3 broad | 3 | standard 3000场完整episode | `60k` | S2 Actor/Critic完整warm start |
| S4 broad | 5 | standard 3000场完整episode | `80k` | S3 Actor/Critic完整warm start |

基础总预算上限为`380k`。S0若持续改善但未达准入，可在不改变超参数的前提下预注册扩展
到`200k`；若动作饱和、NaN、Q爆炸或连续两个eval明显恶化则停止诊断，不扫描validation。
S1每个case阶段后必须回测broad n1 validation，出现遗忘则回滚，不能只按targeted峰值前进。
S0结束后先用冻结best对stage1/e/f/g固定case做只评测诊断，不默认启动S1更新。只有诊断
确认具体缺口时才使用S1的最多`80k`预算；若固定case全部通过，S1训练预算记为`0`并直接
进入S2。这不会改变S2-S4的数据、预算或初始化边界。

S2-S4不经过2D/3D2。原始Critic输入始终为24维且reward保持individual navigation，
因此阶段间完整warm start Actor和Critic，避免fresh Critic解冻造成P1/R1式坍塌。

## 3. S0冻结配置

```text
experiment: G12-R2-S0
model: capacity_wide_r2_s0_broad_n1_seed20260811
train: g12_r2_curriculum_v1/n1/train.json.gz
  SHA-256 c71e4e87bbc528782cb76dc7df076c493900523bb748b3fb646f3d77fa5f0263
validation: g12_r2_curriculum_v1/n1/validation.json.gz
  SHA-256 9ab4c5913f683d01e3ab186ea591d373abe1e835180f4a0bfeb469990269b125
budget: 5 x 20k = 100k agent samples
eval: 120 episodes every 20k
batch: 256
minimum replay: 5000
gamma: 0.999
actor lr: 1e-4
critic lr: 1e-4
policy frequency: 2
tau: 0.005
exploration: 0.35 -> 0.08 over 100k
random linear exploration: first 5000 samples
reward: individual navigation, no local Critic, no cooperative reward
```

S0通过条件以120场validation为准：full success至少`0.85`，collision不高于`0.10`，
timeout不高于`0.10`，且动作与Q诊断无异常。未通过只说明S0尚未形成基础导航，不构成
大Actor容量结论。

S0已经按100k预算完成并通过。五次full success为
`0.983/1.000/1.000/0.992/1.000`，冻结best位于epoch 2。完整结果、诊断和artifact哈希见
[R2_S0_RESULTS](R2_S0_RESULTS.md)。

S1训练前置固定case诊断已经冻结为每case 3次的
[S1诊断协议](R2_S1_DIAGNOSTIC_PROTOCOL.md)。该诊断不更新网络；只在出现可复现缺口后
登记对应case组的训练配置。

S1诊断已经完成，full success为`72/126`，得到`22 pass / 0 borderline / 20 repair`。
缺口集中在近障恢复、贴墙姿态、离墙推进和反向脱离；结果见
[S1诊断结果](R2_S1_DIAGNOSTIC_RESULTS.md)。8个几何去重case、首段20k预算和broad回归
门槛已在[S1补课协议](R2_S1_REPAIR_PROTOCOL.md)冻结，不能直接按历史stage顺序自动跑满
`80k`。

## 4. 启停与日志

```bash
bash scripts/experiment.sh start actor-g12-r2-s0
bash scripts/experiment.sh status
bash scripts/experiment.sh stop actor-g12-r2-s0
```

日志统一写入`logs/active/capacity-wide-g12-r2/`。完成后归档到
`logs/archive/training/g12_r2/`，并在启动S1前记录checkpoint哈希和S0准入结果。
