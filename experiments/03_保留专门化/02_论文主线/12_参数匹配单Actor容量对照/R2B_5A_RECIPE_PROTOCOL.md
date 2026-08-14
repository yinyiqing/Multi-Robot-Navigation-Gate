# G12-R2B：按5A流程重训参数匹配大Actor

状态：`completed / 30k frozen as fair degraded baseline`。登记日期：`2026-08-14`。

## 目的

当前R2-10k通过“随机初始化单车broad -> 两车 -> 三车 -> 五车”课程得到，在G11-F-C同场
pilot上的full success为`0.760`，高于F-A1的`0.710`。R2与历史5A的训练血缘不同，因此
新增R2B回答更窄的问题：

> 当初始化、五车训练场景、reward、Critic信息、学习率、Actor冻结边界和总预算均按5A
> 流程设置时，单纯把Actor参数量扩宽到两个Actor之和会得到什么结果？

R2B是新增的流程匹配公平性对照，不以“必须低于Gate”为停止或选择目标。已经观察到的
R2结果保留为cross-protocol诊断，但由于额外课程与预算不匹配，不作为论文公平容量baseline。

## 与5A逐项对齐

| 项目 | 历史5A | R2B |
| --- | --- | --- |
| warm start | 3D2 best，仅Actor | 同一3D2 Actor函数保持扩宽 |
| Actor | `24-800-600-2` | `24-1137-855-2` |
| Actor参数 | `501,802` | `1,003,127` |
| 控制方式 | 五车共享Actor，全程控制 | 相同 |
| train场景 | procedural `standard` | 相同，固定seed `20260823` |
| reward | individual | 相同 |
| Critic | fresh 24维Critic | 相同 |
| local Critic/dynamic reward | 关闭/关闭 | 相同 |
| Actor/Critic LR | `2e-6/2e-5` | 相同 |
| exploration | `0.025 -> 0.012` | 相同 |
| Actor更新延迟 | `20,000` samples | 相同 |
| 总预算 | `30,000` samples | 相同 |

函数保持扩宽必须在训练前通过：输出最大误差`<=1e-5`、参数量审计以及新增分支梯度非零。
源Actor SHA-256固定为
`9be0658c1f33505103f2a3e92714de3fd3759bf5d7eecec878657f43987333b5`。

## 唯一有意修订：可复现选择

历史5A每5k在40个随机standard场景上选择best，epoch 4位于Actor解冻边界；后续epoch 5/6
明显退化。其best与3D2的权重最大差只有约`3.4e-5`，孤立峰值可能混有随机validation
波动。R2B不复制这一缺陷：

- 每10k评测一次，只产生10k、20k、30k三个候选；
- 使用已在R2启动前冻结的`g12_r2_curriculum_v1/n5/validation.json.gz`共120场；
- manifest SHA-256为
  `e33dbfad3d166fa4500b5997902a94c49108c77c646c7a39c26480b5054daef7`；
- 不读取G11-F-C、G11-D2、G11-E或sealed test选择checkpoint；
- 先满足collision不高于5A同场值、timeout增量不超过`0.02`，再最大化full success；
- 若无候选满足约束，则R2B判为未准入，不从G11-F-C反向挑模型。

冻结候选后，才在G11-F-C相同manifest和两个repeat上与5A、F-A1、F-B2、R2-10k比较。

## 执行

```bash
bash scripts/start_training_g12_r2b_5a_recipe.sh
```

- 模型：`capacity_wide_r2b_5a_recipe_n5_seed20260823`；
- 完整日志：`logs/archive/training/capacity_wide_g12_r2b_5a_recipe/`；
- 初始审计：本实验目录`local_data/r2b_5a_recipe/initialization_audit.json`；
- 预计耗时约3小时；只运行一个seed，先完成最小准入，不自动扩展多seed。

R2B不修改冻结的5A、epoch-17、F-A1或现有R2 artifact。

## 完成结果

10k/20k/30k的full success为`0.533/0.508/0.042`，collision为
`0.195/0.200/0.418`，timeout为`0/0/0.150`。10k候选仍与扩宽前3D2函数等价；30k在
约10k Actor更新后明显坍塌。项目接受退化作为对照结果，因此冻结R2B-30k作为流程/
预算匹配大Actor baseline，不使用冻结期自动best。当前追加G11-F-C同场评测。完整分析见
[R2B结果](R2B_5A_RECIPE_RESULTS.md)。
