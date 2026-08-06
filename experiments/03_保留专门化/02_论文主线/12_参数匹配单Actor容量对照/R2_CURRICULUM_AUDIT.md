# G12-R2历史课程审计

状态：`audit complete / historical scripts are evidence only`。日期：`2026-08-06`。

## 1. 审计结论

历史artifact名称给出的名义链路是：

```text
TD3_velodyne_multi_v4
-> stage1_single -> stage1e -> stage1f -> stage1g
-> 2A -> 2D -> 3A -> 3D2 -> 5A
```

但不能把每个箭头都解释为Actor学到了新能力：

| 节点 | 运行预算 | best位置 | Actor更新解释 |
| --- | ---: | ---: | --- |
| stage1_single | `40k` | epoch 4，约`20k` | 有Actor更新 |
| stage1e | `50k` | epoch 8，约`40k` | 有Actor更新 |
| stage1f | `40k` | epoch 4，约`20k` | 有Actor更新 |
| stage1g | `30k` | epoch 4，约`20k` | 有Actor更新 |
| 2A | `60k` | epoch 2，约`10k` | 有Actor更新 |
| 2D gentle | `50k` | epoch 5，约`25k` | best位于25k解冻边界，不能证明Actor改善 |
| 3A guarded | 约`25k` | epoch 4，约`20k` | best位于20k解冻边界，随后立即退化 |
| 3D2 | 约`85k`后停止 | epoch 3，约`15k` | best早于20k解冻，Actor等同3A输入 |
| 5A | `30k` | epoch 4，约`20k` | best位于20k解冻边界，随后立即退化 |

历史选中checkpoint路径约包含`190k` agent samples，但这个数不完整，因为首个
`stage1_single`从`TD3_velodyne_multi_v4` warm start，而该基础模型的训练数据、seed和
累计预算没有在当前仓库登记。沿历史路径真正有明确Actor更新证据的新增预算主要到2A为止，
约`110k`；2D/3A/3D2/5A更多证明继承Actor能在不同配置中工作。

因此R2不能声称“照原命令从随机初始化复现5A”。直接执行历史脚本还会引入两个问题：

1. standard阶段使用在线随机场景，没有可重放manifest；
2. D分支反复创建fresh Critic，且best常在Actor解冻前，R1已经证明这种无约束解冻会破坏
   已有策略。

## 2. 保留的历史证据

单车定向补课仍有价值，固定case为：

| case文件 | cases | SHA-256 | 历史selected budget |
| --- | ---: | --- | ---: |
| `stage1_single_local_cases.json` | `6` | `9cc79ec2a82908127c77fa00eff1448661814f8025176d769ccd4a03a8fb4b40` | `20k` |
| `stage1e_single_rescue_cases.json` | `12` | `3b2566d8898d5380bc4d5295009d0b81e088bd96bec63939d5184d88a8cce4d9` | `40k` |
| `stage1f_wall_parallel_rescue_cases.json` | `12` | `36906f6164d79551a09264f10e939779c68b8ca8ab366e78b50911f73974f563` | `20k` |
| `stage1g_collision_guard_cases.json` | `12` | `d52dd8d1b5dd904ad7f4b8c55b60a258fc5cc4616469ead9932d93ee11be4403` | `20k` |

其中stage1e/f/g是已有导航能力的定向修补，不适合作为随机网络的第一批数据。R2先建立
broad单车导航，再按固定case补课。

## 3. R2采用的有效链路

```text
S0 random wide Actor：n1 broad standard
-> S1：固定单车case定向补课并复核broad能力
-> S2：n2 broad standard
-> S3：n3 broad standard
-> S4：n5 broad standard
```

S2-S4保持原始24维Critic结构不变，阶段间完整加载Actor和Critic；不插入2D/3D2 fresh
Critic支路。这样保留“逐步增加车数”的课程思想，同时避免把历史上没有Actor学习证据的
节点当作必经步骤。R3/R4才引入完整standard/dense与强交互重采样。

## 4. 结论边界

该审计不证明D/3D2方法无效，只证明历史selected checkpoint不足以说明这些节点训练了
更好的Actor。R2若失败，首先检查随机初始化基础阶段和优化稳定性，不能归因于参数量。

