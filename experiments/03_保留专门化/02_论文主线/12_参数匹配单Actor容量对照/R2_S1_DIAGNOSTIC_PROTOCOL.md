# G12-R2-S1固定困难Case诊断协议

状态：`completed / repair required`。日期：`2026-08-06`。

## 1. 目的

本实验只判断S0冻结best是否仍缺少历史单车课程中的具体能力。它不更新Actor或Critic，
不是论文validation或test，也不用于宣称多车能力。只有诊断确认缺口后才登记S1训练。

## 2. 冻结输入

```text
Actor: capacity_wide_r2_s0_broad_n1_seed20260811_best_actor.pth
Actor SHA-256: 7cb61925a4188e638859f88d38288e0431e5f05be489fa6107a77c7efaed3822
evaluation seed: 20260812
repeats per case: 3
sampling: deterministic cycle
agents: 1
max episode steps: 300
```

| stage | cases | episodes | case SHA-256 |
| --- | ---: | ---: | --- |
| `stage1_single` | 6 | 18 | `9cc79ec2a82908127c77fa00eff1448661814f8025176d769ccd4a03a8fb4b40` |
| `stage1e_single_rescue` | 12 | 36 | `3b2566d8898d5380bc4d5295009d0b81e088bd96bec63939d5184d88a8cce4d9` |
| `stage1f_wall_parallel_rescue` | 12 | 36 | `36906f6164d79551a09264f10e939779c68b8ca8ab366e78b50911f73974f563` |
| `stage1g_collision_guard` | 12 | 36 | `d52dd8d1b5dd904ad7f4b8c55b60a258fc5cc4616469ead9932d93ee11be4403` |

总计42个固定case、126个episode。结果必须逐case恰好包含3次，不允许缺失、重复轮次或
用不同case数的历史汇总替代。

## 3. 决策规则

每个case按3次full success分类：

- `pass`：`3/3`成功；
- `borderline`：`2/3`成功，先按同协议追加3次确认，不直接训练；
- `repair`：`0/3`或`1/3`成功，允许为该case组登记S1补课。

只有全部42个case均为`pass`时，S1训练预算记为`0`并直接进入S2。出现borderline时先复核；
出现repair时只训练对应case组，并在每段更新后同时回测该组与S0 broad n1 validation。

## 4. 执行

```bash
bash scripts/experiment.sh start actor-g12-r2-s1-diagnostic
bash scripts/experiment.sh status
bash scripts/experiment.sh stop actor-g12-r2-s1-diagnostic
```

日志写入`logs/active/capacity-wide-g12-r2/s1-diagnostic/`，完成后归档到
`logs/archive/validation/g12_r2_s1_diagnostic/`。结构化结果写入本目录
`local_data/s1_diagnostic/summary.json`。

## 5. 无效运行记录

`2026-08-06`首次运行在`stage1_single`结束后未清理其ROS/Gazebo进程，后续stage与同一
master上的残留仿真发生模型名冲突。该运行全部作废，不读取任何stage指标；日志归档到
`logs/archive/diagnostic/g12_r2_s1_launcher_cleanup_failed_20260806/`，结构化中间结果移至
`local_data/s1_diagnostic_invalid_launcher_cleanup_20260806/`。执行器现要求每个stage
结束后终止完整进程组并确认`14621/14721`端口释放，随后从126场完整重跑。

有效重跑已经完成：`126/126`场中full success为`72/126`，42个stage-case项中
`22 pass / 0 borderline / 20 repair`。完整结果与下一步边界见
[R2_S1_DIAGNOSTIC_RESULTS](R2_S1_DIAGNOSTIC_RESULTS.md)。
