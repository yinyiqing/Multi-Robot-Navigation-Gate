# G18 Dense256当前大Actor复测

状态：`registered / sequential suite queued`。登记日期：`2026-08-15`。

## 目的

在历史dense256完全相同的场景、seed和物理协议上补跑当前流程匹配容量baseline
`R2B-best`和epoch-17 Gate套件，判断当前方法在高密度多冲突压力集上的表现。该实验
只做评测，不更新任何Actor、Gate或阈值，不读取sealed test。

## 固定协议

- manifest：`datasets/fixed_v1/dense/validation.json.gz`；
- manifest SHA-256：`2d1dde389f927b924fa5993c47460bc60bac42aa9506ae3869c3139c9d1264b7`；
- 场景：manifest原始顺序前`256`场；
- 冲突构成：0-edge `8`、1-edge `61`、2-edge `80`、3+-edge `107`；
- seed：`20260810`；固定物理步长`0.001`；
- Actor：`capacity_wide_r2b_5a_recipe_n5_seed20260823_best`；
- Actor SHA-256：`da28dd5820d09845eea07cb68da45a7afd262fe56e8a71f80bf6b5781551523a`。

R2B-best是按5A流程和预算选择出的训练流程输出，但其best位于Actor有效更新前，新增容量
没有被使用。该限制不因本次评测改变；本实验不是“训练充分大Actor”的最终答案。

R2B-best完成后，严格顺序运行：

1. 5A；
2. epoch-17 + F-A1；
3. epoch-17 + F-B2；
4. epoch-17 + old B2直接迁移；
5. epoch-17 + 2m真值距离规则。

old B2直接迁移只作诊断；F-A1和F-B2才是使用epoch-17数据重建的Gate。2m规则是
privileged固定规则，不是严格性能上界。

## 执行与报告

```bash
bash scripts/start_g18_dense256_r2b.sh
bash scripts/start_g18_dense256_gate_suite.sh
```

任务使用全局单Gazebo锁；若G17仍运行，则在后台等待锁，不并发启动仿真。结果必须满足：

- shape为`(256, 17)`；
- scenario ID逐项等于dense validation前256场；
- 终止计数为`256 x 5 = 1280`。

第二个入口会等待第一个入口成功归档，再自动运行后续五组。完成后报告overall和
`0/1/2/3+`分层，并与历史同场epoch-16 B2、2m规则、R2-10k比较。

- R2B实时日志：`logs/active/g18-dense256-r2b/`；
- Gate套件实时日志：`logs/active/g18-dense256-gate-suite/`；
- 完成归档：`logs/archive/validation/g18_dense256_r2b/`和
  `logs/archive/validation/g18_dense256_gate_suite/`。
