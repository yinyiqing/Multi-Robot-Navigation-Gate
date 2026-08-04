# G11-A1 当前协议时序 Gate Pilot

状态：`manifest frozen and audited / collection pending`。日期：`2026-08-04`。

## 目的

在不读取导航validation/test和sealed test的前提下，用当前论文协议的数据确认8帧GRU
能否学习360度`2.0 m`特权Oracle。两个Actor和G0 detector全部冻结；本阶段不做Actor
训练，也不把离线分类指标写成闭环导航收益。

## 数据协议

数据全部来自导航`fixed_v1` train，并按以下4层确定性抽取：

| stratum | train | validation |
| --- | ---: | ---: |
| standard / full-path 0-edge | 160 | 30 |
| dense / full-path 0-edge | 160 | 30 |
| standard / corrected full-path edge-1 | 160 | 30 |
| dense / corrected full-path edge-1 | 160 | 30 |
| 合计 | 640 | 120 |

- seed：`20260804`；
- train和validation scenario ID互斥；
- 排除旧G0/G2 pilot train/validation的全部200个scenario ID；
- 0-edge重新规划完整静态路径并要求`full_path_conflict_edges == 0`；
- edge-1只来自`edge1_full_horizon_v1/train.json.gz`；
- 不读取导航validation/test和感知sealed test。

manifest由`scripts/build_g11_a1_gate_views.py`生成，输出到
`datasets/fixed_v1/views/g11_a1_gate_v1/`。

| split | 场景 | SHA-256 |
| --- | ---: | --- |
| train | 640 | `a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026` |
| validation | 120 | `e261a7afbac8f7341ab13609c2662a2824a0ff383789287ad7733290389cd99d` |

构建时额外剔除6个stored 0-edge但full-path edge-1的standard场景。独立全量复算确认
train为`0->0:320, 1->1:320`，validation为`0->0:60, 1->1:60`，缺失边为0；
确定性二次重建哈希不变。

## 固定模型

| 组件 | SHA-256 |
| --- | --- |
| 5A Actor | `fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5` |
| epoch-16 Actor | `6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b` |
| G0 detector | `0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56` |

采集轨迹使用冻结5A；Oracle标签只由仿真中的其他机器人真值位置生成，不进入Gate输入。

## 固定比较

1. `S0`：76维单帧静态Gate；
2. `T1`：S0、两个Actor候选动作与动作差的最近8帧GRU；
3. 只训练360度`any`标签；S1和front分支已由A0关闭；
4. 主seed为`20260804`，表示通过后再用4个seed复核波动。

阈值必须在训练前冻结为带约束选择：先要求总体FPR和0-edge/weak FPR不高于同seed S0，
再最大化F1；若没有可行点则判失败，不退化为无约束最大F1。

## 准入与停止

A1只有同时满足以下条件才进入student-rollout数据聚合：

1. T1相对S0的F1、AP和区间IoU提高；
2. 总体FPR和0-edge/weak FPR不高于S0；
3. 事件recall不下降超过2个百分点；
4. 切换次数不增加；
5. 4个复核seed中至少3个保持同方向。

若主seed失败，停止训练，不追加seed、不启动闭环。若主seed通过但复核失败，A1仍判失败。
本阶段checkpoint只允许进入后续navigation-train student rollout，不能直接读取sealed test。

## 采集入口

先做一场train smoke，审计shard后再跑完整train和validation：

```bash
DRL_G11_A1_TARGET_EPISODES=1 bash scripts/experiment.sh start gate-g11-a1-train
bash scripts/experiment.sh status

bash scripts/experiment.sh start gate-g11-a1-train
bash scripts/experiment.sh start gate-g11-a1-validation
```

采集强制设置`CUDA_VISIBLE_DEVICES=""`并以`nice 10`在CPU运行；日志在
`logs/active/g11_a1/`，shard在本目录
`local_data/shards/`。停止命令为
`bash scripts/experiment.sh stop gate-g11-a1-<train|validation>`。

两个split完整采集并通过审计后，主seed训练的唯一入口为：

```bash
bash scripts/run_g11_a1_training.sh 20260804
```

该入口在采集未结束时拒绝启动，并强制CPU、冻结manifest、`any + S0/T1`以及
`match-s0-fpr`阈值策略。训练前会自动运行`scripts/audit_g11_a1_shards.py`，逐文件
验证manifest覆盖、split/stratum、帧和候选形状、时间顺序及有限值；审计失败时不训练。
