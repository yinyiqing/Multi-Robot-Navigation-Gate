# G11-F epoch-17可部署Gate

状态：`A1 offline rebuild registered`。登记日期：`2026-08-14`。

## 目的

在冻结`generalist-5a + avoidance-epoch17`后重建可部署在线Gate。旧G11-A1/B2的结构、
特征和DAgger流程继续作为已验证起点，但旧checkpoint不能直接进入当前方法，因为其候选
动作特征与student访问分布均来自epoch-16。

## 冻结组件

| 组件 | SHA-256 |
| --- | --- |
| 5A Actor | `fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5` |
| epoch-17避障Actor | `149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5` |
| G0 detector | `0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56` |

epoch-17是续训分支自动best的序列化artifact，经审计与epoch 17快照tensor一致。两个Actor
继续冻结；本目录只训练Gate。

## F-A1：离线重建

复用G11-A1的640场train与120场内部validation shard。这些轨迹由冻结5A产生，shard保存
每帧24维Actor状态，因此无需重新运行Gazebo：训练器从相同状态离线计算5A与epoch-17的
候选动作和动作差，再训练相同的8帧GRU Gate。

- manifest、标签、detector、网络结构、40 epoch、主seed `20260804`和FPR约束与旧A1一致；
- S0不读取Actor动作，用作同场FPR上限；
- T1输入为76维本机特征加6维两个Actor动作及动作差；
- 只读取navigation-train内部数据，不读取sealed test；
- 输出写入`local_data/a1_training/seed20260804/`，不得覆盖旧A1。

入口：

```bash
bash scripts/run_g11_f_epoch17_a1_training.sh
```

只有T1满足旧A1的离线准入方向，才允许做1场student-rollout smoke。smoke通过后才启动
640场新student采集；旧epoch-16 B1 shard不得混入当前聚合训练。

## 后续固定顺序

1. F-A1离线重建；
2. epoch-17 student-rollout smoke；
3. navigation-train 640场student rollout；
4. 聚合5A shard与新student shard重训Gate；
5. 小规模闭环pilot后再做独立validation与multi-edge评测。

最终Gate只能读取本机观测、两个冻结Actor候选动作和短时历史。`2.0 m`其他机器人真值
只生成训练标签和oracle上界，不进入部署Gate。
