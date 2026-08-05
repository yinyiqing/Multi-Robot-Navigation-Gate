# G11-E Multi-edge自然泛化Pilot

状态：`protocol preparation / do not launch before G11-D2 completes`。登记日期：`2026-08-05`。

本实验回答A1/B2在只用0-edge与single-edge Gate训练数据时，能否自然改善exact-edge-2
导航。它不改变或训练两个Actor，也不把multi-edge标签输入Gate。

由于interaction-epoch16原训练集经完整路径复审含11个edge-2场景，本实验只称为
“Gate的自然拓扑泛化诊断”，不声称整个系统严格single-to-multi零样本泛化。

## 数据冻结

来源是旧exact-edge-2确认集的200场dense validation。该集合按scenario ID哈希排序产生，
未按策略结果或难度筛选，并与A1训练/内部validation、G11-C和G11-D2均无场景重叠。

保持来源冻结顺序进行一次性划分：

- 前50场：pilot，只决定是否值得扩大；
- 后150场：confirmation，pilot期间禁止读取。

固定视图位于：

```text
experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/
g11_e_edge2_generalization_v1/
```

冻结哈希：

- `pilot.json.gz`：`f73f260ee4394b11e21a791085cf4957ca50ae93f3466df3882f0d15da932c16`；
- `confirmation.json.gz`：`72b4b975825bd33900db21a0cd08e19f0f210d9875caa9e8aa86d95b6b268049`。

pilot/confirmation分别为50/150场、ID交集为0，全部`conflict_edge_count=2`；与A1
train/internal-validation、G11-C和G11-D2场景重叠均为0。

## Pilot方法

同一50场、同一seed、CPU串行比较：

1. 5A always-on；
2. old G2-A；
3. A1；
4. B2；
5. 2.0 m privileged oracle。

不运行min-LiDAR和epoch16 always-on：它们已在G11-D2中用于机制边界，pilot的新增问题
只涉及学习Gate的multi-edge表现。

## 判定边界

必须报告full success、agent success、collision、unresolved、timeout、平均步数、I占比、
切换次数、同场改善/退化和oracle收益恢复。50场只作扩大判断，不作论文最终显著性结论。

仅当A1或B2至少一个同时满足以下条件时，才允许读取后150场：

1. full success高于同场5A；
2. 改善场数多于退化场数；
3. timeout相对5A增加不超过`0.04`；
4. 平均步数不超过5A的`2.5x`；
5. I占比不超过`0.85`。

若两者均不满足，停止自然泛化主张并登记navigation-train multi-edge数据修订；若满足，
先根据G11-D2主准入结果而不是本pilot峰值确定主Gate，再只对冻结主Gate和必要基线读取
confirmation。sealed test仍保持封存。

实际Gazebo运行必须等待G11-D2七策略全部完成和归档；当前只允许构建、审计和提交协议。
