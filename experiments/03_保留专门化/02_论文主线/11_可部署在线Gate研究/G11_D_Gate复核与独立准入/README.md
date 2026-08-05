# G11-D Gate复核与独立准入

状态：`D1 passed / D2 registered and runnable`。登记日期：`2026-08-05`。

G11-C只证明B2的student-rollout聚合值得保留，没有证明训练稳定性或最终闭环准入。
本阶段保持两个Actor、detector、训练数据、模型结构和主seed checkpoint全部冻结。

## D1：B2训练seed复核

- 数据仍为A1冻结5A轨迹640场与B1 student轨迹640场；
- validation仍为A1内部120场，不读取导航test或sealed test；
- 主seed固定为`20260804`，复核seed预注册为`20260805-20260808`；
- 只训练8帧T1 Gate，CPU、2线程、`nice -n 15`，不启动Gazebo、不使用GPU；
- 不从复核seed中挑选新的主模型，也不根据复核结果修改阈值策略。

D1通过条件：4个复核seed中至少3个同时满足冻结overall/weak FPR上限，并且相对A1主seed
的F1下降不超过`0.03`、positive interval IoU下降不超过`0.04`。失败则先检查训练不稳定，
不得直接扩大闭环实验。通过只说明聚合训练可复现，不是导航成绩。

运行入口：

```bash
bash scripts/run_g11_d_seed_replication.sh
```

活动日志写入`logs/active/g11_d/`，完整结束后归档到
`logs/archive/diagnostic/g11_d/`。汇总写入本目录`local_data/seed_replication_summary.json`。

D1已完成，4个复核seed全部通过。F1为`0.84487 +/- 0.00132`，positive interval IoU为
`0.73141 +/- 0.00198`，overall FPR为`0.26147 +/- 0.00235`；主seed继续固定为
`20260804`，没有从复核seed中重新选模型。日志已归档到
`logs/archive/diagnostic/g11_d/`。

## D2：独立闭环准入

D1通过后才启动。D2必须先构建并冻结一套来自导航validation、与A1内部120场和G11-C
完全互斥的0-edge/edge-1 manifest。不得复用G11-C调参，不读取导航test。

D2固定200个独立场景并保持zero/edge-1各100场。由于显式排除旧G3 dense monitor后，
导航validation中full-path dense-zero只有35场，因此在读取任何D2策略结果前预注册配额：

| stratum | 场景数 |
| --- | ---: |
| standard-zero | `65` |
| dense-zero | `35` |
| standard-edge1 | `50` |
| dense-edge1 | `50` |

总指标之外必须逐层报告；不得用standard-zero较高占比掩盖dense-zero能力下降。构建器若
任一层不足登记配额必须失败，不能静默补入旧G3或test场景。

D2 manifest已经冻结：

```text
experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/
g11_d2_admission_v1/validation.json.gz
SHA-256 6250b941f127d550641a621d4253e17ea0770ff3c0cb94e6254e1f26b9f4978a
```

确定性重建哈希一致，完整性审计通过；与G11-A1、G11-C和旧G3的scenario ID重叠均为0。

至少比较：

| 方法 | 作用 |
| --- | --- |
| 5A always-on | 普通导航基线 |
| epoch-16 always-on | 条件Actor全程使用边界 |
| min-LiDAR规则Gate | 不区分机器人与静态障碍的可部署下界 |
| 历史G2-A Gate | 第一版可部署学习Gate基线 |
| A1 Gate | 无student聚合对照 |
| B2主seed Gate | 当前候选 |
| 2.0 m oracle | 不可部署上界 |

主指标为full success，并报告agent success、collision、unresolved、timeout、平均步数、
interaction Actor占比、切换次数、同场配对检验和oracle收益恢复率。准入同时要求：

1. 0-edge full success相对5A下降不超过3个百分点；
2. edge-1上B2的改善明确多于退化，并显著超过5A；
3. timeout不系统增加；
4. B2明确超过规则Gate；
5. full success达到`0.45`或恢复至少60%的同场oracle收益；
6. 不能以更高full success掩盖interaction Actor长期占用和明显效率退化。

D2固定运行协议：

- 一个完整重复，所有策略使用seed `20260809`和同一200场顺序；
- 顺序：`5A -> min-LiDAR -> old G2-A -> A1 -> B2 -> oracle -> epoch16`；
- min-LiDAR只读取本车20维LiDAR最小值，on/off为`2.0/2.2 m`、hold为3步；
- old G2-A沿用冻结的`0.44/0.34/hold=3`，不重新调参；
- A1和B2沿用G11-C冻结配置，oracle仍是不可部署`2.0 m`真值距离；
- CPU串行、固定步进服务、每个策略200场，中断只能从一致state恢复。

除导航准入外，B2还必须同时满足：平均步数不超过5A的`2.0x`、总体interaction Actor
占比不超过`0.75`、0-edge timeout相对5A增加不超过`0.02`。导航通过但效率失败时只能
进入有登记的效率修订，不能称为最终Gate。

运行入口与日志：

```bash
bash scripts/experiment.sh start gate-g11-d2-admission
bash scripts/experiment.sh status
```

活动日志固定为`logs/active/gate-g11-d2-admission/`，七组全部完成后自动归档到
`logs/archive/validation/g11_d2/`。启动前必须先提交本协议和运行器。
通过D2后才评估multi-edge；若multi-edge不足，可在读取sealed test前登记训练数据修订，
但必须取消single-to-multi零样本泛化表述。
